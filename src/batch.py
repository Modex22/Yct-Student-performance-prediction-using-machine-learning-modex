"""
Parse an uploaded roster spreadsheet (CSV or XLSX), run every valid row
through predict_performance(), and summarize the results as a class-level
report: risk-tier breakdown, a score histogram, and a ranked at-risk list.
"""
from __future__ import annotations

import io
from typing import Any

import pandas as pd

from src.config import (
    CATEGORY_VALUES,
    CLASSIFICATION_LABELS,
    COMPONENT_MAX,
    FEATURE_COLUMNS,
    IDENTITY_COLUMN_ALIASES,
    MAX_BATCH_ROWS,
    NUMERIC_FEATURES,
)
from src.predict import predict_performance
from src.next_semester import (
    build_forecast_remark,
    forecast_holdout_mae,
    forecast_next_semester,
)


class BatchError(Exception):
    """A problem with the uploaded file that stops processing entirely."""


def _normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def parse_upload(filename: str, file_bytes: bytes) -> pd.DataFrame:
    lower = filename.lower()
    try:
        if lower.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes), keep_default_na=False, na_values=[""])
        elif lower.endswith((".xlsx", ".xlsm")):
            workbook = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
            sheet_name = "Course Results" if "Course Results" in workbook.sheet_names else (
                "Student Summary" if "Student Summary" in workbook.sheet_names else 0
            )
            df = pd.read_excel(
                workbook,
                sheet_name=sheet_name,
                engine="openpyxl",
                keep_default_na=False,
                na_values=[""],
            )
        else:
            raise BatchError("Unsupported file type. Please upload a .csv or .xlsx file.")
    except BatchError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface parser errors to the user
        raise BatchError(f"Could not read the file: {exc}") from exc

    if df.empty:
        raise BatchError("The uploaded file has no rows.")

    df.columns = [_normalize_column_name(c) for c in df.columns]
    return df


def _map_identity_columns(columns: list[str]) -> dict[str, str]:
    """Return {canonical_identity_name: actual_column_name} for whichever
    identity columns are present in the upload."""
    found = {}
    for canonical, aliases in IDENTITY_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in columns:
                found[canonical] = alias
                break
    return found


def validate_and_prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Check the required feature columns are present. Returns
    (dataframe, identity_column_map, missing_feature_columns)."""
    columns = list(df.columns)
    identity_map = _map_identity_columns(columns)
    if is_course_roster(df):
        return df, identity_map, []
    missing = [c for c in FEATURE_COLUMNS if c not in columns]
    return df, identity_map, missing


def _clean_row(row: pd.Series, row_num: int) -> tuple[dict[str, Any] | None, list[str]]:
    """Coerce one spreadsheet row into a valid student dict. Returns
    (student_dict_or_None, warnings). student is None if a required
    numeric field could not be parsed at all (row is skipped)."""
    warnings: list[str] = []
    student: dict[str, Any] = {}

    for field in NUMERIC_FEATURES:
        raw = row.get(field, "")
        value = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
        if pd.isna(value):
            warnings.append(f"Row {row_num}: '{field}' value '{raw}' is not a number — row skipped.")
            return None, warnings
        value = float(value)
        max_mark = COMPONENT_MAX.get(field)
        if max_mark is not None and not (0 <= value <= max_mark):
            warnings.append(
                f"Row {row_num}: '{field}' value {value:g} is outside the expected 0-{max_mark} range "
                "— check this wasn't entered on a different scale (e.g. out of 100). Still processed as given."
            )
        student[field] = value

    for field, allowed in CATEGORY_VALUES.items():
        raw = str(row.get(field, "")).strip()
        match = next((a for a in allowed if a.lower() == raw.lower()), None)
        if match is None:
            warnings.append(
                f"Row {row_num}: unrecognized '{field}' value '{raw}' (expected one of {', '.join(allowed)}) "
                "— treated as unknown by the model."
            )
            student[field] = raw
        else:
            student[field] = match

    return student, warnings


def is_course_roster(df: pd.DataFrame) -> bool:
    columns = set(df.columns)
    score_columns = {"estimated_score", "final_score", "course_final_score"}
    if "course_code" in columns and score_columns.intersection(columns):
        return True
    course_score_columns = [
        column for column in columns
        if column.endswith("_score") and column not in FEATURE_COLUMNS
    ]
    return bool(course_score_columns) and bool({"student_id", "matric_no", "matric_number"} & columns)


def is_next_semester_forecast(df: pd.DataFrame) -> bool:
    return "forecast_next_semester" in df.columns


def run_next_semester_batch(df: pd.DataFrame) -> dict[str, Any]:
    truncated = len(df) > MAX_BATCH_ROWS
    input_rows = len(df)
    df = df.head(MAX_BATCH_ROWS)
    score_columns = sorted(
        [
            column for column in df.columns
            if column.endswith("_score") and column not in FEATURE_COLUMNS
        ],
        key=lambda column: [
            int(part) if part.isdigit() else part
            for part in column.removesuffix("_score").split("_")
        ],
    )
    results = []
    warnings = []

    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        marks = []
        for column in score_columns:
            raw = row.get(column)
            if pd.isna(raw) or str(raw).strip() == "":
                continue
            value = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
            if pd.isna(value) or not 0 <= float(value) <= 100:
                warnings.append(
                    f"Row {row_number}: '{column}' is not a valid 0-100 course score and was ignored."
                )
                continue
            marks.append(float(value))

        if not marks:
            warnings.append(f"Row {row_number}: no readable course final scores — row skipped.")
            continue

        forecast = forecast_next_semester(marks)
        student_id = str(row.get("student_id", "")).strip() or f"SST{row_number:03d}"
        remark = build_forecast_remark(
            student_id,
            forecast["current_average"],
            forecast["forecast_score"],
            forecast["classification"],
            forecast["score_mae"],
        )
        results.append({
            "row": row_number,
            "display_name": student_id,
            "student_id": student_id,
            "course_count": len(marks),
            "remark": remark,
            **forecast,
        })

    n = len(results)
    scores = [row["forecast_score"] for row in results]
    classification_counts = {
        label: sum(row["classification"] == label for row in results)
        for label in CLASSIFICATION_LABELS
    }
    summary = {
        "n_students": n,
        "avg_current_score": round(
            sum(row["current_average"] for row in results) / n, 1
        ) if n else 0.0,
        "avg_forecast_score": round(sum(scores) / n, 1) if n else 0.0,
        "score_mae": round(forecast_holdout_mae(), 1),
        "classification_counts": classification_counts,
    }
    return {
        "mode": "forecast",
        "results": results,
        "summary": summary,
        "warnings": warnings,
        "truncated": truncated,
        "n_input_rows": input_rows,
    }


def run_course_batch(df: pd.DataFrame, identity_map: dict[str, str]) -> dict[str, Any]:
    is_long_format = "course_code" in df.columns
    row_limit = MAX_BATCH_ROWS * 20 if is_long_format else MAX_BATCH_ROWS
    input_rows = len(df)
    truncated = input_rows > row_limit
    df = df.head(row_limit)
    warnings = []
    students: dict[str, dict[str, Any]] = {}

    if is_long_format:
        student_column = identity_map.get("matric_no", "student_id")
        score_column = next(
            column for column in ("estimated_score", "final_score", "course_final_score")
            if column in df.columns
        )
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            student_id = str(row.get(student_column, "")).strip()
            course = str(row.get("course_code", "")).strip()
            score = pd.to_numeric(pd.Series([row.get(score_column)]), errors="coerce").iloc[0]
            if not student_id or not course or pd.isna(score):
                warnings.append(f"Row {i}: missing student, course, or numeric final score — row skipped.")
                continue
            if not 0 <= float(score) <= 100:
                warnings.append(f"Row {i}: '{course}' final score {float(score):g} is outside 0-100 — row skipped.")
                continue
            entry = students.setdefault(student_id, {"identity": {"matric_no": student_id}, "courses": []})
            entry["courses"].append({"course": course, "score": float(score)})
    else:
        score_columns = [
            column for column in df.columns
            if column.endswith("_score") and column not in FEATURE_COLUMNS
        ]
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            identity = {
                canonical: str(row[col]).strip()
                for canonical, col in identity_map.items()
                if str(row.get(col, "")).strip()
            }
            student_id = identity.get("matric_no") or identity.get("name") or f"Student {i}"
            courses = []
            for column in score_columns:
                value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
                if pd.isna(value):
                    continue
                score = float(value)
                if not 0 <= score <= 100:
                    warnings.append(f"Row {i}: '{column}' final score {score:g} is outside 0-100 — ignored.")
                    continue
                course = column[:-6].replace("_", " ").upper()
                courses.append({"course": course, "score": score})
            if not courses:
                warnings.append(f"Row {i}: no numeric course final scores — row skipped.")
                continue
            students[student_id] = {"identity": identity, "courses": courses}

    results = []
    for i, (student_id, entry) in enumerate(students.items(), start=1):
        courses = entry["courses"]
        average_score = sum(course["score"] for course in courses) / len(courses)
        if average_score >= 70:
            classification = "Distinction"
        elif average_score >= 60:
            classification = "Upper Credit"
        elif average_score >= 50:
            classification = "Lower Credit"
        elif average_score >= 45:
            classification = "Pass"
        else:
            classification = "Fail"
        results.append({
            "row": i,
            "identity": entry["identity"],
            "display_name": entry["identity"].get("name") or student_id,
            "average_score": round(average_score, 1),
            "classification": classification,
            "courses": sorted(courses, key=lambda course: course["course"]),
        })

    n = len(results)
    if n:
        scores = [r["average_score"] for r in results]
        avg_score = sum(scores) / n
        classification_counts = {
            label: sum(1 for r in results if r["classification"] == label)
            for label in ["Fail", "Pass", "Lower Credit", "Upper Credit", "Distinction"]
        }
        at_risk = sorted(
            [r for r in results if r["classification"] in {"Fail", "Pass"}],
            key=lambda r: r["average_score"],
        )
        summary = {
            "n_students": n,
            "avg_score": round(avg_score, 1),
            "at_risk_count": len(at_risk),
            "classification_counts": classification_counts,
            "at_risk_list": at_risk,
        }
    else:
        summary = {"n_students": 0, "avg_score": 0.0, "at_risk_count": 0,
                   "classification_counts": {}, "at_risk_list": []}

    return {
        "mode": "course",
        "results": results,
        "summary": summary,
        "warnings": warnings,
        "truncated": truncated,
        "n_input_rows": input_rows,
    }


def run_batch(df: pd.DataFrame, identity_map: dict[str, str]) -> dict[str, Any]:
    truncated = len(df) > MAX_BATCH_ROWS
    df = df.head(MAX_BATCH_ROWS)

    results = []
    warnings: list[str] = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        student, row_warnings = _clean_row(row, i)
        warnings.extend(row_warnings)
        if student is None:
            continue

        prediction = predict_performance(student)
        identity = {
            canonical: str(row[col]).strip()
            for canonical, col in identity_map.items()
            if str(row.get(col, "")).strip()
        }
        results.append({
            "row": i,
            "identity": identity,
            "display_name": identity.get("name") or identity.get("matric_no") or f"Student {i}",
            "student": student,
            **prediction,
        })

    summary = summarize(results)
    return {
        "results": results,
        "summary": summary,
        "warnings": warnings,
        "truncated": truncated,
        "n_input_rows": len(df),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"n_students": 0}

    scores = [r["predicted_score"] for r in results]
    avg_score = sum(scores) / n
    sorted_scores = sorted(scores)
    mid = n // 2
    median_score = (
        sorted_scores[mid] if n % 2 == 1 else (sorted_scores[mid - 1] + sorted_scores[mid]) / 2
    )
    pass_count = sum(1 for r in results if r["pass_fail"] == "Pass")

    classification_counts = {label: 0 for label in CLASSIFICATION_LABELS}
    for r in results:
        classification_counts[r["classification"]] += 1

    histogram = [{"range": f"{b}-{b + 9}", "count": 0} for b in range(0, 100, 10)]
    for score in scores:
        idx = min(int(score // 10), 9)
        histogram[idx]["count"] += 1

    at_risk = sorted(
        [r for r in results if r["performance_category"] == "At Risk"],
        key=lambda r: r["predicted_score"],
    )

    return {
        "n_students": n,
        "avg_score": round(avg_score, 1),
        "median_score": round(median_score, 1),
        "pass_rate": round(pass_count / n * 100, 1),
        "at_risk_count": len(at_risk),
        "classification_counts": classification_counts,
        "histogram": histogram,
        "at_risk_list": at_risk,
    }


def build_template_csv() -> str:
    """A ready-to-fill CSV: identity columns + every model feature, with
    three illustrative example rows."""
    columns = ["name", "matric_no", "department", "level"] + FEATURE_COLUMNS
    rows = [
        {
            # exam_score /60, test_score /10, assignment_score /10, practical_score /20
            "name": "Adaeze Okafor", "matric_no": "ND/CS/23/0142", "department": "Computer Science", "level": "ND2",
            "exam_score": 47, "test_score": 8, "assignment_score": 9, "practical_score": 16,
        },
        {
            "name": "Emeka Chukwu", "matric_no": "HND/EEE/22/0088", "department": "Electrical Engineering", "level": "HND1",
            "exam_score": 19, "test_score": 4, "assignment_score": 5, "practical_score": 8,
        },
        {
            "name": "Fatima Bello", "matric_no": "ND/MC/23/0207", "department": "Mass Communication", "level": "ND1",
            "exam_score": 33, "test_score": 6, "assignment_score": 7, "practical_score": 13,
        },
    ]
    out = io.StringIO()
    pd.DataFrame(rows, columns=columns).to_csv(out, index=False)
    return out.getvalue()
