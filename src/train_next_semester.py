"""Train the semester-average forecast from paired course-result workbooks."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from openpyxl import load_workbook
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split

from src.config import NEXT_SEMESTER_MODEL_PATH

MINIMUM_TRAINING_STUDENTS = 10
CV_FOLDS = 5
CV_REPEATS = 10


def _read_course_scores(
    workbook_path: Path,
    *,
    observed_only: bool,
) -> tuple[dict[str, list[float]], Counter[str]]:
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        if "Course Results" not in workbook.sheetnames:
            raise ValueError(f"{workbook_path.name} must contain a 'Course Results' sheet.")

        rows = workbook["Course Results"].iter_rows(values_only=True)
        first_row = next(rows, None)
        if first_row is None:
            raise ValueError(f"{workbook_path.name} has an empty 'Course Results' sheet.")

        headers = [str(value).strip() if value is not None else "" for value in first_row]
        required = {"Student_ID", "Course_Code", "Score"}
        if observed_only:
            required.add("Data_Source")
        missing = required - set(headers)
        if missing:
            raise ValueError(
                f"{workbook_path.name} is missing required course-result columns: "
                f"{', '.join(sorted(missing))}."
            )

        student_column = headers.index("Student_ID")
        course_column = headers.index("Course_Code")
        score_column = headers.index("Score")
        source_column = headers.index("Data_Source") if observed_only else None
        scores: dict[str, list[float]] = defaultdict(list)
        source_counts: Counter[str] = Counter()
        seen_courses: set[tuple[str, str]] = set()

        for row_number, row in enumerate(rows, start=2):
            if all(value is None or str(value).strip() == "" for value in row):
                continue

            student_value = row[student_column]
            course_value = row[course_column]
            if student_value is None or course_value is None:
                raise ValueError(
                    f"{workbook_path.name} has an incomplete course-result row at row {row_number}."
                )
            student_id = str(student_value).strip()
            course_code = str(course_value).strip()
            if not student_id or not course_code:
                raise ValueError(
                    f"{workbook_path.name} has an incomplete course-result row at row {row_number}."
                )

            source = ""
            if source_column is not None:
                raw_source = row[source_column]
                source = str(raw_source).strip().lower() if raw_source is not None else ""
                if source not in {"observed", "estimated"}:
                    raise ValueError(
                        f"{workbook_path.name} has an unknown Data_Source at row {row_number}."
                    )
                source_counts[source] += 1
                if source != "observed":
                    continue

            raw_score = row[score_column]
            if (
                isinstance(raw_score, bool)
                or not isinstance(raw_score, (int, float))
                or not math.isfinite(float(raw_score))
                or not 0 <= float(raw_score) <= 100
            ):
                raise ValueError(
                    f"{workbook_path.name} has an invalid score at row {row_number}."
                )

            course_key = (student_id, course_code)
            if course_key in seen_courses:
                raise ValueError(
                    f"{workbook_path.name} repeats a student/course result at row {row_number}."
                )
            seen_courses.add(course_key)
            scores[student_id].append(float(raw_score))

        return dict(scores), source_counts
    finally:
        workbook.close()


def _benchmark_models(
    features: np.ndarray,
    targets: np.ndarray,
) -> list[dict[str, Any]]:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.2,
        random_state=42,
    )
    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=1,
            min_samples_leaf=5,
            random_state=42,
        ),
        "Mean baseline": DummyRegressor(strategy="mean"),
    }

    results = []
    for name, estimator in candidates.items():
        cv_results = cross_validate(
            estimator,
            features,
            targets,
            cv=RepeatedKFold(
                n_splits=CV_FOLDS,
                n_repeats=CV_REPEATS,
                random_state=42,
            ),
            scoring={
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
        )
        estimator.fit(x_train, y_train)
        holdout_predictions = estimator.predict(x_test)
        results.append({
            "model": name,
            "cv_mae": float(-cv_results["test_mae"].mean()),
            "cv_r2": float(cv_results["test_r2"].mean()),
            "cv_mae_folds": [
                float(value) for value in -cv_results["test_mae"]
            ],
            "cv_r2_folds": [
                float(value) for value in cv_results["test_r2"]
            ],
            "holdout_mae": float(mean_absolute_error(y_test, holdout_predictions)),
            "holdout_r2": float(r2_score(y_test, holdout_predictions)),
        })

    return sorted(results, key=lambda result: float(result["cv_mae"]))


def train_and_save(
    first_semester_path: Path,
    second_semester_path: Path,
    output_path: Path = Path(NEXT_SEMESTER_MODEL_PATH),
) -> dict[str, Any]:
    first_semester, source_counts = _read_course_scores(
        first_semester_path,
        observed_only=True,
    )
    second_semester, _ = _read_course_scores(
        second_semester_path,
        observed_only=False,
    )

    paired_students = sorted(first_semester.keys() & second_semester.keys())
    paired = [
        (
            float(np.mean(first_semester[student_id])),
            float(np.mean(second_semester[student_id])),
        )
        for student_id in paired_students
        if first_semester[student_id] and second_semester[student_id]
    ]
    if len(paired) < MINIMUM_TRAINING_STUDENTS:
        raise ValueError(
            f"At least {MINIMUM_TRAINING_STUDENTS} matched students with observed "
            "first-semester scores are required."
        )

    features = np.array([current for current, _ in paired], dtype=float).reshape(-1, 1)
    targets = np.array([following for _, following in paired], dtype=float)
    benchmark_results = _benchmark_models(features, targets)
    selected_model_name = str(benchmark_results[0]["model"])
    if selected_model_name == "Linear Regression":
        model = LinearRegression().fit(features, targets)
        intercept = float(model.intercept_)
        slope = float(model.coef_[0])
    elif selected_model_name == "Ridge":
        model = Ridge(alpha=1.0).fit(features, targets)
        intercept = float(model.intercept_)
        slope = float(model.coef_[0])
    elif selected_model_name == "Mean baseline":
        model = DummyRegressor(strategy="mean").fit(features, targets)
        intercept = float(model.constant_[0])
        slope = 0.0
    else:
        raise ValueError(
            f"{selected_model_name} had the lowest cross-validation MAE, but the "
            "forecast runtime currently supports linear model formulas only."
        )
    selected_metrics = benchmark_results[0]
    payload = {
        "target": "mean next-semester course final score (0-100)",
        "selected_model": selected_model_name,
        "selection_criterion": "lowest mean MAE across repeated 5-fold cross-validation",
        "intercept": intercept,
        "slope": slope,
        "training_students": len(paired),
        "first_semester_observed_course_scores": source_counts["observed"],
        "first_semester_estimated_course_scores_excluded": source_counts["estimated"],
        "holdout_r2": selected_metrics["holdout_r2"],
        "holdout_mae": selected_metrics["holdout_mae"],
        "cross_validation_method": (
            f"Repeated {CV_FOLDS}-fold cross-validation ({CV_REPEATS} repeats)"
        ),
        "cross_validation_r2": selected_metrics["cv_r2"],
        "cross_validation_mae": selected_metrics["cv_mae"],
        "benchmark_results": benchmark_results,
        "source": (
            "User-provided YABATECH semester workbooks; first-semester estimated "
            "scores excluded from input averages"
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_semester_workbook", type=Path)
    parser.add_argument("second_semester_workbook", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(NEXT_SEMESTER_MODEL_PATH),
        help="model JSON path (defaults to the app's active forecast model)",
    )
    args = parser.parse_args()

    model = train_and_save(
        args.first_semester_workbook,
        args.second_semester_workbook,
        args.output,
    )
    print(
        "Saved forecast model: "
        f"{model['selected_model']} selected from "
        f"{len(model['benchmark_results'])} models on {model['training_students']} matched students, "
        f"holdout MAE {model['holdout_mae']:.2f}, "
        f"repeated-CV MAE {model['cross_validation_mae']:.2f}."
    )


if __name__ == "__main__":
    main()
