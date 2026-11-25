"""Flask web app for the Yabatech Student Performance Predictor."""
from __future__ import annotations

import json
import hmac
import os
import csv

from flask import Flask, Response, redirect, render_template, request, send_from_directory, session, url_for

from src.batch import (
    BatchError,
    build_template_csv,
    is_course_roster,
    is_next_semester_forecast,
    parse_upload,
    run_batch,
    run_course_batch,
    run_next_semester_batch,
    validate_and_prepare,
)
from src.config import (
    FEATURE_COLUMNS,
    FIGURES_DIR,
    METRICS_PATH,
    NEXT_SEMESTER_MODEL_PATH,
)
from src.chapter_four_catalog import CHAPTER_FOUR_DIAGRAMS
from src.course_planner import MAX_COURSES, calculate_course_gpa
from src.factor_model import (
    FACTOR_FIELDS,
    FACTOR_METRICS_PATH,
    KAGGLE_DATASET_URL,
    predict_factor_performance,
)
from src.history import WrongPin, build_trend_chart, delete_history, get_history, save_entry
from src.history import delete_history_as_admin, get_admin_overview
from src.predict import predict_performance
from src.next_semester import build_forecast_remark, forecast_next_semester

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-this-secret")

FORM_FIELDS = [
    {"name": "exam_score", "label": "Exam score", "type": "number", "min": 0, "max": 60, "default": 36,
     "help": "Out of 60."},
    {"name": "test_score", "label": "Test / CA score", "type": "number", "min": 0, "max": 10, "default": 6,
     "help": "Out of 10."},
    {"name": "assignment_score", "label": "Assignment score", "type": "number", "min": 0, "max": 10, "default": 7,
     "help": "Out of 10."},
    {"name": "practical_score", "label": "Practical score", "type": "number", "min": 0, "max": 20, "default": 13,
     "help": "Out of 20."},
]

NUMERIC_FIELD_NAMES = {f["name"] for f in FORM_FIELDS if f["type"] == "number"}


def parse_form(form) -> dict:
    student = {}
    for field in FORM_FIELDS:
        raw = form.get(field["name"])
        if field["name"] in NUMERIC_FIELD_NAMES:
            student[field["name"]] = float(raw)
        else:
            student[field["name"]] = raw
    return student


@app.route("/", methods=["GET"])
def index():
    dataset_overview = {
        "total_students": 0,
        "graduate_count": 0,
        "dropout_count": 0,
        "graduate_percent": 0,
        "dropout_percent": 0,
        "features": [],
    }

    kaggle_path = os.path.join(os.path.dirname(__file__), "data", "kaggle_student_performance.csv")
    try:
        with open(kaggle_path, newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))

        graduate_grades = {"AA", "BA", "BB", "CC", "CB"}
        graduate_count = sum(row["Grade"] in graduate_grades for row in rows)
        dropout_count = len(rows) - graduate_count
        total = len(rows)

        feature_fields = [
            ("High school type", "High_School_Type"),
            ("Scholarship", "Scholarship"),
            ("Additional work", "Additional_Work"),
            ("Sports activity", "Sports_activity"),
            ("Attendance", "Attendance"),
            ("Study hours", "Weekly_Study_Hours"),
            ("Reading", "Reading"),
        ]
        features = []
        for label, field in feature_fields:
            counts = {}
            for row in rows:
                value = row[field].strip()
                counts[value] = counts.get(value, 0) + 1
            features.append({
                "label": label,
                "count": sum(counts.values()),
                "categories": sorted(counts.items(), key=lambda item: item[1], reverse=True)[:3],
            })

        dataset_overview = {
            "total_students": total,
            "graduate_count": graduate_count,
            "dropout_count": dropout_count,
            "graduate_percent": round((graduate_count / total) * 100, 1) if total else 0,
            "dropout_percent": round((dropout_count / total) * 100, 1) if total else 0,
            "features": features,
        }
    except (FileNotFoundError, KeyError, OSError):
        pass

    return render_template("landing.html", dataset_overview=dataset_overview)


@app.route("/batch", methods=["GET"])
def batch_upload_page():
    return render_template("batch_upload.html")


@app.route("/batch/template", methods=["GET"])
def batch_template():
    csv_text = build_template_csv()
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=yabatech_roster_template.csv"},
    )


@app.route("/batch/predict", methods=["POST"])
def batch_predict():
    upload = request.files.get("roster")
    if upload is None or upload.filename == "":
        return render_template("batch_error.html", message="No file was uploaded."), 400

    try:
        df = parse_upload(upload.filename, upload.read())
        df, identity_map, missing = validate_and_prepare(df)
    except BatchError as exc:
        return render_template("batch_error.html", message=str(exc)), 400

    if missing:
        return render_template(
            "batch_error.html",
            message=f"The file is missing {len(missing)} required column(s): {', '.join(missing)}.",
            missing=missing,
            all_columns=FEATURE_COLUMNS,
        ), 400

    if is_next_semester_forecast(df):
        outcome = run_next_semester_batch(df)
    elif is_course_roster(df):
        outcome = run_course_batch(df, identity_map)
    else:
        outcome = run_batch(df, identity_map)
    return render_template("batch_results_fragment.html", outcome=outcome)


@app.route("/quick-check", methods=["GET"])
def quick_check():
    return render_template(
        "index.html",
        fields=FORM_FIELDS,
        course_count=1,
        course_input_rows=[],
    )


def _course_rows_from_form(form) -> list[dict]:
    try:
        course_count = int(form.get("course_count", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError("Choose a whole number of courses.") from exc
    if not 1 <= course_count <= MAX_COURSES:
        raise ValueError(f"Choose between 1 and {MAX_COURSES} courses.")

    return [
        {
            "name": form.get(f"course_name_{index}", ""),
            "assessment_type": form.get(f"assessment_type_{index}", ""),
            "credits": form.get(f"course_credits_{index}", ""),
            "exam_score": form.get(f"exam_score_{index}", ""),
            "test_score": form.get(f"test_score_{index}", ""),
            "assignment_score": form.get(f"assignment_score_{index}", ""),
            "practical_score": form.get(f"practical_score_{index}", ""),
        }
        for index in range(course_count)
    ]


def _course_input_rows_from_form(form, course_count: int) -> list[dict]:
    if not 1 <= course_count <= MAX_COURSES:
        return []
    return [
        {
            "name": form.get(f"course_name_{index}", ""),
            "assessment_type": form.get(f"assessment_type_{index}", "with_practical"),
            "credits": form.get(f"course_credits_{index}", ""),
            "exam_score": form.get(f"exam_score_{index}", ""),
            "test_score": form.get(f"test_score_{index}", ""),
            "assignment_score": form.get(f"assignment_score_{index}", ""),
            "practical_score": form.get(f"practical_score_{index}", ""),
        }
        for index in range(course_count)
    ]


@app.route("/quick-check/courses/calculate", methods=["POST"])
def quick_check_courses():
    raw_count = request.form.get("course_count", "1")
    try:
        rows = _course_rows_from_form(request.form)
        summary = calculate_course_gpa(rows)
    except ValueError as exc:
        try:
            course_count = int(raw_count)
        except ValueError:
            course_count = 1
        return render_template(
            "index.html",
            fields=FORM_FIELDS,
            course_count=raw_count,
            course_input_rows=_course_input_rows_from_form(request.form, course_count),
            course_error=str(exc),
        ), 400

    return render_template("course_result.html", summary=summary, forecast=None, advice=None)


@app.route("/quick-check/courses/forecast", methods=["POST"])
def quick_check_courses_forecast():
    raw_count = request.form.get("course_count", "1")
    try:
        rows = _course_rows_from_form(request.form)
        summary = calculate_course_gpa(rows)
    except ValueError as exc:
        try:
            course_count = int(raw_count)
        except ValueError:
            course_count = 1
        return render_template(
            "index.html",
            fields=FORM_FIELDS,
            course_count=raw_count,
            course_input_rows=_course_input_rows_from_form(request.form, course_count),
            course_error=str(exc),
        ), 400

    forecast = forecast_next_semester([course["score"] for course in summary["courses"]])
    advice = build_forecast_remark(
        "your current course average",
        forecast["current_average"],
        forecast["forecast_score"],
        forecast["classification"],
        forecast["score_mae"],
    )
    return render_template(
        "course_result.html",
        summary=summary,
        forecast=forecast,
        advice=advice,
    )


@app.route("/quick-check/predict", methods=["POST"])
def quick_check_predict():
    student = parse_form(request.form)
    result = predict_performance(student)

    save_to_history = request.form.get("save_to_history") == "on"
    matric_no = (request.form.get("matric_no") or "").strip()
    pin = request.form.get("pin") or ""
    saved = False
    save_error = None
    trend = None

    if save_to_history:
        if not matric_no:
            save_error = "Matric number is required to save to history."
        elif not pin or len(pin) < 4:
            save_error = "A PIN of at least 4 characters is required to save to history."
        else:
            try:
                save_entry(matric_no, pin, student, result)
                saved = True
                trend = build_trend_chart(get_history(matric_no, pin))
            except WrongPin:
                save_error = (
                    f"That PIN doesn't match the one already set for {matric_no}. "
                    "If this is your first time saving, double-check the matric number for typos."
                )

    return render_template(
        "result.html",
        student=student,
        fields=FORM_FIELDS,
        result=result,
        matric_no=matric_no,
        saved=saved,
        save_error=save_error,
        trend=trend,
    )


@app.route("/history", methods=["GET"])
def history_lookup():
    return render_template("history_lookup.html")


@app.route("/history", methods=["POST"])
def history_lookup_submit():
    matric_no = (request.form.get("matric_no") or "").strip()
    pin = request.form.get("pin") or ""
    if not matric_no or not pin:
        return render_template("history_lookup.html", error="Enter both a matric number and its PIN.")

    try:
        entries = get_history(matric_no, pin)
    except WrongPin:
        return render_template("history_lookup.html", error="Incorrect PIN for that matric number.")

    if not entries:
        return render_template(
            "history_lookup.html",
            error=f"No saved history for {matric_no} yet — nothing to unlock with a PIN.",
        )

    trend = build_trend_chart(entries)
    return render_template("history_view.html", matric_no=matric_no, pin=pin, entries=entries, trend=trend)


@app.route("/history/delete", methods=["POST"])
def history_delete():
    matric_no = (request.form.get("matric_no") or "").strip()
    pin = request.form.get("pin") or ""
    try:
        delete_history(matric_no, pin)
        return render_template(
            "history_lookup.html",
            notice=f"History for {matric_no} was deleted.",
        )
    except WrongPin:
        return render_template("history_lookup.html", error="Incorrect PIN — nothing was deleted.")


@app.route("/factors", methods=["GET"])
def factor_check():
    return render_template("factor_check.html", fields=FACTOR_FIELDS)


@app.route("/factors/predict", methods=["POST"])
def factor_check_predict():
    inputs = {k: request.form.get(k) for k in request.form}
    result = predict_factor_performance(inputs)
    return render_template("factor_result.html", result=result)


@app.route("/about", methods=["GET"])
def about():
    metrics = {}
    factor_metrics = {}
    forecast_model = {}
    try:
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    except FileNotFoundError:
        pass
    try:
        with open(FACTOR_METRICS_PATH) as f:
            factor_metrics = json.load(f)
    except FileNotFoundError:
        pass
    try:
        with open(NEXT_SEMESTER_MODEL_PATH) as f:
            forecast_model = json.load(f)
    except FileNotFoundError:
        pass
    return render_template(
        "about.html",
        metrics=metrics,
        factor_metrics=factor_metrics,
        forecast_model=forecast_model,
        chapter_four_diagrams=CHAPTER_FOUR_DIAGRAMS,
        kaggle_url=KAGGLE_DATASET_URL,
    )


@app.route("/figures/<path:filename>", methods=["GET"])
def figures(filename):
    return send_from_directory(FIGURES_DIR, filename)


def _admin_logged_in() -> bool:
    return session.get("admin_logged_in") is True


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if _admin_logged_in():
        return redirect(url_for("admin_dashboard"))

    error = None
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        expected_username = os.environ.get("ADMIN_USERNAME", "admin")
        expected_password = os.environ.get("ADMIN_PASSWORD", "admin123")
        if (
            hmac.compare_digest(username, expected_username)
            and hmac.compare_digest(password, expected_password)
        ):
            session["admin_logged_in"] = True
            session["admin_username"] = username
            return redirect(url_for("admin_dashboard"))
        error = "Incorrect admin username or password."

    return render_template("admin_login.html", error=error)


@app.route("/admin", methods=["GET"])
def admin_dashboard():
    if not _admin_logged_in():
        return redirect(url_for("admin_login"))

    search = (request.args.get("q") or "").strip()
    overview = get_admin_overview(search)
    return render_template(
        "admin_dashboard.html",
        overview=overview,
        search=search,
        admin_username=session.get("admin_username", "Admin"),
    )


@app.route("/admin/student/delete", methods=["POST"])
def admin_delete_student():
    if not _admin_logged_in():
        return redirect(url_for("admin_login"))

    matric_no = (request.form.get("matric_no") or "").strip()
    if matric_no:
        delete_history_as_admin(matric_no)
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/logout", methods=["POST"])
def admin_logout():
    session.pop("admin_logged_in", None)
    session.pop("admin_username", None)
    return redirect(url_for("admin_login"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
