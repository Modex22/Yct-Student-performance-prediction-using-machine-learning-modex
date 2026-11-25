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
import pandas as pd
from src.predict import build_counsellor_note


def test_template_csv_round_trips_through_the_full_pipeline():
    csv_text = build_template_csv()
    df = parse_upload("template.csv", csv_text.encode("utf-8"))
    df, identity_map, missing = validate_and_prepare(df)

    assert missing == []
    assert set(identity_map) == {"name", "matric_no", "department", "level"}

    outcome = run_batch(df, identity_map)
    assert outcome["summary"]["n_students"] == 3
    assert outcome["warnings"] == []
    for row in outcome["results"]:
        assert 0 <= row["predicted_score"] <= 100
        assert row["classification"] in {"Fail", "Pass", "Lower Credit", "Upper Credit", "Distinction"}
        assert row["counsellor_note"]


def test_missing_feature_columns_are_reported():
    df = parse_upload("bad.csv", b"name,age\nJohn,18\n")
    df, identity_map, missing = validate_and_prepare(df)
    assert "exam_score" in missing
    assert len(missing) > 0


def test_unsupported_file_type_raises_batch_error():
    try:
        parse_upload("notes.txt", b"hello")
        assert False, "expected BatchError"
    except BatchError:
        pass


def test_non_numeric_row_is_skipped_with_a_warning():
    csv_text = build_template_csv().replace(",47,8,9,16", ",not-a-number,8,9,16")

    parsed = parse_upload("roster.csv", csv_text.encode("utf-8"))
    parsed, identity_map, missing = validate_and_prepare(parsed)
    outcome = run_batch(parsed, identity_map)

    assert outcome["summary"]["n_students"] == 2  # one row skipped
    assert any("not-a-number" in w for w in outcome["warnings"])


def test_out_of_range_value_is_flagged_but_still_processed():
    """Catches the likely real mistake: someone enters exam_score out of
    100 instead of out of 60 (its actual max)."""
    csv_text = build_template_csv().replace(",47,8,9,16", ",95,8,9,16")

    parsed = parse_upload("roster.csv", csv_text.encode("utf-8"))
    parsed, identity_map, missing = validate_and_prepare(parsed)
    outcome = run_batch(parsed, identity_map)

    assert outcome["summary"]["n_students"] == 3  # still processed, not skipped
    assert any("outside the expected 0-60 range" in w for w in outcome["warnings"])


def test_counsellor_note_mentions_the_driving_factor_for_at_risk_students():
    student = {"exam_score": 10, "test_score": 8, "assignment_score": 8, "practical_score": 16}
    note = build_counsellor_note(student, "At Risk")
    assert "exam performance" in note.lower()


def test_counsellor_note_is_positive_for_excellent_category():
    note = build_counsellor_note({}, "Excellent")
    assert "outstanding" in note.lower()


def test_course_final_scores_are_processed_directly():
    df = pd.DataFrame([
        {"student_id": "STU999", "course_code": "COM 411", "estimated_score": 70},
        {"student_id": "STU999", "course_code": "COM 412", "estimated_score": 50},
    ])
    df.columns = [column.lower().replace(" ", "_") for column in df.columns]
    df, identity_map, missing = validate_and_prepare(df)

    assert is_course_roster(df)
    assert missing == []
    outcome = run_course_batch(df, identity_map)

    assert outcome["mode"] == "course"
    assert outcome["summary"]["n_students"] == 1
    assert outcome["results"][0]["average_score"] == 60
    assert outcome["results"][0]["classification"] == "Upper Credit"
    assert len(outcome["results"][0]["courses"]) == 2


def test_course_summary_needs_only_present_final_score_columns():
    df = pd.DataFrame([{
        "student_id": "STU100",
        "com_411_score": 75,
        "com_412_score": 55,
    }])
    df, identity_map, missing = validate_and_prepare(df)
    outcome = run_course_batch(df, identity_map)

    assert missing == []
    assert outcome["summary"]["n_students"] == 1
    assert outcome["results"][0]["average_score"] == 65
    assert len(outcome["results"][0]["courses"]) == 2


def test_scanned_marks_get_ordered_student_codes_and_next_semester_forecasts():
    df = pd.DataFrame([
        {
            "forecast_next_semester": "yes",
            "student_id": "",
            "course_01_score": 50,
            "course_02_score": 70,
        },
        {
            "forecast_next_semester": "yes",
            "student_id": "",
            "course_01_score": 80,
            "course_02_score": 90,
        },
    ])

    assert is_next_semester_forecast(df)
    outcome = run_next_semester_batch(df)

    assert [row["student_id"] for row in outcome["results"]] == ["SST001", "SST002"]
    assert outcome["results"][0]["current_average"] == 60
    assert outcome["results"][0]["forecast_score"] == 55.4
    assert outcome["results"][0]["score_mae"] == 4.0
    assert "decline next semester" in outcome["results"][0]["remark"]
    assert "Lower Credit" in outcome["results"][0]["remark"]
    assert outcome["summary"]["n_students"] == 2
