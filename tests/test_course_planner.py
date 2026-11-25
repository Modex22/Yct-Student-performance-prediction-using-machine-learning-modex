import pytest

from src.course_planner import calculate_course_gpa


def test_course_schemes_produce_credit_weighted_four_point_cgpa():
    summary = calculate_course_gpa([
        {
            "name": "Mathematics",
            "assessment_type": "with_practical",
            "credits": 3,
            "exam_score": 45,
            "test_score": 8,
            "assignment_score": 7,
            "practical_score": 10,
        },
        {
            "name": "Communication",
            "assessment_type": "without_practical",
            "credits": 1,
            "exam_score": 30,
            "test_score": 10,
            "assignment_score": 10,
        },
    ])

    assert [course["score"] for course in summary["courses"]] == [70, 50]
    assert [course["grade_point"] for course in summary["courses"]] == [4, 2]
    assert summary["cgpa"] == 3.5
    assert summary["total_credits"] == 4
    assert summary["average_score"] == 60


def test_invalid_component_mark_is_rejected():
    with pytest.raises(ValueError, match="between 0 and 60"):
        calculate_course_gpa([{
            "name": "Mathematics",
            "assessment_type": "with_practical",
            "credits": 3,
            "exam_score": 61,
            "test_score": 8,
            "assignment_score": 7,
            "practical_score": 10,
        }])


def test_zero_course_credits_are_rejected():
    with pytest.raises(ValueError, match="greater than zero"):
        calculate_course_gpa([{
            "name": "Mathematics",
            "assessment_type": "without_practical",
            "credits": 0,
            "exam_score": 30,
            "test_score": 10,
            "assignment_score": 10,
        }])


def test_quick_check_omits_grading_note_but_keeps_course_calculator():
    from app import app

    response = app.test_client().get("/quick-check")

    assert response.status_code == 200
    assert b"CGPA uses a credit-weighted 4.0 scale" not in response.data
    assert b"Calculate CGPA" in response.data


def test_forecast_endpoint_rejects_tampered_course_data_without_server_error():
    from app import app

    response = app.test_client().post(
        "/quick-check/courses/forecast",
        data={
            "course_count": "1",
            "course_name_0": "Mathematics",
            "assessment_type_0": "with_practical",
            "course_credits_0": "3",
            "exam_score_0": "61",
            "test_score_0": "8",
            "assignment_score_0": "7",
            "practical_score_0": "10",
        },
    )

    assert response.status_code == 400
    assert b"must be between 0 and 60" in response.data
