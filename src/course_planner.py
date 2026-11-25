"""Course score, credit-weighted CGPA, and grade-point calculations."""
from __future__ import annotations

import math
from bisect import bisect_left
from typing import Any

from src.config import CLASSIFICATION_BINS, CLASSIFICATION_LABELS

MAX_COURSES = 20

ASSESSMENT_SCHEMES = {
    "with_practical": {
        "exam_score": 60,
        "test_score": 10,
        "assignment_score": 10,
        "practical_score": 20,
    },
    "without_practical": {
        "exam_score": 60,
        "test_score": 20,
        "assignment_score": 20,
    },
}


def _number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def calculate_course_gpa(courses: list[dict[str, Any]]) -> dict[str, Any]:
    if not courses:
        raise ValueError("Enter at least one course.")
    if len(courses) > MAX_COURSES:
        raise ValueError(f"Enter no more than {MAX_COURSES} courses.")

    calculated = []
    total_credits = 0.0
    weighted_grade_points = 0.0

    for index, course in enumerate(courses, start=1):
        name = str(course.get("name", "")).strip()
        if not name:
            raise ValueError(f"Enter a name for course {index}.")

        scheme = str(course.get("assessment_type", ""))
        component_maxima = ASSESSMENT_SCHEMES.get(scheme)
        if component_maxima is None:
            raise ValueError(f"Choose an assessment scheme for course {index}.")

        credits = _number(course.get("credits"), f"Credit units for course {index}")
        if credits <= 0:
            raise ValueError(f"Credit units for course {index} must be greater than zero.")

        normalized = {
            "name": name,
            "assessment_type": scheme,
            "credits": credits,
        }
        score = 0.0
        for field, maximum in component_maxima.items():
            component_score = _number(
                course.get(field),
                f"{field.replace('_', ' ').title()} for course {index}",
            )
            if not 0 <= component_score <= maximum:
                raise ValueError(
                    f"{field.replace('_', ' ').title()} for course {index} "
                    f"must be between 0 and {maximum}."
                )
            normalized[field] = component_score
            score += component_score

        grade_index = bisect_left(CLASSIFICATION_BINS[1:-1], score)
        classification = CLASSIFICATION_LABELS[grade_index]
        grade_point = float(grade_index)
        normalized.update({
            "score": round(score, 1),
            "classification": classification,
            "grade_point": grade_point,
            "weighted_grade_points": round(grade_point * credits, 2),
        })
        calculated.append(normalized)
        total_credits += credits
        weighted_grade_points += grade_point * credits

    return {
        "courses": calculated,
        "cgpa": round(weighted_grade_points / total_credits, 2),
        "total_credits": round(total_credits, 2),
        "average_score": round(
            sum(course["score"] for course in calculated) / len(calculated), 1
        ),
    }
