"""Forecast next-semester average marks from scanned current course finals."""
from __future__ import annotations

import json
import math
from typing import Any

from src.config import NEXT_SEMESTER_MODEL_PATH


def _load_model() -> dict[str, Any]:
    with open(NEXT_SEMESTER_MODEL_PATH, encoding="utf-8") as file:
        return json.load(file)


def forecast_holdout_mae() -> float:
    return float(_load_model()["holdout_mae"])


def build_forecast_remark(
    student_id: str,
    current_average: float,
    forecast_score: float,
    classification: str,
    error_margin: float,
) -> str:
    change = forecast_score - current_average
    if change > error_margin:
        trend = "improve"
    elif change < -error_margin:
        trend = "decline"
    else:
        trend = "maintain similar performance"

    next_step = {
        "Fail": "Targeted academic support is recommended.",
        "Pass": "Focus on strengthening course marks for more consistent results.",
        "Lower Credit": "Improving course consistency could help reach Upper Credit.",
        "Upper Credit": "Maintaining this progress could support a Distinction result.",
        "Distinction": "Sustained effort should help maintain this strong performance.",
    }[classification]
    return (
        f"Based on the available semester results, {student_id} is projected to "
        f"{trend} next semester, averaging about {forecast_score:.1f}/100 "
        f"({classification}). {next_step}"
    )


def forecast_next_semester(course_scores: list[float]) -> dict[str, float | str]:
    if not course_scores:
        raise ValueError("At least one course final score is required.")
    if any(not math.isfinite(score) or score < 0 or score > 100 for score in course_scores):
        raise ValueError("Course final scores must be between 0 and 100.")

    model = _load_model()
    current_average = sum(course_scores) / len(course_scores)
    forecast = model["intercept"] + model["slope"] * current_average
    forecast = max(0.0, min(100.0, forecast))

    if forecast >= 70:
        classification = "Distinction"
    elif forecast >= 60:
        classification = "Upper Credit"
    elif forecast >= 50:
        classification = "Lower Credit"
    elif forecast >= 45:
        classification = "Pass"
    else:
        classification = "Fail"

    return {
        "current_average": round(current_average, 1),
        "forecast_score": round(forecast, 1),
        "score_mae": round(float(model["holdout_mae"]), 1),
        "classification": classification,
    }
