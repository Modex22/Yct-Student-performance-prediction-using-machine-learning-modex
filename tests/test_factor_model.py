"""Tests for the Kaggle factor-based performance prediction model."""
from src.factor_model import (
    clean_dataframe,
    get_factor_model,
    predict_factor_performance,
)
import pandas as pd


def test_clean_dataframe():
    raw_df = pd.DataFrame([{
        "Scholarship": None,
        "Attendance": "3",
        "Notes": "6",
        "Listening_in_Class": "6",
        "Weekly_Study_Hours": "12",
    }])
    cleaned = clean_dataframe(raw_df)
    assert cleaned.loc[0, "Scholarship"] == "0%"
    assert cleaned.loc[0, "Attendance"] == "Never"
    assert cleaned.loc[0, "Notes"] == "Yes"
    assert cleaned.loc[0, "Listening_in_Class"] == "Yes"
    assert cleaned.loc[0, "Weekly_Study_Hours"] == 12.0


def test_predict_factor_performance():
    sample_input = {
        "Student_Age": "19-22",
        "Sex": "Male",
        "High_School_Type": "State",
        "Scholarship": "50%",
        "Additional_Work": "No",
        "Sports_activity": "Yes",
        "Transportation": "Private",
        "Weekly_Study_Hours": "8",
        "Attendance": "Always",
        "Reading": "Yes",
        "Notes": "Yes",
        "Listening_in_Class": "Yes",
        "Project_work": "Yes",
    }
    result = predict_factor_performance(sample_input)
    assert "predicted_grade" in result
    assert result["predicted_grade"] in ["AA", "BA", "BB", "CB", "CC", "DC", "DD", "Fail"]
    assert "probabilities" in result
    assert len(result["probabilities"]) == 8
    assert "recommendations" in result
    assert len(result["recommendations"]) > 0
    assert "dataset_url" in result
