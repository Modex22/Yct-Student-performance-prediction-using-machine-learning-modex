"""
Model training and prediction pipeline for student performance based on
demographic, behavioral, and engagement factors.

Dataset source:
Kaggle: Student Performance Prediction Dataset (by Prajwal Kanade)
https://www.kaggle.com/datasets/prajwalkanade/student-performance-prediction-dataset
"""
from __future__ import annotations

import json
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import BASE_DIR, MODELS_DIR

KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/prajwalkanade/student-performance-prediction-dataset"
KAGGLE_DATA_PATH = os.path.join(BASE_DIR, "data", "kaggle_student_performance.csv")
FACTOR_MODEL_PATH = os.path.join(MODELS_DIR, "factor_classifier.joblib")
FACTOR_METRICS_PATH = os.path.join(MODELS_DIR, "factor_metrics.json")

FEATURE_COLUMNS = [
    "Student_Age",
    "Sex",
    "High_School_Type",
    "Scholarship",
    "Additional_Work",
    "Sports_activity",
    "Transportation",
    "Weekly_Study_Hours",
    "Attendance",
    "Reading",
    "Notes",
    "Listening_in_Class",
    "Project_work",
]

CATEGORICAL_COLS = [c for c in FEATURE_COLUMNS if c != "Weekly_Study_Hours"]
NUMERIC_COLS = ["Weekly_Study_Hours"]

GRADE_INFO = {
    "AA": {"title": "Excellent / Distinction", "gpa": "4.0", "badge": "badge-distinction", "status": "Outstanding"},
    "BA": {"title": "Very Good / Upper Credit", "gpa": "3.5", "badge": "badge-upper", "status": "Strong"},
    "BB": {"title": "Good / Upper Credit", "gpa": "3.0", "badge": "badge-upper", "status": "Good Standing"},
    "CB": {"title": "Above Average / Lower Credit", "gpa": "2.5", "badge": "badge-lower", "status": "Satisfactory"},
    "CC": {"title": "Average / Pass", "gpa": "2.0", "badge": "badge-pass", "status": "Average"},
    "DC": {"title": "Marginal Pass", "gpa": "1.5", "badge": "badge-pass", "status": "Needs Improvement"},
    "DD": {"title": "Conditional Pass / At Risk", "gpa": "1.0", "badge": "badge-risk", "status": "Academic Warning"},
    "Fail": {"title": "Fail", "gpa": "0.0", "badge": "badge-fail", "status": "Academic Difficulty"},
}

FACTOR_FIELDS = [
    {
        "name": "Student_Age",
        "label": "Student Age",
        "group": "Demographics & Background",
        "type": "select",
        "options": ["18", "19-22", "23-27"],
        "default": "19-22",
        "help": "Age group at enrollment.",
    },
    {
        "name": "Sex",
        "label": "Gender",
        "group": "Demographics & Background",
        "type": "select",
        "options": ["Male", "Female"],
        "default": "Male",
        "help": "Student gender.",
    },
    {
        "name": "High_School_Type",
        "label": "Graduated High-School Type",
        "group": "Demographics & Background",
        "type": "select",
        "options": ["State", "Private", "Other"],
        "default": "State",
        "help": "Type of secondary institution attended.",
    },
    {
        "name": "Scholarship",
        "label": "Scholarship / Aid",
        "group": "Demographics & Background",
        "type": "select",
        "options": ["None", "25%", "50%", "75%", "100%"],
        "default": "50%",
        "help": "Tuition scholarship or financial support percentage.",
    },
    {
        "name": "Transportation",
        "label": "Mode of Transportation",
        "group": "Demographics & Background",
        "type": "select",
        "options": ["Private", "Bus"],
        "default": "Bus",
        "help": "Primary daily commute method to campus.",
    },
    {
        "name": "Additional_Work",
        "label": "Additional Work / Part-time Job",
        "group": "Lifestyle & Commitments",
        "type": "select",
        "options": ["No", "Yes"],
        "default": "No",
        "help": "Employment or jobs outside coursework.",
    },
    {
        "name": "Sports_activity",
        "label": "Sports Activity",
        "group": "Lifestyle & Commitments",
        "type": "select",
        "options": ["No", "Yes"],
        "default": "Yes",
        "help": "Regular participation in athletic/sports activities.",
    },
    {
        "name": "Weekly_Study_Hours",
        "label": "Weekly Study Hours",
        "group": "Study Habits & Engagement",
        "type": "select",
        "options": ["0", "2", "8", "12"],
        "default": "8",
        "help": "Self-study and revision hours dedicated per week outside class.",
    },
    {
        "name": "Attendance",
        "label": "Lecture Attendance",
        "group": "Study Habits & Engagement",
        "type": "select",
        "options": ["Always", "Sometimes", "Never"],
        "default": "Always",
        "help": "Frequency of attending scheduled lectures.",
    },
    {
        "name": "Reading",
        "label": "Regular Course Reading",
        "group": "Study Habits & Engagement",
        "type": "select",
        "options": ["Yes", "No"],
        "default": "Yes",
        "help": "Engages in regular reading of textbooks and materials.",
    },
    {
        "name": "Notes",
        "label": "Active Note Taking",
        "group": "Study Habits & Engagement",
        "type": "select",
        "options": ["Yes", "No"],
        "default": "Yes",
        "help": "Takes and reviews comprehensive lecture notes.",
    },
    {
        "name": "Listening_in_Class",
        "label": "Listening in Class",
        "group": "Study Habits & Engagement",
        "type": "select",
        "options": ["Yes", "No"],
        "default": "Yes",
        "help": "Attentive listening and participating in classroom discussions.",
    },
    {
        "name": "Project_work",
        "label": "Project Work Involvement",
        "group": "Study Habits & Engagement",
        "type": "select",
        "options": ["Yes", "No"],
        "default": "Yes",
        "help": "Active participation in project assignments and group work.",
    },
]

_factor_model = None


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize anomalies in the raw dataset."""
    df = df.copy()
    if "Scholarship" in df.columns:
        df["Scholarship"] = df["Scholarship"].fillna("None").astype(str).replace({"None": "0%"})
    if "Attendance" in df.columns:
        df["Attendance"] = df["Attendance"].astype(str).replace({"3": "Never"})
    if "Notes" in df.columns:
        df["Notes"] = df["Notes"].astype(str).replace({"6": "Yes"})
    if "Listening_in_Class" in df.columns:
        df["Listening_in_Class"] = df["Listening_in_Class"].astype(str).replace({"6": "Yes"})
    if "Weekly_Study_Hours" in df.columns:
        df["Weekly_Study_Hours"] = pd.to_numeric(df["Weekly_Study_Hours"], errors="coerce").fillna(0)
    return df


def train_factor_model() -> Pipeline:
    """Train Random Forest classifier on the Kaggle student performance dataset."""
    if not os.path.exists(KAGGLE_DATA_PATH):
        raise FileNotFoundError(f"Kaggle dataset not found at {KAGGLE_DATA_PATH}")

    df = pd.read_csv(KAGGLE_DATA_PATH)
    df = clean_dataframe(df)

    X = df[FEATURE_COLUMNS]
    y = df["Grade"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=160,
        max_depth=6,
        min_samples_split=3,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline([("prep", preprocessor), ("clf", clf)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

    pipeline.fit(X, y)

    # Calculate train accuracy and metrics
    train_preds = pipeline.predict(X)
    train_acc = accuracy_score(y, train_preds)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, FACTOR_MODEL_PATH)

    metrics = {
        "dataset_name": "Student Performance Prediction Dataset (Kaggle)",
        "dataset_url": KAGGLE_DATASET_URL,
        "sample_count": len(df),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "train_accuracy": round(float(train_acc), 4),
        "classes": list(pipeline.classes_),
    }

    with open(FACTOR_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    global _factor_model
    _factor_model = pipeline
    return pipeline


def get_factor_model() -> Pipeline:
    """Load cached model or train fresh."""
    global _factor_model
    if _factor_model is not None:
        return _factor_model
    if os.path.exists(FACTOR_MODEL_PATH):
        try:
            _factor_model = joblib.load(FACTOR_MODEL_PATH)
            return _factor_model
        except Exception:
            pass
    return train_factor_model()


def build_factor_recommendations(inputs: dict[str, Any], predicted_grade: str) -> list[dict[str, str]]:
    """Generate personalized recommendations based on student input values."""
    recs = []

    # Attendance
    att = inputs.get("Attendance", "Always")
    if att == "Never":
        recs.append({
            "category": "Critical Attendance Alert",
            "type": "danger",
            "text": "Class attendance is 'Never'. Non-attendance is the single largest driver of academic failure in student performance surveys. Prioritize attending scheduled lectures consistently.",
        })
    elif att == "Sometimes":
        recs.append({
            "category": "Attendance Improvement",
            "type": "warning",
            "text": "Attending lectures only 'Sometimes' creates knowledge gaps. Increasing attendance to 'Always' is proven to elevate grades across all subjects.",
        })
    else:
        recs.append({
            "category": "Consistent Attendance",
            "type": "success",
            "text": "Excellent habit! Attending lectures consistently establishes a solid foundation for test and exam performance.",
        })

    # Weekly Study Hours
    study_hrs = float(inputs.get("Weekly_Study_Hours", 0))
    if study_hrs == 0:
        recs.append({
            "category": "Study Time Shortfall",
            "type": "danger",
            "text": "Reporting 0 weekly study hours outside class. Allocating even 4–8 focused hours each week can dramatically improve grade outcomes.",
        })
    elif study_hrs <= 2:
        recs.append({
            "category": "Increase Revision Time",
            "type": "warning",
            "text": "1–2 hours per week is minimal. Expanding dedicated self-study to 6–10 hours per week will reinforce difficult topics.",
        })
    else:
        recs.append({
            "category": "Strong Study Commitment",
            "type": "success",
            "text": f"Dedicating {int(study_hrs)} hours weekly to study gives you a major academic advantage over peers.",
        })

    # Note taking & Reading
    notes = inputs.get("Notes", "Yes")
    reading = inputs.get("Reading", "Yes")
    if notes == "No":
        recs.append({
            "category": "Active Note Taking",
            "type": "warning",
            "text": "Active note-taking during class aids cognitive retention. Writing summarized notes will make exam revision much faster.",
        })
    if reading == "No":
        recs.append({
            "category": "Curricular Reading",
            "type": "warning",
            "text": "Supplementing class lectures with textbooks or syllabus readings broadens conceptual understanding.",
        })

    # Additional Work
    work = inputs.get("Additional_Work", "No")
    if work == "Yes":
        recs.append({
            "category": "Work-Study Balance",
            "type": "info",
            "text": "Balancing outside work with studies requires intentional calendar blocking. Protect your core study and resting hours.",
        })

    # Project Work
    proj = inputs.get("Project_work", "Yes")
    if proj == "Yes":
        recs.append({
            "category": "Hands-on Project Work",
            "type": "success",
            "text": "Active engagement in practical projects reinforces theoretical learning and boosts continuous assessment marks.",
        })

    return recs


def predict_factor_performance(inputs: dict[str, Any]) -> dict[str, Any]:
    """Turn raw factor inputs into grade prediction and analysis."""
    model = get_factor_model()

    # Pre-process inputs
    row = {}
    for col in FEATURE_COLUMNS:
        val = inputs.get(col, "")
        if col == "Weekly_Study_Hours":
            try:
                row[col] = float(val)
            except (ValueError, TypeError):
                row[col] = 0.0
        elif col == "Scholarship":
            row[col] = "0%" if val == "None" else str(val)
        else:
            row[col] = str(val)

    df_single = pd.DataFrame([row])
    predicted_grade = str(model.predict(df_single)[0])

    # Probabilities
    probs = {}
    if hasattr(model, "predict_proba"):
        prob_values = model.predict_proba(df_single)[0]
        classes = list(model.classes_)
        for cls_name, p in zip(classes, prob_values):
            probs[cls_name] = round(float(p) * 100, 1)

    grade_meta = GRADE_INFO.get(
        predicted_grade,
        {"title": predicted_grade, "gpa": "N/A", "badge": "badge-upper", "status": "Recorded"},
    )

    recs = build_factor_recommendations(inputs, predicted_grade)

    return {
        "predicted_grade": predicted_grade,
        "grade_title": grade_meta["title"],
        "grade_gpa": grade_meta["gpa"],
        "grade_badge": grade_meta["badge"],
        "status": grade_meta["status"],
        "probabilities": probs,
        "recommendations": recs,
        "inputs": inputs,
        "dataset_url": KAGGLE_DATASET_URL,
    }
