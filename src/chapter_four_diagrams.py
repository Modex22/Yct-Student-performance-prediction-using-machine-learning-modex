"""Generate aggregate Chapter Four diagrams from paired semester workbooks."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import (
    KFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from src.chapter_four_catalog import CHAPTER_FOUR_DIAGRAMS
from src.config import FIGURES_DIR, NEXT_SEMESTER_MODEL_PATH

CV_FOLDS = 5
CV_REPEATS = 10
RANDOM_STATE = 42

FIG_BG = "#121b16"
FIG_TEXT = "#eaf5ee"
FIG_MUTED = "#8ca897"
FIG_GRID = "#22322a"
FIG_ACCENT = "#4ade80"
FIG_SECONDARY = "#60a5fa"
FIG_WARNING = "#fbbf24"
FIG_RISK = "#f87171"


def _style_axis(axis: Any) -> None:
    axis.set_facecolor(FIG_BG)
    axis.tick_params(colors=FIG_MUTED, labelsize=9)
    for spine in axis.spines.values():
        spine.set_color(FIG_GRID)
    axis.title.set_color(FIG_TEXT)
    axis.xaxis.label.set_color(FIG_MUTED)
    axis.yaxis.label.set_color(FIG_MUTED)
    axis.grid(color=FIG_GRID, linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)


def _save_figure(figure: Any, output_dir: Path, filename: str) -> None:
    figure.set_facecolor(FIG_BG)
    figure.tight_layout()
    figure.savefig(output_dir / filename, dpi=120, facecolor=FIG_BG)
    plt.close(figure)


def _read_grade_rules(workbook_path: Path) -> list[tuple[float, float, str, float]]:
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        if "Grading & Policy" not in workbook.sheetnames:
            raise ValueError(
                f"{workbook_path.name} must contain a 'Grading & Policy' sheet."
            )
        rules = []
        for row in workbook["Grading & Policy"].iter_rows(values_only=True):
            if (
                len(row) >= 4
                and isinstance(row[0], (int, float))
                and isinstance(row[1], (int, float))
                and isinstance(row[3], (int, float))
                and row[2] is not None
            ):
                rules.append((
                    float(row[0]),
                    float(row[1]) + 1,
                    str(row[2]).strip(),
                    float(row[3]),
                ))
        rules.sort(key=lambda rule: rule[0])
        if not rules:
            raise ValueError("No numeric score-to-grade rules were found.")
        return rules
    finally:
        workbook.close()


def _grade_for_score(
    score: float,
    rules: list[tuple[float, float, str, float]],
) -> tuple[str, float]:
    for lower, upper_exclusive, grade, grade_point in rules:
        if lower <= score < upper_exclusive:
            return grade, grade_point
    raise ValueError("A score falls outside the workbook's grading policy.")


def _read_course_results(
    workbook_path: Path,
    *,
    require_source: bool,
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, list[float]]]:
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        if "Course Results" not in workbook.sheetnames:
            raise ValueError(f"{workbook_path.name} must contain a 'Course Results' sheet.")
        rows = workbook["Course Results"].iter_rows(values_only=True)
        first_row = next(rows, None)
        if first_row is None:
            raise ValueError(f"{workbook_path.name} has an empty 'Course Results' sheet.")
        headers = [str(value).strip() if value is not None else "" for value in first_row]
        required = {"Student_ID", "Course_Unit", "Score"}
        if require_source:
            required.add("Data_Source")
        missing = required - set(headers)
        if missing:
            raise ValueError(
                f"{workbook_path.name} is missing required result columns: "
                f"{', '.join(sorted(missing))}."
            )

        student_index = headers.index("Student_ID")
        unit_index = headers.index("Course_Unit")
        score_index = headers.index("Score")
        source_index = headers.index("Data_Source") if require_source else None
        all_results: dict[str, list[tuple[float, float]]] = defaultdict(list)
        observed_scores: dict[str, list[float]] = defaultdict(list)

        for row_number, row in enumerate(rows, start=2):
            if all(value is None or str(value).strip() == "" for value in row):
                continue
            student_value = row[student_index]
            raw_score = row[score_index]
            raw_units = row[unit_index]
            if student_value is None:
                raise ValueError(
                    f"{workbook_path.name} has an incomplete result at row {row_number}."
                )
            if (
                isinstance(raw_score, bool)
                or not isinstance(raw_score, (int, float))
                or not np.isfinite(float(raw_score))
                or not 0 <= float(raw_score) <= 100
                or isinstance(raw_units, bool)
                or not isinstance(raw_units, (int, float))
                or not np.isfinite(float(raw_units))
                or float(raw_units) <= 0
            ):
                raise ValueError(
                    f"{workbook_path.name} has an invalid score or credit value "
                    f"at row {row_number}."
                )

            student_id = str(student_value).strip()
            score = float(raw_score)
            units = float(raw_units)
            all_results[student_id].append((score, units))

            if source_index is not None:
                source_value = row[source_index]
                source = str(source_value).strip().lower() if source_value is not None else ""
                if source not in {"observed", "estimated"}:
                    raise ValueError(
                        f"{workbook_path.name} has an unknown score source "
                        f"at row {row_number}."
                    )
                if source == "observed":
                    observed_scores[student_id].append(score)

        return dict(all_results), dict(observed_scores)
    finally:
        workbook.close()


def _read_summary(
    workbook_path: Path,
    fields: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        if "Student Summary" not in workbook.sheetnames:
            raise ValueError(f"{workbook_path.name} must contain a 'Student Summary' sheet.")
        rows = workbook["Student Summary"].iter_rows(values_only=True)
        first_row = next(rows, None)
        if first_row is None:
            raise ValueError(f"{workbook_path.name} has an empty 'Student Summary' sheet.")
        headers = [str(value).strip() if value is not None else "" for value in first_row]
        required = {"Student_ID", *fields}
        missing = required - set(headers)
        if missing:
            raise ValueError(
                f"{workbook_path.name} is missing summary fields: "
                f"{', '.join(sorted(missing))}."
            )
        student_index = headers.index("Student_ID")
        indices = {field: headers.index(field) for field in fields}
        summary: dict[str, dict[str, float]] = {}
        for row_number, row in enumerate(rows, start=2):
            if row[student_index] is None:
                continue
            student_id = str(row[student_index]).strip()
            values = {}
            for field, index in indices.items():
                value = row[index]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(float(value))
                ):
                    raise ValueError(
                        f"{workbook_path.name} has an invalid {field} at row {row_number}."
                    )
                values[field] = float(value)
            summary[student_id] = values
        return summary
    finally:
        workbook.close()


def _weighted_gpa(
    results: list[tuple[float, float]],
    rules: list[tuple[float, float, str, float]],
) -> tuple[float, float, float]:
    total_units = sum(units for _, units in results)
    weighted_points = sum(
        _grade_for_score(score, rules)[1] * units
        for score, units in results
    )
    return weighted_points / total_units, weighted_points, total_units


def _collect_records(
    first_semester_path: Path,
    second_semester_path: Path,
    rules: list[tuple[float, float, str, float]],
) -> tuple[list[dict[str, float | int | str]], list[float]]:
    first_all, first_observed = _read_course_results(
        first_semester_path,
        require_source=True,
    )
    second_all, _ = _read_course_results(second_semester_path, require_source=False)
    second_summary = _read_summary(
        second_semester_path,
        ("Prior_CGPA", "Prior_CTU", "Prior_CWGP"),
    )
    matched_ids = sorted(
        first_all.keys()
        & first_observed.keys()
        & second_all.keys()
        & second_summary.keys()
    )
    if len(matched_ids) < 10:
        raise ValueError("At least 10 matched students are required to generate diagrams.")

    records: list[dict[str, float | int | str]] = []
    second_scores = []
    for student_id in matched_ids:
        first_results = first_all[student_id]
        second_results = second_all[student_id]
        observed_values = first_observed[student_id]
        if not first_results or not second_results or not observed_values:
            continue

        first_average = float(np.mean([score for score, _ in first_results]))
        observed_average = float(np.mean(observed_values))
        second_average = float(np.mean([score for score, _ in second_results]))
        first_gpa, _, _ = _weighted_gpa(first_results, rules)
        second_gpa, second_weighted_points, second_units = _weighted_gpa(
            second_results,
            rules,
        )
        prior = second_summary[student_id]
        final_cgpa = (
            prior["Prior_CWGP"] + second_weighted_points
        ) / (
            prior["Prior_CTU"] + second_units
        )
        failed_results = sum(
            _grade_for_score(score, rules)[0] == "F"
            for score, _ in second_results
        )
        first_band, _ = _grade_for_score(first_average, rules)
        second_band, _ = _grade_for_score(second_average, rules)
        records.append({
            "first_average": first_average,
            "observed_first_average": observed_average,
            "first_gpa": first_gpa,
            "prior_cgpa": prior["Prior_CGPA"],
            "second_average": second_average,
            "second_gpa": second_gpa,
            "final_cgpa": final_cgpa,
            "failed_results": failed_results,
            "risk": int(failed_results > 0),
            "first_band": first_band,
            "second_band": second_band,
        })
        second_scores.extend(score for score, _ in second_results)

    if len(records) != len(matched_ids):
        raise ValueError("Some matched students have incomplete score coverage.")
    return records, second_scores


def _regression_estimator(name: str) -> Any:
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Ridge":
        return Ridge(alpha=1.0)
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=1,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        )
    if name == "Mean baseline":
        return DummyRegressor(strategy="mean")
    raise ValueError(f"Unsupported regression model in benchmark: {name}.")


def _risk_estimators() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
    }


def _train_risk_models(
    features: np.ndarray,
    targets: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, str]:
    counts = np.bincount(targets.astype(int), minlength=2)
    if counts.min() < CV_FOLDS:
        raise ValueError(
            "Both academic-risk groups need at least five students for "
            "cross-validated classification diagrams."
        )
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=targets,
    )
    scoring = {
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }
    results = {}
    fitted_models = {}
    for name, estimator in _risk_estimators().items():
        cv = cross_validate(
            estimator,
            x_train,
            y_train,
            cv=RepeatedStratifiedKFold(
                n_splits=CV_FOLDS,
                n_repeats=CV_REPEATS,
                random_state=RANDOM_STATE,
            ),
            scoring=scoring,
            n_jobs=1,
        )
        estimator.fit(x_train, y_train)
        predictions = estimator.predict(x_test)
        probabilities = estimator.predict_proba(x_test)[:, list(estimator.classes_).index(1)]
        results[name] = {
            "cv_precision": float(cv["test_precision"].mean()),
            "cv_recall": float(cv["test_recall"].mean()),
            "cv_f1": float(cv["test_f1"].mean()),
            "holdout_precision": float(precision_score(y_test, predictions, zero_division=0)),
            "holdout_recall": float(recall_score(y_test, predictions, zero_division=0)),
            "holdout_f1": float(f1_score(y_test, predictions, zero_division=0)),
            "predictions": predictions,
            "probabilities": probabilities,
        }
        fitted_models[name] = estimator

    selected_name = max(results, key=lambda name: results[name]["cv_f1"])
    out_of_fold_probabilities = cross_val_predict(
        _risk_estimators()[selected_name],
        features,
        targets,
        cv=StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        ),
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    return results, y_test, out_of_fold_probabilities, selected_name


def _grade_order(rules: list[tuple[float, float, str, float]]) -> list[str]:
    return [rule[2] for rule in rules]


def _generate_diagrams(
    records: list[dict[str, float | int | str]],
    second_scores: list[float],
    grade_rules: list[tuple[float, float, str, float]],
    model_metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    record_values = lambda field: np.array(
        [float(record[field]) for record in records],
        dtype=float,
    )
    first_average_all = record_values("first_average")
    observed_first_average = record_values("observed_first_average")
    second_average = record_values("second_average")
    first_gpa = record_values("first_gpa")
    second_gpa = record_values("second_gpa")
    prior_cgpa = record_values("prior_cgpa")
    final_cgpa = record_values("final_cgpa")
    failed_results = record_values("failed_results")
    grade_order = _grade_order(grade_rules)
    fail_threshold = next(rule[1] - 1 for rule in grade_rules if rule[2] == "F") + 1

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(first_gpa, bins=np.linspace(0, 4, 17), color=FIG_ACCENT, edgecolor=FIG_BG)
    axis.set(xlim=(0, 4), xlabel="Grade point average", ylabel="Students")
    axis.set_title("First-semester GPA distribution")
    _style_axis(axis)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[0]["filename"])

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(second_scores, bins=np.arange(0, 105, 5), color=FIG_SECONDARY, edgecolor=FIG_BG)
    axis.set(xlim=(0, 100), xlabel="Score", ylabel="Results")
    axis.set_title("Second-semester score distribution")
    _style_axis(axis)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[1]["filename"])

    first_bands = Counter(str(record["first_band"]) for record in records)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(grade_order, [first_bands[grade] for grade in grade_order], color=FIG_ACCENT)
    axis.set(xlabel="Grade band of semester average", ylabel="Students")
    axis.set_title("First-semester performance distribution")
    _style_axis(axis)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[2]["filename"])

    failed_mark_count = sum(score < fail_threshold for score in second_scores)
    pass_fail = [len(second_scores) - failed_mark_count, failed_mark_count]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(["Pass", "Fail"], pass_fail, color=[FIG_ACCENT, FIG_RISK])
    axis.bar_label(bars, padding=3, color=FIG_TEXT)
    axis.set(ylabel="Result records")
    axis.set_title("Second-semester pass/fail distribution")
    _style_axis(axis)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[3]["filename"])

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(second_gpa, final_cgpa, alpha=0.7, color=FIG_ACCENT, edgecolor="none")
    axis.set(xlabel="Second-semester GPA", ylabel="Cumulative GPA after semester")
    axis.set_title("Student GPA and CGPA comparison")
    _style_axis(axis)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[4]["filename"])

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(prior_cgpa, second_average, alpha=0.7, color=FIG_SECONDARY, edgecolor="none")
    axis.set(xlabel="Prior CGPA", ylabel="Second-semester average score")
    axis.set_title("Prior CGPA vs second-semester average")
    _style_axis(axis)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[5]["filename"])

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(first_average_all, second_average, alpha=0.7, color=FIG_ACCENT, edgecolor="none")
    axis.set(xlabel="Semester 1 average score", ylabel="Semester 2 average score")
    axis.set_title("Semester 1 vs Semester 2 performance")
    _style_axis(axis)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[6]["filename"])

    maximum_fails = int(failed_results.max())
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(
        failed_results,
        bins=np.arange(-0.5, maximum_fails + 1.5, 1),
        color=FIG_RISK,
        edgecolor=FIG_BG,
    )
    axis.set(xlabel="Number of failed results", ylabel="Students")
    axis.set_title("Failed results per student")
    _style_axis(axis)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[7]["filename"])

    feature_columns = [
        first_average_all,
        first_gpa,
        prior_cgpa,
        second_average,
        second_gpa,
        failed_results,
    ]
    feature_names = ["Sem 1 avg", "Sem 1 GPA", "Prior CGPA", "Sem 2 avg", "Sem 2 GPA", "Failed results"]
    correlations = np.corrcoef(np.column_stack(feature_columns), rowvar=False)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(correlations, vmin=-1, vmax=1, cmap="RdYlGn")
    axis.set_xticks(range(len(feature_names)), labels=feature_names, rotation=35, ha="right")
    axis.set_yticks(range(len(feature_names)), labels=feature_names)
    for row_index in range(len(feature_names)):
        for column_index in range(len(feature_names)):
            axis.text(
                column_index,
                row_index,
                f"{correlations[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                color=FIG_TEXT,
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label="Correlation")
    axis.set_title("Student-level feature correlation")
    _style_axis(axis)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[8]["filename"])

    distribution_data = [
        (first_average_all, "Sem 1 average", (0, 100)),
        (second_average, "Sem 2 average", (0, 100)),
        (first_gpa, "Sem 1 GPA", (0, 4)),
        (second_gpa, "Sem 2 GPA", (0, 4)),
        (prior_cgpa, "Prior CGPA", (0, 4)),
        (failed_results, "Failed results", (0, max(1, maximum_fails))),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(11, 7))
    for axis, (values, label, limits) in zip(axes.flat, distribution_data):
        axis.hist(values, bins=12, color=FIG_ACCENT, edgecolor=FIG_BG)
        axis.set(xlabel=label, ylabel="Students", xlim=limits)
        _style_axis(axis)
    figure.suptitle("Student-level feature distributions", color=FIG_TEXT)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[9]["filename"])

    regression_ids = np.arange(len(records))
    train_ids, test_ids = train_test_split(
        regression_ids,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, values, label in (
        (axes[0, 0], observed_first_average, "Observed semester 1 average"),
        (axes[0, 1], second_average, "Semester 2 average"),
    ):
        axis.hist(values[train_ids], bins=10, color=FIG_ACCENT, alpha=0.8, label="Training")
        axis.hist(values[test_ids], bins=10, color=FIG_SECONDARY, alpha=0.8, label="Testing")
        axis.set(xlabel=label, ylabel="Students")
        axis.legend(facecolor=FIG_BG, labelcolor=FIG_TEXT)
        _style_axis(axis)
    axes[1, 0].boxplot(
        [observed_first_average[train_ids], observed_first_average[test_ids]],
        patch_artist=True,
        boxprops={"facecolor": FIG_ACCENT, "alpha": 0.55},
        medianprops={"color": FIG_TEXT},
    )
    axes[1, 0].set_xticks([1, 2], labels=["Training", "Testing"])
    axes[1, 0].set_ylabel("Observed semester 1 average")
    _style_axis(axes[1, 0])
    axes[1, 1].boxplot(
        [second_average[train_ids], second_average[test_ids]],
        patch_artist=True,
        boxprops={"facecolor": FIG_SECONDARY, "alpha": 0.55},
        medianprops={"color": FIG_TEXT},
    )
    axes[1, 1].set_xticks([1, 2], labels=["Training", "Testing"])
    axes[1, 1].set_ylabel("Semester 2 average")
    _style_axis(axes[1, 1])
    figure.suptitle("Training and testing data distribution", color=FIG_TEXT)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[10]["filename"])

    model_x_train = observed_first_average[train_ids].reshape(-1, 1)
    model_x_test = observed_first_average[test_ids].reshape(-1, 1)
    model_y_train = second_average[train_ids]
    model_y_test = second_average[test_ids]
    selected_regressor_name = str(model_metadata["selected_model"])
    regressor = _regression_estimator(selected_regressor_name)
    regressor.fit(model_x_train, model_y_train)
    predictions = regressor.predict(model_x_test)
    importance = permutation_importance(
        regressor,
        model_x_test,
        model_y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=30,
        random_state=RANDOM_STATE,
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(
        ["Observed semester 1 average"],
        importance.importances_mean,
        xerr=importance.importances_std,
        color=FIG_ACCENT,
        capsize=4,
    )
    axis.set(xlabel="Increase in holdout MAE when shuffled")
    axis.set_title("Forecast model feature importance")
    _style_axis(axis)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[11]["filename"])

    regression_results = model_metadata["benchmark_results"]
    model_names = [str(result["model"]) for result in regression_results]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(model_names, [result["cv_mae"] for result in regression_results], color=FIG_ACCENT)
    axes[0].set(ylabel="Mean absolute error (lower is better)", title="Repeated CV MAE")
    axes[0].tick_params(axis="x", rotation=25)
    _style_axis(axes[0])
    axes[1].bar(model_names, [result["cv_r2"] for result in regression_results], color=FIG_SECONDARY)
    axes[1].set(ylabel="R² (higher is better)", title="Repeated CV R²")
    axes[1].tick_params(axis="x", rotation=25)
    _style_axis(axes[1])
    figure.suptitle("Regression model performance comparison", color=FIG_TEXT)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[12]["filename"])

    risk_features = np.column_stack([
        observed_first_average,
    ])
    risk_targets = record_values("risk").astype(int)
    risk_results, risk_y_test, oof_risk_probabilities, selected_classifier = _train_risk_models(
        risk_features,
        risk_targets,
    )
    classifier_names = list(risk_results)
    metric_names = ("cv_precision", "cv_recall", "cv_f1")
    metric_labels = ("Precision", "Recall", "F1")
    x_positions = np.arange(len(classifier_names))
    width = 0.24
    figure, axis = plt.subplots(figsize=(9, 5))
    for metric_index, (metric, label, color) in enumerate(zip(
        metric_names,
        metric_labels,
        (FIG_ACCENT, FIG_SECONDARY, FIG_WARNING),
    )):
        values = [risk_results[name][metric] for name in classifier_names]
        axis.bar(
            x_positions + (metric_index - 1) * width,
            values,
            width,
            label=label,
            color=color,
        )
    axis.set(
        xticks=x_positions,
        xticklabels=classifier_names,
        ylim=(0, 1),
        ylabel="Cross-validation score",
    )
    axis.legend(facecolor=FIG_BG, labelcolor=FIG_TEXT)
    axis.set_title("Risk-classifier precision, recall and F1")
    _style_axis(axis)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[13]["filename"])

    selected_name = selected_classifier
    selected_predictions = risk_results[selected_name]["predictions"]
    matrix = confusion_matrix(risk_y_test, selected_predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Greens")
    for row_index in range(2):
        for column_index in range(2):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color=FIG_TEXT,
                fontsize=13,
            )
    axis.set(
        xticks=[0, 1],
        xticklabels=["No failure", "At risk"],
        yticks=[0, 1],
        yticklabels=["No failure", "At risk"],
        xlabel="Predicted",
        ylabel="Actual",
    )
    axis.set_title(f"Confusion matrix: {selected_name}")
    figure.colorbar(image, ax=axis)
    _style_axis(axis)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[14]["filename"])

    figure, axis = plt.subplots(figsize=(7, 6))
    for name in classifier_names:
        probabilities = risk_results[name]["probabilities"]
        false_positive, true_positive, _ = roc_curve(risk_y_test, probabilities)
        axis.plot(
            false_positive,
            true_positive,
            label=f"{name} (AUC {auc(false_positive, true_positive):.2f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color=FIG_MUTED, label="Chance")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
    axis.set_title("Academic-risk ROC curves")
    axis.legend(facecolor=FIG_BG, labelcolor=FIG_TEXT, fontsize=8)
    _style_axis(axis)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[15]["filename"])

    figure, axis = plt.subplots(figsize=(7, 6))
    for name in classifier_names:
        probabilities = risk_results[name]["probabilities"]
        precision, recall, _ = precision_recall_curve(risk_y_test, probabilities)
        average_precision = average_precision_score(risk_y_test, probabilities)
        axis.plot(recall, precision, label=f"{name} (AP {average_precision:.2f})")
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    axis.set_title("Academic-risk precision-recall curves")
    axis.legend(facecolor=FIG_BG, labelcolor=FIG_TEXT, fontsize=8)
    _style_axis(axis)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[16]["filename"])

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(model_y_test, predictions, alpha=0.75, color=FIG_ACCENT, edgecolor="none")
    axis.plot([0, 100], [0, 100], linestyle="--", color=FIG_RISK)
    axis.set(
        xlim=(0, 100),
        ylim=(0, 100),
        xlabel="Actual semester 2 average",
        ylabel="Predicted semester 2 average",
    )
    axis.set_title("Actual vs predicted semester average")
    _style_axis(axis)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[17]["filename"])

    figure, axis = plt.subplots(figsize=(10, 5))
    fold_values = [
        result["cv_mae_folds"]
        for result in regression_results
    ]
    axis.boxplot(
        fold_values,
        patch_artist=True,
        boxprops={"facecolor": FIG_ACCENT, "alpha": 0.55},
        medianprops={"color": FIG_TEXT},
    )
    axis.set_xticks(range(1, len(model_names) + 1), labels=model_names)
    axis.set(ylabel="Mean absolute error per fold")
    axis.tick_params(axis="x", rotation=20)
    axis.set_title("Repeated cross-validation performance")
    _style_axis(axis)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[18]["filename"])

    selected_oof_probabilities = oof_risk_probabilities
    risk_values = risk_targets.astype(bool)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(
        selected_oof_probabilities[~risk_values],
        bins=np.linspace(0, 1, 11),
        alpha=0.75,
        color=FIG_SECONDARY,
        label="No failed results",
    )
    axis.hist(
        selected_oof_probabilities[risk_values],
        bins=np.linspace(0, 1, 11),
        alpha=0.75,
        color=FIG_RISK,
        label="One or more failed results",
    )
    axis.axvline(0.5, color=FIG_WARNING, linestyle="--", label="0.5 threshold")
    axis.set(xlim=(0, 1), xlabel="Out-of-fold predicted risk probability", ylabel="Students")
    axis.set_title("Predicted academic-risk distribution")
    axis.legend(facecolor=FIG_BG, labelcolor=FIG_TEXT, fontsize=8)
    _style_axis(axis)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[19]["filename"])

    second_bands = Counter(str(record["second_band"]) for record in records)
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(grade_order, [second_bands[grade] for grade in grade_order], color=FIG_ACCENT)
    axis.bar_label(bars, padding=3, color=FIG_TEXT)
    axis.set(xlabel="Grade band of semester average", ylabel="Students")
    axis.set_title("Student performance classification")
    _style_axis(axis)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[20]["filename"])

    predicted_bands = [
        _grade_for_score(float(score), grade_rules)[0]
        for score in predictions
    ]
    actual_bands = [
        _grade_for_score(float(score), grade_rules)[0]
        for score in model_y_test
    ]
    actual_counts = Counter(actual_bands)
    predicted_counts = Counter(predicted_bands)
    positions = np.arange(len(grade_order))
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(
        positions - 0.18,
        [actual_counts[grade] for grade in grade_order],
        0.36,
        color=FIG_ACCENT,
        label="Actual",
    )
    axis.bar(
        positions + 0.18,
        [predicted_counts[grade] for grade in grade_order],
        0.36,
        color=FIG_SECONDARY,
        label="Predicted",
    )
    axis.set(
        xticks=positions,
        xticklabels=grade_order,
        xlabel="Grade band",
        ylabel="Holdout students",
    )
    axis.legend(facecolor=FIG_BG, labelcolor=FIG_TEXT)
    axis.set_title("Actual vs predicted classification")
    _style_axis(axis)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[21]["filename"])

    cv_residual_predictions = cross_val_predict(
        _regression_estimator(selected_regressor_name),
        model_x_train,
        model_y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    )
    residuals = model_y_train - cv_residual_predictions
    rng = np.random.default_rng(RANDOM_STATE)
    bootstrap_predictions = np.empty((500, len(model_x_test)))
    for iteration in range(bootstrap_predictions.shape[0]):
        sample_indices = rng.integers(0, len(model_x_train), len(model_x_train))
        bootstrap_model = _regression_estimator(selected_regressor_name)
        bootstrap_model.fit(model_x_train[sample_indices], model_y_train[sample_indices])
        mean_predictions = bootstrap_model.predict(model_x_test)
        residual_draws = rng.choice(residuals, size=len(model_x_test), replace=True)
        bootstrap_predictions[iteration] = np.clip(
            mean_predictions + residual_draws,
            0,
            100,
        )
    lower = np.percentile(bootstrap_predictions, 5, axis=0)
    upper = np.percentile(bootstrap_predictions, 95, axis=0)
    interval_widths = upper - lower
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(interval_widths, bins=12, color=FIG_WARNING, edgecolor=FIG_BG)
    axis.set(xlabel="90% bootstrap prediction-interval width (score points)", ylabel="Holdout students")
    axis.set_title("Prediction confidence distribution")
    axis.text(
        0.98,
        0.92,
        "Intervals use bootstrap model fits and cross-validated residuals.",
        transform=axis.transAxes,
        ha="right",
        color=FIG_MUTED,
        fontsize=8,
    )
    _style_axis(axis)
    _save_figure(figure, output_dir, CHAPTER_FOUR_DIAGRAMS[22]["filename"])


def generate_chapter_four_diagrams(
    first_semester_path: Path,
    second_semester_path: Path,
    output_dir: Path = Path(FIGURES_DIR) / "chapter_four",
    model_path: Path = Path(NEXT_SEMESTER_MODEL_PATH),
) -> list[Path]:
    grade_rules = _read_grade_rules(first_semester_path)
    records, second_scores = _collect_records(
        first_semester_path,
        second_semester_path,
        grade_rules,
    )
    with model_path.open(encoding="utf-8") as file:
        model_metadata = json.load(file)
    if not model_metadata.get("benchmark_results"):
        raise ValueError("The forecast model file has no saved benchmark results.")

    _generate_diagrams(
        records,
        second_scores,
        grade_rules,
        model_metadata,
        output_dir,
    )
    paths = [output_dir / diagram["filename"] for diagram in CHAPTER_FOUR_DIAGRAMS]
    missing = [path.name for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Some diagrams were not generated: {', '.join(missing)}.")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_semester_workbook", type=Path)
    parser.add_argument("second_semester_workbook", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(FIGURES_DIR) / "chapter_four",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(NEXT_SEMESTER_MODEL_PATH),
        help="forecast model metadata containing the real benchmark results",
    )
    args = parser.parse_args()
    generated = generate_chapter_four_diagrams(
        args.first_semester_workbook,
        args.second_semester_workbook,
        args.output_dir,
        args.model,
    )
    print(f"Generated {len(generated)} aggregate diagrams in {args.output_dir}.")


if __name__ == "__main__":
    main()
