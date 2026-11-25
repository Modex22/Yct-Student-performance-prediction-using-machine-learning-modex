"""Names and image files for the Chapter Four analysis gallery."""

CHAPTER_FOUR_DIAGRAMS = [
    {
        "filename": "01_first_semester_gpa.png",
        "title": "First-semester GPA distribution",
        "description": "Shows how student GPA values are spread across the official 0–4 scale; estimated marks are included.",
    },
    {
        "filename": "02_second_semester_scores.png",
        "title": "Second-semester score distribution",
        "description": "Counts recorded second-semester marks in five-point score intervals.",
    },
    {
        "filename": "03_first_semester_performance.png",
        "title": "First-semester performance distribution",
        "description": "Groups students by the grade band of their first-semester average, including estimated marks.",
    },
    {
        "filename": "04_second_semester_pass_fail.png",
        "title": "Second-semester pass/fail distribution",
        "description": "Counts second-semester results below 40 as fails and results of 40 or higher as passes.",
    },
    {
        "filename": "05_gpa_cgpa_comparison.png",
        "title": "Student GPA and CGPA comparison",
        "description": "Compares each student's second-semester GPA with cumulative GPA after that semester.",
    },
    {
        "filename": "06_prior_cgpa_vs_second_average.png",
        "title": "Prior CGPA vs second-semester average",
        "description": "Shows the relationship between prior CGPA and the recorded second-semester average.",
    },
    {
        "filename": "07_semester_performance_comparison.png",
        "title": "Semester 1 vs Semester 2 performance",
        "description": "Compares student averages across semesters; first-semester averages include estimates.",
    },
    {
        "filename": "08_failed_results_per_student.png",
        "title": "Failed results per student",
        "description": "Shows how many second-semester results below 40 each student received.",
    },
    {
        "filename": "09_feature_correlation.png",
        "title": "Feature correlation heatmap",
        "description": "Displays pairwise correlations among semester averages, GPA/CGPA, and failed-result counts.",
    },
    {
        "filename": "10_feature_distributions.png",
        "title": "Feature distribution analysis",
        "description": "Shows distributions of semester averages, GPA/CGPA, and failed-result counts.",
    },
    {
        "filename": "11_train_test_distribution.png",
        "title": "Training and testing data distribution",
        "description": "Compares observed first-semester model inputs and second-semester targets in training and holdout sets.",
    },
    {
        "filename": "12_feature_importance.png",
        "title": "Feature importance",
        "description": "Shows how much holdout MAE rises when the observed first-semester average is shuffled.",
    },
    {
        "filename": "13_algorithm_performance.png",
        "title": "Algorithm performance comparison (MAE/R²)",
        "description": "Compares five regression models; lower cross-validation MAE and higher R² are better.",
    },
    {
        "filename": "14_precision_recall_f1.png",
        "title": "Precision, recall and F1-score comparison",
        "description": "Compares risk-classifier precision, recall, and F1 across training cross-validation folds.",
    },
    {
        "filename": "15_confusion_matrix.png",
        "title": "Confusion matrix",
        "description": "Shows correct and incorrect at-risk predictions on the held-out student set.",
    },
    {
        "filename": "16_roc_curves.png",
        "title": "ROC curves",
        "description": "Plots true-positive against false-positive rates across thresholds; AUC summarizes each classifier.",
    },
    {
        "filename": "17_precision_recall_curves.png",
        "title": "Precision-recall curves",
        "description": "Shows the precision-recall trade-off across thresholds; average precision summarizes each classifier.",
    },
    {
        "filename": "18_actual_vs_predicted.png",
        "title": "Actual vs predicted outcomes",
        "description": "Compares recorded and predicted holdout averages; the diagonal represents exact predictions.",
    },
    {
        "filename": "19_cross_validation.png",
        "title": "Cross-validation performance",
        "description": "Shows the spread of fold-level MAE for each model; lower and tighter results are preferable.",
    },
    {
        "filename": "20_predicted_risk_distribution.png",
        "title": "Predicted academic-risk distribution",
        "description": "Shows out-of-fold risk probabilities by actual risk group; probabilities above 0.5 indicate predicted risk.",
    },
    {
        "filename": "21_student_classification.png",
        "title": "Student performance classification",
        "description": "Counts students by the official grade band of their second-semester average.",
    },
    {
        "filename": "22_actual_vs_predicted_classes.png",
        "title": "Actual vs predicted class distribution",
        "description": "Compares actual and predicted grade-band counts in the held-out student set.",
    },
    {
        "filename": "23_prediction_confidence.png",
        "title": "Prediction confidence distribution",
        "description": "Shows widths of 90% bootstrap prediction intervals; wider intervals mean greater uncertainty.",
    },
]
