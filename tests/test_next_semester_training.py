import json

from openpyxl import Workbook
import pytest

from src.train_next_semester import train_and_save


def _write_course_results(path, headers, rows):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Course Results"
    sheet.append(headers)
    for row in rows:
        sheet.append(row)
    workbook.save(path)


def test_training_uses_only_observed_first_semester_scores_and_saves_aggregates(tmp_path):
    first_path = tmp_path / "first.xlsx"
    second_path = tmp_path / "second.xlsx"
    output_path = tmp_path / "model.json"
    first_rows = []
    second_rows = []

    for index in range(12):
        student_id = f"TEST{index:03d}"
        observed_average = 40 + index
        first_rows.extend([
            [student_id, "A", observed_average - 1, "Observed"],
            [student_id, "B", observed_average + 1, "Observed"],
            [student_id, "C", 99, "Estimated"],
        ])
        next_semester_average = 30 + index
        second_rows.extend([
            [student_id, "X", next_semester_average - 1],
            [student_id, "Y", next_semester_average + 1],
        ])

    _write_course_results(
        first_path,
        ["Student_ID", "Course_Code", "Score", "Data_Source"],
        first_rows,
    )
    _write_course_results(
        second_path,
        ["Student_ID", "Course_Code", "Score"],
        second_rows,
    )

    model = train_and_save(first_path, second_path, output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert model["training_students"] == 12
    assert model["first_semester_observed_course_scores"] == 24
    assert model["first_semester_estimated_course_scores_excluded"] == 12
    assert model["intercept"] == pytest.approx(-10)
    assert model["slope"] == pytest.approx(1)
    assert "TEST000" not in output_path.read_text(encoding="utf-8")
    assert saved == model
