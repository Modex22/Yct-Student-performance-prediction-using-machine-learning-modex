import os

import pytest

import app as app_module
from src.history import save_entry
from src.predict import predict_performance


@pytest.fixture(autouse=True)
def _isolated_admin_db(tmp_path, monkeypatch):
    import src.config as config
    import src.history as history

    db_path = os.path.join(tmp_path, "admin_test.db")
    monkeypatch.setattr(config, "HISTORY_DB_PATH", db_path)
    monkeypatch.setattr(history, "HISTORY_DB_PATH", db_path)
    monkeypatch.setenv("ADMIN_USERNAME", "test-admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "test-password")


def test_admin_requires_login():
    client = app_module.app.test_client()
    response = client.get("/admin")
    assert response.status_code == 302
    assert "/admin/login" in response.headers["Location"]


def test_admin_login_and_dashboard():
    student = {"exam_score": 36, "test_score": 6, "assignment_score": 7, "practical_score": 13}
    save_entry("ND/CS/24/0001", "1234", student, predict_performance(student))

    client = app_module.app.test_client()
    response = client.post(
        "/admin/login",
        data={"username": "test-admin", "password": "test-password"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"ND/CS/24/0001" in response.data
    assert b"Saved predictions" in response.data
