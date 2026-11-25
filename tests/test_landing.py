from app import app


def test_landing_page_does_not_render_dataset_overview():
    response = app.test_client().get("/")

    assert response.status_code == 200
    assert b"Dataset Overview" not in response.data
    assert b"Understand your performance." in response.data
