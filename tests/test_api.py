from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get(
        "/health"
    )
    assert (
        response.status_code
        == 200
    )
    assert (
        response.json()[
            "status"
        ]
        == "ok"
    )

def test_classification():

    response = client.post(
        "/classify",
        json={
            "text":
                "Deep learning neural "
                "network for image "
                "classification"
        }
    )

    assert (
        response.status_code
        == 200
    )

    assert "label" in (
        response.json()
    )