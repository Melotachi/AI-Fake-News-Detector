from fastapi.testclient import TestClient
from backend import app, get_rate_limiter
from fastapi_limiter.depends import RateLimiter

app.dependency_overrides[get_rate_limiter] = lambda: lambda: None
client = TestClient(app)


def test_analyze_news():
    response = client.post("/analyze", json={"text": "The earth is flat."})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "score" in data
    assert "text" in data
    assert data["text"] == "The earth is flat."


def test_history_endpoint():
    response = client.get("/history")
    assert response.status_code == 200
    json_data = response.json()
    assert "history" in json_data
    assert isinstance(json_data["history"], list)


def test_feedback_submission():
    feedback_payload = {
        "text": "Sample feedback text.",
        "predicted_label": "Fake",
        "correct_label": "Real",
        "score": 0.78
    }
    response = client.post("/feedback", json=feedback_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "message" in data


def test_invalid_feedback_submission():
    invalid_payload = {
        "predicted_label": "Fake"
    }
    response = client.post("/feedback", json=invalid_payload)
    assert response.status_code == 422  # Missing required fields