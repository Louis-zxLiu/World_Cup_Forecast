from fastapi.testclient import TestClient

from apps.api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_llm_settings_round_trip_masks_key():
    payload = {
        "base_url": "https://example.com/v1",
        "api_key": "sk-test-123456",
        "model": "test-model",
        "temperature": 0.1,
        "timeout_seconds": 10,
        "enabled": False,
    }
    response = client.put("/api/settings/llm", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["base_url"] == "https://example.com/v1"
    assert data["api_key_masked"].startswith("sk-t")
    assert "123456" not in data["api_key_masked"]


def test_predict_match_endpoint():
    response = client.post(
        "/api/predict/match",
        json={"home_team": "Brazil", "away_team": "Germany", "bankroll": 1000},
    )

    assert response.status_code == 200
    data = response.json()
    assert "probabilities" in data
    assert len(data["agent_findings"]) >= 4
    assert len(data["bet_signals"]) == 3

