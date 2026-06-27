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


def test_predict_today_endpoint():
    response = client.get("/api/predict/today")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert isinstance(data["cards"], list)


def test_ask_endpoint_matches_teams():
    response = client.post("/api/ask", json={"question": "巴西对德国谁会赢"})
    assert response.status_code == 200
    data = response.json()
    assert data["matched"] is True
    assert data["home_team"] == "Brazil"
    assert data["away_team"] == "Germany"
    assert "Brazil" in data["answer"]


def test_ask_endpoint_no_teams():
    response = client.post("/api/ask", json={"question": "今天天气怎么样"})
    assert response.status_code == 200
    assert response.json()["matched"] is False


def test_predict_match_stream_emits_reasoning():
    with client.stream(
        "POST",
        "/api/predict/match/stream",
        json={"home_team": "Brazil", "away_team": "Germany", "bankroll": 1000},
    ) as response:
        assert response.status_code == 200
        events = [
            line.split(":", 1)[1].strip()
            for line in response.iter_lines()
            if line.startswith("event:")
        ]
    assert "prediction" in events
    assert "reasoning" in events
    assert "node_start" in events
    assert "node_end" in events
    assert events[-1] == "done"

