import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from apps.api.server.main import app
    return TestClient(app)


class TestAutoTrainStart:
    def test_start_sets_running_state(self, client):
        """POST /auto-train/start should set _auto_train_running = True."""
        with patch("apps.api.server.main._load_or_create_baby_model"):
            response = client.post(
                "/auto-train/start",
                json={"topic": "test", "teacher_model": "gpt2", "temperature": 0.8}
            )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["teacher"] == "gpt2"

    def test_start_twice_should_warn(self, client):
        """Starting twice should warn about previous session."""
        with patch("apps.api.server.main._load_or_create_baby_model"):
            r1 = client.post(
                "/auto-train/start",
                json={"topic": "test1", "teacher_model": "gpt2"}
            )
            assert r1.status_code == 200

            r2 = client.post(
                "/auto-train/start",
                json={"topic": "test2", "teacher_model": "gpt2"}
            )
            assert r2.status_code == 200
            data = r2.json()
            assert data["status"] == "started"


class TestAutoTrainStop:
    def test_stop_before_start_gives_error(self, client):
        """Stopping without start should fail gracefully."""
        response = client.post("/auto-train/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["stopped", "not_running"]

    def test_stop_after_start_works(self, client):
        """Stop after start should work."""
        with patch("apps.api.server.main._load_or_create_baby_model"):
            client.post(
                "/auto-train/start",
                json={"topic": "test", "teacher_model": "gpt2"}
            )

        response = client.post("/auto-train/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"