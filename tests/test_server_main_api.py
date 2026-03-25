"""
Tests for FastAPI app in server/main.py (training resolve, v1 infer, request validation).

Imports use ``server/`` on ``sys.path`` (same as ``uvicorn`` working directory) so
``federated_routes`` and other sibling modules resolve.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SERVER_DIR = _REPO_ROOT / "server"
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))


@pytest.fixture(scope="module")
def client() -> TestClient:
    from main import app

    return TestClient(app)


def test_train_resolve_rejects_duplicate_sources(client: TestClient) -> None:
    r = client.post(
        "/train/resolve",
        json={"dataset": "foo", "manifest_uri": "bar.json"},
    )
    assert r.status_code == 422


def test_train_resolve_rejects_missing_legacy_dataset(client: TestClient) -> None:
    r = client.post("/train/resolve", json={"dataset": "nonexistent_folder___xyz"})
    assert r.status_code == 400
    body = r.json()
    detail = body.get("detail") or body.get("error") or ""
    assert "Missing training file" in str(detail)


def test_train_resolve_dataset_ref_happy_path(client: TestClient, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle_ref"
    bundle.mkdir()
    (bundle / "input.txt").write_text("hello world " * 100, encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "dataset_id": "pytest_ref_corpus",
        "version": "1.2.3",
        "domain": "general",
        "pii_policy": "none_expected",
        "sources": [{"type": "local_path", "uri": "./input.txt"}],
        "splits": {"train": "input.txt"},
    }
    mp = bundle / "dataset_manifest.json"
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    r = client.post(
        "/train/resolve",
        json={
            "dataset_ref": {
                "dataset_id": "pytest_ref_corpus",
                "version": "1.2.3",
                "manifest_uri": str(mp.resolve()),
            }
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert data["data_source"] == "ref"
    assert data["output_checkpoint_stem"] == "pytest_ref_corpus"


def test_train_resolve_manifest_happy_path(client: TestClient, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "input.txt").write_text("hello world " * 100, encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "dataset_id": "pytest_corpus",
        "version": "0.0.1",
        "domain": "general",
        "pii_policy": "none_expected",
        "sources": [{"type": "local_path", "uri": "./input.txt"}],
        "splits": {"train": "input.txt"},
    }
    mp = bundle / "dataset_manifest.json"
    mp.write_text(json.dumps(manifest), encoding="utf-8")

    r = client.post("/train/resolve", json={"manifest_uri": str(mp.resolve())})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert data["data_source"] == "manifest"
    assert data["output_checkpoint_stem"] == "pytest_corpus"
    assert "input.txt" in data["data_path"]


def test_list_training_jobs_returns_json_array(client: TestClient) -> None:
    r = client.get("/training/jobs")
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data, list)


def test_root_returns_api_metadata(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("name") == "SloughGPT API"
    assert "endpoints" in data
    assert "v1_infer" in data["endpoints"]


def test_list_datasets_returns_wrapped_array(client: TestClient) -> None:
    r = client.get("/datasets")
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data.get("datasets"), list)


def test_health_returns_status(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("status") == "healthy"
    assert "model_loaded" in data
    assert "model_type" in data


def test_health_live_returns_alive(client: TestClient) -> None:
    r = client.get("/health/live")
    assert r.status_code == 200, r.text
    assert r.json().get("status") == "alive"


def test_health_ready_includes_model_loaded(client: TestClient) -> None:
    r = client.get("/health/ready")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("status") in ("ready", "initializing")
    assert isinstance(data.get("model_loaded"), bool)
    assert "model_type" in data


def test_health_detailed_json_shape(client: TestClient) -> None:
    r = client.get("/health/detailed")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("status") == "healthy"
    assert isinstance(data.get("model_loaded"), bool)
    assert "model_type" in data
    assert isinstance(data.get("rate_limiter"), dict)
    assert isinstance(data.get("websocket"), dict)
    assert isinstance(data.get("security"), dict)
    assert isinstance(data.get("system"), dict)


def test_security_keys_json_shape(client: TestClient) -> None:
    r = client.get("/security/keys")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("rate_limiting_enabled") is True
    assert isinstance(data.get("jwt_auth_enabled"), bool)
    assert isinstance(data.get("api_keys_configured"), int)


def test_security_audit_returns_logs_array(client: TestClient) -> None:
    r = client.get("/security/audit")
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data.get("logs"), list)


def test_personalities_returns_list_or_error(client: TestClient) -> None:
    r = client.get("/personalities")
    assert r.status_code == 200, r.text
    data = r.json()
    if data.get("error"):
        assert isinstance(data["error"], str)
    else:
        assert isinstance(data.get("personalities"), list)


def test_get_dataset_unknown_returns_404(client: TestClient) -> None:
    r = client.get("/datasets/nonexistent_dataset___xyz___")
    assert r.status_code == 404


def test_info_returns_api_version_and_model_block(client: TestClient) -> None:
    r = client.get("/info")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("api_version")
    assert isinstance(data.get("model"), dict)
    assert "type" in data["model"]
    assert "loaded" in data["model"]
    assert isinstance(data["model"]["loaded"], bool)
    assert data.get("pytorch_version")


def test_list_models_returns_wrapped_array(client: TestClient) -> None:
    r = client.get("/models")
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data.get("models"), list)


def test_inference_generate_stream_returns_sse_terminal_done(client: TestClient) -> None:
    """Stream must end with a done frame; avoid crashing when inference engine is unset."""
    r = client.post(
        "/inference/generate/stream",
        json={"prompt": "Hello", "max_new_tokens": 8},
    )
    assert r.status_code == 200, r.text
    assert "event-stream" in (r.headers.get("content-type") or "")
    lines = [ln for ln in r.text.splitlines() if ln.startswith("data:")]
    assert lines, r.text
    last = json.loads(lines[-1].split("data:", 1)[1].strip())
    assert last.get("done") is True


def test_inference_generate_json_shape(client: TestClient) -> None:
    """Non-stream inference returns ``text``; errors use ``error`` + empty ``text``."""
    r = client.post(
        "/inference/generate",
        json={"prompt": "Hello", "max_new_tokens": 8},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "text" in data
    assert isinstance(data["text"], str)
    if data.get("error"):
        assert data["text"] == ""
    else:
        assert isinstance(data.get("tokens_generated"), int)


def test_inference_stats_json_shape(client: TestClient) -> None:
    """``GET /inference/stats`` returns either engine stats or a clear error object."""
    r = client.get("/inference/stats")
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data, dict)
    if data.get("error"):
        assert "Engine" in data["error"] or "engine" in data["error"].lower()


def test_list_experiments_returns_json_array(client: TestClient) -> None:
    r = client.get("/experiments")
    assert r.status_code == 200, r.text
    assert isinstance(r.json(), list)


def test_get_experiment_runs_unknown_returns_empty_list(client: TestClient) -> None:
    r = client.get("/experiments/nonexistent_exp___/runs")
    assert r.status_code == 200, r.text
    assert r.json() == []


def test_get_run_unknown_returns_404(client: TestClient) -> None:
    r = client.get("/runs/nonexistent_run___")
    assert r.status_code == 404


def test_metrics_prometheus_contains_sloughgpt_series(client: TestClient) -> None:
    r = client.get("/metrics/prometheus")
    assert r.status_code == 200, r.text
    assert r.headers.get("content-type", "").startswith("text/plain")
    assert b"sloughgpt_uptime_seconds" in r.content


def test_get_metrics_json_includes_system_block(client: TestClient) -> None:
    r = client.get("/metrics")
    assert r.status_code == 200, r.text
    data = r.json()
    assert "uptime" in data
    assert isinstance(data.get("system"), dict)
    assert "cpu_percent" in data["system"]
    assert "memory_percent" in data["system"]


def test_rate_limit_status_json_shape(client: TestClient) -> None:
    r = client.get("/rate-limit/status")
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data.get("requests_per_minute"), int)
    assert isinstance(data.get("burst_size"), int)
    assert isinstance(data.get("active_clients"), int)


def test_rate_limit_check_json_shape(client: TestClient) -> None:
    r = client.get("/rate-limit/check")
    assert r.status_code == 200, r.text
    data = r.json()
    assert "client_ip" in data
    assert isinstance(data.get("requests_used"), int)
    assert isinstance(data.get("requests_remaining"), int)
    assert "retry_after" in data


def test_get_export_formats_returns_formats_map(client: TestClient) -> None:
    r = client.get("/model/export/formats")
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data.get("formats"), dict)
    assert "safetensors" in data["formats"]


def test_registry_models_list_wrapped(client: TestClient) -> None:
    r = client.get("/registry/models")
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data.get("models"), list)


def test_cache_stats_json_shape(client: TestClient) -> None:
    r = client.get("/cache/stats")
    assert r.status_code == 200, r.text
    data = r.json()
    assert "size" in data
    assert isinstance(data.get("hits"), int)
    assert isinstance(data.get("misses"), int)
    assert isinstance(data.get("hit_rate"), (int, float))


def test_vector_stats_json_shape(client: TestClient) -> None:
    r = client.get("/vector/stats")
    assert r.status_code == 200, r.text
    data = r.json()
    assert "provider" in data
    assert isinstance(data.get("count"), int)


def test_benchmark_compare_returns_json_object(client: TestClient) -> None:
    """Either ``{\"error\": ...}`` when no model, or per-quantization rows when loaded."""
    r = client.get("/benchmark/compare")
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data, dict)
    if data.get("error"):
        assert "model" in data["error"].lower()
    else:
        assert any(k in data for k in ("fp32", "fp16", "int8"))


def test_registry_stats_json_shape(client: TestClient) -> None:
    r = client.get("/registry/stats")
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data.get("total_models"), int)
    assert isinstance(data.get("total_requests"), int)
    assert isinstance(data.get("total_tokens"), int)
    assert isinstance(data.get("by_status"), dict)
    assert isinstance(data.get("by_framework"), dict)


def test_registry_best_empty_or_model(client: TestClient) -> None:
    r = client.get("/registry/best")
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data, dict)
    assert data.get("error") == "No models found" or "id" in data


def test_registry_model_not_found_returns_404(client: TestClient) -> None:
    r = client.get("/registry/models/no_such_model___")
    assert r.status_code == 404


def test_registry_model_metrics_not_found_returns_404(client: TestClient) -> None:
    r = client.get("/registry/models/no_such_model___/metrics")
    assert r.status_code == 404


def test_dataset_stats_unknown_returns_404(client: TestClient) -> None:
    r = client.get("/datasets/nonexistent___/stats")
    assert r.status_code == 404


def test_benchmark_perplexity_json_shape(client: TestClient) -> None:
    """No model → error; otherwise perplexity + text_length."""
    r = client.post("/benchmark/perplexity", params={"text": "hello world"})
    assert r.status_code == 200, r.text
    data = r.json()
    if data.get("error"):
        assert "model" in data["error"].lower() or "text" in data["error"].lower()
    else:
        assert isinstance(data.get("perplexity"), (int, float))
        assert data.get("text_length") == len("hello world")


def test_v1_infer_rejects_invalid_mode(client: TestClient) -> None:
    r = client.post(
        "/v1/infer",
        json={
            "task_type": "general_generate",
            "mode": "not_a_mode",
            "input": {"prompt": "Hello"},
        },
    )
    assert r.status_code == 422


def test_v1_infer_generate_return_shape(client: TestClient) -> None:
    r = client.post(
        "/v1/infer",
        json={
            "task_type": "general_generate",
            "mode": "generate",
            "input": {"prompt": "Hello"},
            "generation": {"max_new_tokens": 8},
        },
    )
    assert r.status_code == 200, r.text
    assert r.headers.get("X-SloughGPT-Standard") == "1"
    body = r.json()
    assert "trace_id" in body
    assert body["task_type"] == "general_generate"
    assert body["output"]["text"] is not None


def test_v1_infer_chat_mode_return_shape(client: TestClient) -> None:
    r = client.post(
        "/v1/infer",
        json={
            "task_type": "chat",
            "mode": "chat",
            "input": {
                "messages": [
                    {"role": "user", "content": "Say hi in one short sentence."},
                ],
            },
            "generation": {"max_new_tokens": 16},
        },
    )
    assert r.status_code == 200, r.text
    assert r.headers.get("X-SloughGPT-Standard") == "1"
    body = r.json()
    assert body["mode"] == "chat"
    assert body["output"]["text"] is not None


def test_training_request_exactly_one_source() -> None:
    from pydantic import ValidationError

    from main import TrainingRequest

    with pytest.raises(ValidationError):
        TrainingRequest(name="a", model="m", dataset="x", manifest_uri="y.json")

    t = TrainingRequest(name="a", model="m", dataset="x")
    assert t.dataset == "x"
