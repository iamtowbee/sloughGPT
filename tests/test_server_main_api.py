"""
Tests for FastAPI app in apps/api/server/main.py (training resolve, v1 infer, request validation).

Imports use ``apps/api/server/`` on ``sys.path`` (same as ``uvicorn`` working directory) so
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
_SERVER_DIR = _REPO_ROOT / "apps" / "api" / "server"
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))


def _write_v1_manifest_bundle(
    tmp_path: Path,
    bundle_dir: str,
    *,
    dataset_id: str,
    version: str = "0.0.1",
    corpus_repeats: int = 100,
) -> Path:
    """Create ``bundle_dir/input.txt`` + ``dataset_manifest.json``; return resolved manifest path."""
    bundle = tmp_path / bundle_dir
    bundle.mkdir()
    (bundle / "input.txt").write_text("hello world " * corpus_repeats, encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "version": version,
        "domain": "general",
        "pii_policy": "none_expected",
        "sources": [{"type": "local_path", "uri": "./input.txt"}],
        "splits": {"train": "input.txt"},
    }
    mp = bundle / "dataset_manifest.json"
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    return mp.resolve()


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
    mp = _write_v1_manifest_bundle(
        tmp_path,
        "bundle_ref",
        dataset_id="pytest_ref_corpus",
        version="1.2.3",
    )
    r = client.post(
        "/train/resolve",
        json={
            "dataset_ref": {
                "dataset_id": "pytest_ref_corpus",
                "version": "1.2.3",
                "manifest_uri": str(mp),
            }
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert data["data_source"] == "ref"
    assert data["output_checkpoint_stem"] == "pytest_ref_corpus"


def test_train_resolve_manifest_happy_path(client: TestClient, tmp_path: Path) -> None:
    mp = _write_v1_manifest_bundle(tmp_path, "bundle", dataset_id="pytest_corpus")
    r = client.post("/train/resolve", json={"manifest_uri": str(mp)})
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


def test_training_start_accepts_extended_trainer_fields(
    client: TestClient, tmp_path: Path
) -> None:
    """POST /training/start validates extended SloughGPTTrainer-style JSON (Pydantic + job spawn)."""
    mp = _write_v1_manifest_bundle(
        tmp_path,
        "bundle_train_start",
        dataset_id="pytest_train_start_corpus",
        corpus_repeats=50,
    )

    body = {
        "name": "pytest-extended-req",
        "model": "sloughgpt",
        "manifest_uri": str(mp),
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "n_embed": 32,
        "n_layer": 1,
        "n_head": 1,
        "block_size": 16,
        "max_steps": 2,
        "log_interval": 1,
        "eval_interval": 2,
        "dropout": 0.05,
        "weight_decay": 0.0,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "use_mixed_precision": False,
        "mixed_precision_dtype": "fp16",
        "warmup_steps": 0,
        "min_lr": 1e-6,
        "scheduler": "cosine",
        "use_lora": True,
        "lora_rank": 4,
        "lora_alpha": 8,
        "checkpoint_dir": str(tmp_path / "ckpts_http"),
        "checkpoint_interval": 500,
        "save_best_only": False,
        "max_checkpoints": 2,
        "device": "cpu",
    }
    r = client.post("/training/start", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("status") == "running"
    assert "id" in data and str(data["id"]).startswith("job_")
    jobs = client.get("/training/jobs").json()
    assert isinstance(jobs, list)
    assert any(j.get("id") == data["id"] for j in jobs)


def test_training_start_rejects_invalid_mixed_precision_dtype(
    client: TestClient, tmp_path: Path
) -> None:
    mp = _write_v1_manifest_bundle(
        tmp_path,
        "bundle_bad_amp",
        dataset_id="pytest_bad_amp",
        corpus_repeats=20,
    )

    r = client.post(
        "/training/start",
        json={
            "name": "bad-amp",
            "model": "sloughgpt",
            "manifest_uri": str(mp),
            "mixed_precision_dtype": "float32",
        },
    )
    assert r.status_code == 422, r.text


def test_train_post_rejects_invalid_mixed_precision_dtype(
    client: TestClient, tmp_path: Path
) -> None:
    mp = _write_v1_manifest_bundle(
        tmp_path,
        "bundle_bad_amp_train",
        dataset_id="pytest_bad_amp_tr",
        corpus_repeats=20,
    )

    r = client.post(
        "/train",
        json={
            "manifest_uri": str(mp),
            "mixed_precision_dtype": "float32",
        },
    )
    assert r.status_code == 422, r.text


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


def test_models_load_forwards_device_in_local_mode(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def fake_load(model_id: str, mode: str = "local", **kwargs):
        captured["model_id"] = model_id
        captured["mode"] = mode
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("domains.training.model_registry.load_hf_model", fake_load)

    r = client.post(
        "/models/load",
        json={"model_id": "gpt2", "mode": "local", "device": "cpu"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "loaded"
    assert data["device"] == "cpu"
    assert captured["model_id"] == "gpt2"
    assert captured["mode"] == "local"
    assert captured["kwargs"].get("device") == "cpu"
    assert data["model_type"] == "hf/gpt2"
    assert data.get("effective_device") is None


def test_models_load_omits_device_for_api_mode(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def fake_load(model_id: str, mode: str = "local", **kwargs):
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("domains.training.model_registry.load_hf_model", fake_load)

    r = client.post(
        "/models/load",
        json={"model_id": "gpt2", "mode": "api", "device": "cpu"},
    )
    assert r.status_code == 200, r.text
    assert "device" not in captured["kwargs"]
    data = r.json()
    assert data.get("status") == "loaded"
    assert data["model_type"] == "hf/gpt2"


def test_models_load_wires_globals_for_local_hf_client(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import main as app_main

    saved_model, saved_tok, saved_type = app_main.model, app_main.tokenizer, app_main.model_type
    try:

        class _FakeLoader:
            def __init__(self) -> None:
                self.model = object()
                self.tokenizer = object()

        class _FakeHF:
            def __init__(self, loader: _FakeLoader) -> None:
                self._client = loader

        fake_loader = _FakeLoader()

        def fake_load(model_id: str, mode: str = "local", **_kwargs):  # noqa: ANN001
            assert model_id == "gpt2"
            assert mode == "local"
            return _FakeHF(fake_loader)

        monkeypatch.setattr("domains.training.model_registry.load_hf_model", fake_load)

        r = client.post("/models/load", json={"model_id": "gpt2", "mode": "local"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "loaded"
        assert body["model_type"] == "gpt2"
        assert body.get("effective_device") == "cpu"

        h = client.get("/health")
        assert h.status_code == 200, h.text
        assert h.json()["model_loaded"] is True
        assert app_main.model is fake_loader.model
        assert app_main.tokenizer is fake_loader.tokenizer
    finally:
        app_main.model, app_main.tokenizer, app_main.model_type = saved_model, saved_tok, saved_type


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
    if data.get("host"):
        h = data["host"]
        assert "cpu_percent" in h
        assert isinstance(h["cpu_percent"], (int, float))
        assert "memory_total_bytes" in h
        assert "memory_percent" in h
        if h.get("process_rss_bytes") is not None:
            assert isinstance(h["process_rss_bytes"], int)
            assert h["process_rss_bytes"] >= 0


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


def test_inference_generate_empty_prompt_domain_422(client: TestClient) -> None:
    r = client.post("/inference/generate", json={"prompt": "  ", "max_new_tokens": 4})
    assert r.status_code == 422, r.text
    data = r.json()
    assert data.get("code") == "empty_prompt"
    assert "error" in data


def test_generate_empty_prompt_domain_422(client: TestClient) -> None:
    r = client.post("/generate", json={"prompt": ""})
    assert r.status_code == 422, r.text
    assert r.json().get("code") == "empty_prompt"


def test_chat_post_empty_messages_400(client: TestClient) -> None:
    r = client.post("/chat", json={"messages": []})
    assert r.status_code == 400, r.text


def test_chat_post_json_shape(client: TestClient) -> None:
    """``POST /chat`` mirrors inference: ``text`` + optional error when engine unset."""
    r = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hello"}], "max_new_tokens": 8},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "text" in data
    assert isinstance(data["text"], str)
    if data.get("error"):
        assert data["text"] == ""
    else:
        assert isinstance(data.get("tokens_generated"), int)


def test_chat_stream_requires_messages(client: TestClient) -> None:
    r = client.post("/chat/stream", json={"messages": []})
    assert r.status_code == 400, r.text


def test_chat_stream_returns_sse_terminal_done(client: TestClient) -> None:
    """``/chat/stream`` ends with a ``done`` frame (same contract as inference stream)."""
    r = client.post(
        "/chat/stream",
        json={"messages": [{"role": "user", "content": "Hi"}], "max_new_tokens": 8},
    )
    assert r.status_code == 200, r.text
    assert "event-stream" in (r.headers.get("content-type") or "")
    lines = [ln for ln in r.text.splitlines() if ln.startswith("data:")]
    assert lines, r.text
    last = json.loads(lines[-1].split("data:", 1)[1].strip())
    assert last.get("done") is True


def test_inference_stats_json_shape(client: TestClient) -> None:
    """``GET /inference/stats`` returns either engine stats or a clear error object."""
    r = client.get("/inference/stats")
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data, dict)
    if data.get("error"):
        assert "Engine" in data["error"] or "engine" in data["error"].lower()


def test_inference_batch_returns_results_wrapped(client: TestClient) -> None:
    r = client.post(
        "/inference/batch",
        json={"prompts": ["Hello", "World"], "max_new_tokens": 8},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data.get("results"), list)
    assert len(data["results"]) == 2
    assert data.get("count") == 2
    assert "cache_stats" in data


def test_get_soul_returns_status(client: TestClient) -> None:
    r = client.get("/soul")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("status") in ("no_soul", "loaded")


def test_delete_cache_returns_stats(client: TestClient) -> None:
    r = client.delete("/cache")
    assert r.status_code == 200, r.text
    data = r.json()
    assert "cache_stats" in data


def test_inference_quantize_no_model_or_ok(client: TestClient) -> None:
    r = client.post("/inference/quantize", json={"quantization_type": "fp16"})
    assert r.status_code == 200, r.text
    data = r.json()
    if data.get("error"):
        assert "model" in data["error"].lower()
    else:
        assert data.get("status") == "quantized"


def test_list_experiments_returns_json_array(client: TestClient) -> None:
    r = client.get("/experiments")
    assert r.status_code == 200, r.text
    assert isinstance(r.json(), list)


def test_get_experiment_unknown_returns_404(client: TestClient) -> None:
    r = client.get("/experiments/nonexistent_experiment___")
    assert r.status_code == 404


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


def test_benchmark_run_no_model_or_result_dict(client: TestClient) -> None:
    r = client.post(
        "/benchmark/run",
        params={"max_new_tokens": 4, "num_runs": 1},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    if data.get("error"):
        assert "model" in data["error"].lower()
    else:
        assert isinstance(data, dict)


def test_model_export_no_model_returns_error(client: TestClient) -> None:
    r = client.post(
        "/model/export",
        json={"output_path": "models/out", "format": "safetensors"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    if data.get("error"):
        assert "model" in data["error"].lower()
    else:
        assert data.get("status") == "exported"


def test_generate_demo_returns_text_without_model(client: TestClient) -> None:
    r = client.post(
        "/generate/demo",
        json={"prompt": "Hello from pytest", "max_new_tokens": 16},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("model") == "demo"
    assert isinstance(data.get("text"), str)
    assert data["text"]


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
