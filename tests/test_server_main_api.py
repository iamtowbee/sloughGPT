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


def test_metrics_prometheus_contains_sloughgpt_series(client: TestClient) -> None:
    r = client.get("/metrics/prometheus")
    assert r.status_code == 200, r.text
    assert r.headers.get("content-type", "").startswith("text/plain")
    assert b"sloughgpt_uptime_seconds" in r.content


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


def test_training_request_exactly_one_source() -> None:
    from pydantic import ValidationError

    from main import TrainingRequest

    with pytest.raises(ValidationError):
        TrainingRequest(name="a", model="m", dataset="x", manifest_uri="y.json")

    t = TrainingRequest(name="a", model="m", dataset="x")
    assert t.dataset == "x"
