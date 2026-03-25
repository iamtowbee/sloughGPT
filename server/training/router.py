"""FastAPI routes for char-level training and job orchestration."""

from __future__ import annotations

import logging
import threading
from typing import Any

from fastapi import APIRouter, HTTPException

from training.jobs import training_jobs
from training.resolution import resolve_training_inputs
from training.schemas import TrainingRequest, TrainRequest, TrainResolveRequest

logger = logging.getLogger("sloughgpt")

router = APIRouter(tags=["training"])


@router.post("/train")
async def train(request: TrainRequest):
    """Start a training job (background thread)."""
    from domains.training.dataset_manifest import ManifestError
    from domains.training.train_pipeline import SloughGPTTrainer

    try:
        data_path_str, out_stem, manifest_meta, source_kind = resolve_training_inputs(
            request.dataset,
            request.manifest_uri,
            request.dataset_ref,
        )
    except ManifestError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    req_snapshot = request.model_dump()

    def train_model() -> None:
        try:
            trainer = SloughGPTTrainer(
                data_path=data_path_str,
                n_embed=req_snapshot["n_embed"],
                n_layer=req_snapshot["n_layer"],
                n_head=req_snapshot["n_head"],
                block_size=req_snapshot["block_size"],
                batch_size=req_snapshot["batch_size"],
                epochs=req_snapshot["epochs"],
                lr=req_snapshot["learning_rate"],
                max_steps=req_snapshot["max_steps"],
            )
            trainer.train()
            safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in out_stem)[:120]
            trainer.save(f"models/{safe_stem}_trained.pt")
        except Exception as e:
            logger.exception("Background /train failed: %s", e)

    thread = threading.Thread(target=train_model, daemon=True)
    thread.start()

    out: dict[str, Any] = {
        "status": "started",
        "data_path": data_path_str,
        "output_checkpoint_stem": out_stem,
        "data_source": source_kind,
        "epochs": request.epochs,
        "message": "Training started in background",
    }
    if request.dataset is not None:
        out["dataset"] = request.dataset.strip()
    if manifest_meta is not None:
        out["manifest"] = manifest_meta
    return out


@router.post("/train/resolve")
async def train_resolve(body: TrainResolveRequest) -> dict[str, Any]:
    """Resolve ``data_path`` and checkpoint stem (dry run; no training)."""
    from domains.training.dataset_manifest import ManifestError

    try:
        data_path_str, out_stem, manifest_meta, source_kind = resolve_training_inputs(
            body.dataset,
            body.manifest_uri,
            body.dataset_ref,
        )
    except ManifestError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    out: dict[str, Any] = {
        "ok": True,
        "data_path": data_path_str,
        "output_checkpoint_stem": out_stem,
        "data_source": source_kind,
    }
    if body.dataset is not None:
        out["dataset"] = body.dataset.strip()
    if manifest_meta is not None:
        out["manifest"] = manifest_meta
    return out


@router.get("/train/status")
async def train_status():
    """Legacy training status stub."""
    return {"status": "ready", "message": "Use /train endpoint to start training"}


@router.get("/training/jobs")
async def list_training_jobs():
    """List all tracked training jobs."""
    return list(training_jobs.values())


@router.get("/training/jobs/{job_id}")
async def get_training_job(job_id: str):
    """Get one training job by id."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return training_jobs[job_id]


@router.post("/training/start")
async def start_training(request: TrainingRequest):
    """Start a tracked training job (web UI)."""
    from domains.training.dataset_manifest import ManifestError

    try:
        data_path_str, out_stem, manifest_meta, source_kind = resolve_training_inputs(
            request.dataset,
            request.manifest_uri,
            request.dataset_ref,
        )
    except ManifestError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    job_id = f"job_{len(training_jobs) + 1}"
    job: dict[str, Any] = {
        "id": job_id,
        "name": request.name,
        "model": request.model,
        "dataset": request.dataset.strip() if request.dataset else out_stem,
        "data_path": data_path_str,
        "output_checkpoint_stem": out_stem,
        "data_source": source_kind,
        "status": "running",
        "progress": 0,
        "epochs": request.epochs,
        "current_epoch": 0,
        "loss": None,
    }
    if manifest_meta is not None:
        job["manifest"] = manifest_meta
    training_jobs[job_id] = job

    req_snapshot = request.model_dump()
    data_path_for_thread = data_path_str
    out_stem_for_thread = out_stem
    jid = job_id

    def run_training() -> None:
        from domains.training.train_pipeline import SloughGPTTrainer

        try:
            training_jobs[jid]["progress"] = 5
            trainer = SloughGPTTrainer(
                data_path=data_path_for_thread,
                n_embed=int(req_snapshot.get("n_embed") or 128),
                n_layer=int(req_snapshot.get("n_layer") or 4),
                n_head=int(req_snapshot.get("n_head") or 4),
                block_size=int(req_snapshot.get("block_size") or 128),
                batch_size=int(req_snapshot.get("batch_size") or 32),
                epochs=int(req_snapshot.get("epochs") or 3),
                lr=float(req_snapshot.get("learning_rate") or 1e-3),
                max_steps=req_snapshot.get("max_steps"),
            )
            training_jobs[jid]["progress"] = 15
            result = trainer.train()
            safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in out_stem_for_thread)[:120]
            trainer.save(f"models/{safe_stem}_trained.pt")
            training_jobs[jid]["status"] = "completed"
            training_jobs[jid]["progress"] = 100
            training_jobs[jid]["current_epoch"] = int(req_snapshot.get("epochs") or 3)
            bel = result.get("best_eval_loss")
            training_jobs[jid]["loss"] = bel if bel is not None and bel < float("inf") else None
            training_jobs[jid]["checkpoint"] = f"models/{safe_stem}_trained.pt"
        except Exception as e:
            logger.exception("Training job %s failed", jid)
            training_jobs[jid]["status"] = "failed"
            training_jobs[jid]["error"] = str(e)
            training_jobs[jid]["progress"] = 0

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    return job
