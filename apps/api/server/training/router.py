"""FastAPI routes for char-level training and job orchestration.

Trainer ``step_*.pt`` charset maps: ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
"""

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


def _sloughgpt_trainer_kwds(req_snapshot: dict[str, Any]) -> dict[str, Any]:
    """Build ``SloughGPTTrainer`` keyword arguments from a request ``model_dump()`` (except ``data_path``)."""
    device = req_snapshot.get("device")
    return {
        "n_embed": int(req_snapshot.get("n_embed") or 128),
        "n_layer": int(req_snapshot.get("n_layer") or 4),
        "n_head": int(req_snapshot.get("n_head") or 4),
        "block_size": int(req_snapshot.get("block_size") or 128),
        "dropout": float(req_snapshot.get("dropout") if req_snapshot.get("dropout") is not None else 0.1),
        "batch_size": int(req_snapshot.get("batch_size") or 32),
        "epochs": int(req_snapshot.get("epochs") or 3),
        "lr": float(req_snapshot.get("learning_rate") or 1e-3),
        "max_steps": req_snapshot.get("max_steps"),
        "gradient_accumulation_steps": int(req_snapshot.get("gradient_accumulation_steps") or 1),
        "max_grad_norm": float(req_snapshot.get("max_grad_norm") if req_snapshot.get("max_grad_norm") is not None else 1.0),
        "use_mixed_precision": bool(req_snapshot.get("use_mixed_precision", True)),
        "mixed_precision_dtype": str(req_snapshot.get("mixed_precision_dtype") or "bf16"),
        "checkpoint_dir": str(req_snapshot.get("checkpoint_dir") or "checkpoints"),
        "checkpoint_interval": int(req_snapshot.get("checkpoint_interval") or 500),
        "save_best_only": bool(req_snapshot.get("save_best_only", False)),
        "max_checkpoints": int(req_snapshot.get("max_checkpoints") or 5),
        "scheduler_type": str(req_snapshot.get("scheduler") or "cosine"),
        "warmup_steps": int(req_snapshot.get("warmup_steps") if req_snapshot.get("warmup_steps") is not None else 100),
        "min_lr": float(req_snapshot.get("min_lr") if req_snapshot.get("min_lr") is not None else 1e-5),
        "weight_decay": float(req_snapshot.get("weight_decay") if req_snapshot.get("weight_decay") is not None else 0.01),
        "use_lora": bool(req_snapshot.get("use_lora", False)),
        "lora_rank": int(req_snapshot.get("lora_rank") or 8),
        "lora_alpha": int(req_snapshot.get("lora_alpha") or 16),
        "log_interval": int(req_snapshot.get("log_interval") or 10),
        "eval_interval": int(req_snapshot.get("eval_interval") or 100),
        "device": device if device is not None and str(device).strip() != "" else None,
    }


@router.post("/train")
async def train(request: TrainRequest):
    """Start a training job (background thread).

    ``SloughGPTTrainer`` writes periodic ``step_*.pt`` under ``checkpoint_dir`` with
    ``stoi`` / ``itos`` / ``chars`` for char-LM eval; see
    ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
    """
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
                **_sloughgpt_trainer_kwds(req_snapshot),
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
    """Resolve ``data_path`` and checkpoint stem (dry run; no training).

    Does not write ``.pt`` artifacts. After ``POST /train`` or ``POST /training/start``,
    native ``step_*.pt`` includes char vocab; see ``docs/policies/CONTRIBUTING.md``
    (*Checkpoint vocabulary*).
    """
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
    """List all tracked training jobs.

    Completed jobs may expose ``checkpoint``; native ``step_*.pt`` uses charset maps for
    ``cli.py eval`` — ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
    """
    return list(training_jobs.values())


@router.get("/training/jobs/{job_id}")
async def get_training_job(job_id: str):
    """Get one training job by id.

    See list endpoint docstring for ``checkpoint`` / ``step_*.pt`` vocabulary semantics.
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return training_jobs[job_id]


@router.post("/training/start")
async def start_training(request: TrainingRequest):
    """Start a tracked training job (web UI).

    ``step_*.pt`` files saved on the server include ``stoi`` / ``itos`` / ``chars``
    for char-LM eval; see ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
    """
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
        "global_step": 0,
        "loss": None,
        "train_loss": None,
        "eval_loss": None,
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
            def on_progress(info: dict[str, Any]) -> None:
                rec = training_jobs.get(jid)
                if not rec:
                    return
                rec["progress"] = int(info.get("progress_percent", rec.get("progress", 0)))
                rec["current_epoch"] = int(info.get("epoch", rec.get("current_epoch", 0)))
                rec["global_step"] = int(info.get("global_step", 0))
                tl = info.get("train_loss")
                if tl is not None:
                    rec["train_loss"] = float(tl)
                el = info.get("eval_loss")
                if el is not None:
                    fe = float(el)
                    rec["eval_loss"] = fe
                    rec["loss"] = fe

            trainer = SloughGPTTrainer(
                data_path=data_path_for_thread,
                **_sloughgpt_trainer_kwds(req_snapshot),
            )
            result = trainer.train(on_progress=on_progress)
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
