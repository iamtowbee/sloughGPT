"""Pydantic models for training HTTP API (dataset manifests, job payloads)."""

from __future__ import annotations

from pydantic import BaseModel, model_validator


class TrainDatasetRef(BaseModel):
    """Standard v1 dataset pointer (see standards/v1/schemas/dataset_manifest.json)."""

    dataset_id: str
    version: str
    manifest_uri: str


class TrainDataSourceBody(BaseModel):
    """Exactly one training corpus selector (legacy folder, manifest file, or versioned ref)."""

    dataset: str | None = None
    manifest_uri: str | None = None
    dataset_ref: TrainDatasetRef | None = None

    @model_validator(mode="after")
    def _exactly_one_dataset_source(self) -> TrainDataSourceBody:
        has_d = self.dataset is not None and str(self.dataset).strip() != ""
        has_m = self.manifest_uri is not None and str(self.manifest_uri).strip() != ""
        has_r = self.dataset_ref is not None
        if sum(bool(x) for x in (has_d, has_m, has_r)) != 1:
            raise ValueError(
                "Specify exactly one of: `dataset` (folder under datasets/), "
                "`manifest_uri`, or `dataset_ref`."
            )
        return self


class TrainRequest(TrainDataSourceBody):
    """Train char-level model from ``datasets/<name>/input.txt`` or from a v1 manifest."""

    epochs: int | None = 3
    batch_size: int | None = 32
    learning_rate: float | None = 1e-3
    n_embed: int | None = 128
    n_layer: int | None = 4
    n_head: int | None = 4
    block_size: int | None = 128
    max_steps: int | None = None


class TrainResolveRequest(TrainDataSourceBody):
    """Preview resolved training file path without starting a job."""

    pass


class TrainingRequest(TrainDataSourceBody):
    """UI/orchestrator training job (metadata + same corpus selectors as ``TrainRequest``)."""

    name: str
    model: str
    epochs: int | None = 3
    batch_size: int | None = 32
    learning_rate: float | None = 1e-3
    n_embed: int | None = 128
    n_layer: int | None = 4
    n_head: int | None = 4
    block_size: int | None = 128
    max_steps: int | None = None
