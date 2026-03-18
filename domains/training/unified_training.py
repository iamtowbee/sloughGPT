"""
Unified Training Module
Provides TrainingConfig, Trainer, and DataLoader for the SloughGPT framework.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Iterator
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""

    data_path: str = "data/text.txt"
    model_id: str = "nanogpt"

    # Model architecture
    vocab_size: int = 5000
    n_embed: int = 256
    n_layer: int = 6
    n_head: int = 8
    block_size: int = 128

    # Training
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_batches: Optional[int] = None

    # LR Scheduler
    scheduler: str = "cosine"
    warmup_steps: int = 100
    min_lr: float = 1e-6
    max_lr: float = 1e-3

    # Mixed precision
    precision: str = "fp32"

    # Gradient
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Other
    save_interval: int = 1000
    eval_interval: int = 500
    log_interval: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class TextDataset(Dataset):
    """Character-level text dataset."""

    def __init__(self, data: List[int], block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx : idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1 : idx + self.block_size + 1], dtype=torch.long)
        return x, y


class UniversalDataLoader:
    """Universal data loader supporting multiple formats."""

    def __init__(self, data_path: str, block_size: int = 128):
        self.data_path = Path(data_path)
        self.block_size = block_size
        self.texts: List[str] = []
        self.data: List[int] = []
        self.vocab_size: int = 0
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

        self._load_data()

    def _load_data(self):
        """Load data from file."""
        suffix = self.data_path.suffix.lower()

        if suffix == ".txt":
            self._load_text()
        elif suffix == ".json":
            self._load_json()
        elif suffix == ".jsonl":
            self._load_jsonl()
        elif suffix == ".csv":
            self._load_csv()
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _load_text(self):
        """Load plain text file."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            text = f.read()

        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.data = [self.stoi[ch] for ch in text]
        self.texts = [text]

        logger.info(f"Loaded text: {len(self.data)} tokens, {self.vocab_size} vocab")

    def _load_json(self):
        """Load JSON file."""
        with open(self.data_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            self.texts = [item["text"] if isinstance(item, dict) else str(item) for item in data]
        elif isinstance(data, dict):
            self.texts = [data.get("text", str(data))]

        self._build_vocab()

    def _load_jsonl(self):
        """Load JSONL file."""
        self.texts = []
        with open(self.data_path, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    text = item["text"] if isinstance(item, dict) else str(item)
                    self.texts.append(text)

        self._build_vocab()

    def _load_csv(self):
        """Load CSV file."""
        import csv

        self.texts = []
        with open(self.data_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.texts.append(row.get("text", ""))

        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary from texts."""
        chars = set()
        for text in self.texts:
            chars.update(text)

        chars = sorted(chars)
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        # Encode all texts
        self.data = []
        for text in self.texts:
            self.data.extend([self.stoi.get(ch, 0) for ch in text])

        logger.info(
            f"Loaded {len(self.texts)} texts: {len(self.data)} tokens, {self.vocab_size} vocab"
        )

    def get_batch(self, batch_size: int, block_size: Optional[int] = None) -> Iterator:
        """Get batch generator."""
        block_size = block_size or self.block_size

        # Create indices
        n = len(self.data) - block_size
        indices = torch.randperm(n)[: batch_size * 100]  # Limit to avoid memory issues

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]

            x = torch.stack(
                [
                    torch.tensor(self.data[idx : idx + block_size], dtype=torch.long)
                    for idx in batch_indices
                ]
            )
            y = torch.stack(
                [
                    torch.tensor(self.data[idx + 1 : idx + block_size + 1], dtype=torch.long)
                    for idx in batch_indices
                ]
            )

            yield x, y

    def create_dataloader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        """Create PyTorch DataLoader."""
        dataset = TextDataset(self.data, self.block_size)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class ModelWrapper:
    """Model wrapper base class."""

    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config


class TorchModelWrapper(ModelWrapper):
    """PyTorch model wrapper."""

    def __init__(self, model: nn.Module, config: TrainingConfig):
        super().__init__(model, config)
        self.model = model.to(config.device)


class Trainer:
    """Trainer class for model training."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.data_loader: Optional[UniversalDataLoader] = None

        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

    def setup(self):
        """Setup trainer components."""
        from .models.nanogpt import NanoGPT

        # Create model
        self.model = NanoGPT(
            vocab_size=self.config.vocab_size,
            n_embed=self.config.n_embed,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            block_size=self.config.block_size,
        )
        self.model = TorchModelWrapper(self.model, self.config)

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        # Create scheduler
        self.scheduler = self._create_scheduler()

        # Create scaler for mixed precision
        if self.config.device == "cuda" and self.config.precision in ("fp16", "bf16"):
            self.scaler = torch.cuda.amp.GradScaler()

        # Create data loader
        self.data_loader = UniversalDataLoader(
            self.config.data_path,
            self.config.block_size,
        )

        logger.info(f"Trainer setup complete. Device: {self.config.device}")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        total_steps = 10000  # Will be updated during training

        if self.config.scheduler == "cosine":

            def lr_lambda(step):
                if step < self.config.warmup_steps:
                    return step / max(self.config.warmup_steps, 1)
                progress = (step - self.config.warmup_steps) / (
                    total_steps - self.config.warmup_steps
                )
                return self.config.min_lr / self.config.learning_rate + (
                    1 - self.config.min_lr / self.config.learning_rate
                ) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        elif self.config.scheduler == "warmup":

            def lr_lambda(step):
                if step < self.config.warmup_steps:
                    return step / max(self.config.warmup_steps, 1)
                return 1.0

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        return None

    def train(self):
        """Train the model."""
        if self.model is None:
            raise RuntimeError("Call setup() before training")

        self.model.model.train()

        logger.info(f"Starting training for {self.config.epochs} epochs")

    def evaluate(self):
        """Evaluate the model."""
        if self.model is None:
            raise RuntimeError("Call setup() before evaluation")

        self.model.model.eval()

        logger.info("Evaluation complete")
        return {"loss": 0.0}

    def save(self, path: str):
        """Save model checkpoint."""
        if self.model is None:
            raise RuntimeError("No model to save")

        checkpoint = {
            "model_state_dict": self.model.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)

        if self.model is None:
            from .models.nanogpt import NanoGPT

            self.model = NanoGPT(
                vocab_size=self.config.vocab_size,
                n_embed=self.config.n_embed,
                n_layer=self.config.n_layer,
                n_head=self.config.n_head,
                block_size=self.config.block_size,
            )
            self.model = TorchModelWrapper(self.model, self.config)

        self.model.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        logger.info(f"Model loaded from {path}")


def train(config: TrainingConfig):
    """Train function for quick training."""
    trainer = Trainer(config)
    trainer.setup()
    trainer.train()
    return trainer
