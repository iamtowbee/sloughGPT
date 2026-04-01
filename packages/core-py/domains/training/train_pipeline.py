#!/usr/bin/env python3
"""
SloughGPT Training Pipeline

Unified training with:
- Mixed precision (FP16/BF16)
- Gradient accumulation
- Automatic checkpointing
- Distributed training (DDP)
- LoRA support
- Learning rate scheduling
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from domains.models import SloughGPTModel
from domains.training.checkpoint_utils import torch_load_checkpoint
from domains.training.lora import apply_lora_to_model, LoRAConfig

logger = logging.getLogger("sloughgpt.trainer")


# =============================================================================
# Data Utilities
# =============================================================================

class TextDataset(Dataset):
    """Character-level text dataset."""

    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def prepare_data(data_path, block_size=128):
    """Prepare training data from text file."""
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    print(f"Data: {len(data)} tokens, {len(chars)} chars")
    return data, len(chars), stoi, itos


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainerConfig:
    """Training configuration with sensible defaults."""

    # Model
    vocab_size: int = 256
    n_embed: int = 256
    n_layer: int = 6
    n_head: int = 8
    block_size: int = 128
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    epochs: int = 10
    max_steps: Optional[int] = None
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Mixed precision
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "bf16"  # "fp16" or "bf16"

    # Distributed
    use_distributed: bool = False
    use_fsdp: bool = False  # Fully Sharded Data Parallel
    backend: str = "nccl"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 500
    save_best_only: bool = False
    max_checkpoints: int = 5

    # Scheduler
    scheduler_type: str = "cosine"
    warmup_steps: int = 100
    min_lr: float = 1e-5

    # LoRA
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16

    # Logging
    log_interval: int = 10
    eval_interval: int = 100

    # Device
    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Checkpoint Manager
# =============================================================================

class CheckpointManager:
    """Manages model checkpointing with automatic cleanup."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 5,
        save_best_only: bool = False,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.best_metric = float("inf")
        self.checkpoints: List[Dict[str, Any]] = []

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        step: int,
        metrics: Dict[str, float],
        config: TrainerConfig,
        epoch: int = 0,
        is_final: bool = False,
    ) -> Optional[str]:
        """Save a checkpoint."""
        metric_value = metrics.get("eval_loss", metrics.get("loss", float("inf")))

        if self.save_best_only and metric_value >= self.best_metric and not is_final:
            return None

        if metric_value < self.best_metric:
            self.best_metric = metric_value

        model_path = self.checkpoint_dir / f"step_{step}.pt"

        save_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "vocab_size": config.vocab_size,
                "n_embed": config.n_embed,
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "block_size": config.block_size,
            },
        }

        torch.save(save_dict, model_path)
        self.checkpoints.append({"step": step, "path": str(model_path), "metrics": metrics})

        logger.info(f"Checkpoint saved: {model_path} (step={step}, loss={metric_value:.4f})")

        self._cleanup_old_checkpoints()
        return str(model_path)

    @staticmethod
    def load_from_path(path: str, map_location: str = "cpu") -> Optional[Dict[str, Any]]:
        """Load a training checkpoint from an explicit path (.pt file)."""
        p = Path(path).expanduser()
        if not p.is_file():
            logger.warning("Checkpoint file not found: %s", p)
            return None
        return torch_load_checkpoint(str(p), map_location=map_location)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only the most recent ones."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return

        to_remove = self.checkpoints[:-self.max_checkpoints]
        self.checkpoints = self.checkpoints[-self.max_checkpoints:]

        for ckpt in to_remove:
            path = Path(ckpt["path"])
            if path.exists():
                path.unlink()

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1])
        )
        if not checkpoints:
            return None
        return torch_load_checkpoint(str(checkpoints[-1]), map_location="cpu")

    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load the checkpoint with the best metric."""
        if not self.checkpoints:
            return self.load_latest()
        best = min(self.checkpoints, key=lambda c: c["metrics"].get("eval_loss", float("inf")))
        path = Path(best["path"])
        if path.exists():
            return torch_load_checkpoint(str(path), map_location="cpu")
        return None


# =============================================================================
# Main Trainer
# =============================================================================

class SloughGPTTrainer:
    """
    Unified trainer for SloughGPTModel.

    Features:
    - Mixed precision (FP16/BF16)
    - Gradient accumulation
    - Automatic checkpointing
    - Distributed training (DDP)
    - LoRA fine-tuning
    - Learning rate scheduling
    """

    def __init__(
        self,
        data_path: str,
        config: Optional[TrainerConfig] = None,
        # Legacy parameters (for backward compatibility)
        vocab_size: Optional[int] = None,
        n_embed: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        block_size: int = 128,
        batch_size: int = 32,
        epochs: int = 10,
        lr: float = 1e-3,
        max_steps: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_mixed_precision: bool = True,
        mixed_precision_dtype: str = "bf16",
        checkpoint_dir: str = "checkpoints",
        checkpoint_interval: int = 500,
        save_best_only: bool = False,
        scheduler_type: str = "cosine",
        warmup_steps: int = 100,
        min_lr: float = 1e-5,
        weight_decay: float = 0.01,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        device: Optional[str] = None,
        soul_name: Optional[str] = None,
    ):
        # Handle both TrainerConfig and legacy parameters
        if config is not None:
            self.config = config
        else:
            self.config = TrainerConfig(
                vocab_size=vocab_size or 256,
                n_embed=n_embed,
                n_layer=n_layer,
                n_head=n_head,
                block_size=block_size,
                batch_size=batch_size,
                epochs=epochs,
                max_steps=max_steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                use_mixed_precision=use_mixed_precision,
                mixed_precision_dtype=mixed_precision_dtype,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval,
                save_best_only=save_best_only,
                scheduler_type=scheduler_type,
                warmup_steps=warmup_steps,
                min_lr=min_lr,
                weight_decay=weight_decay,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=device or "auto",
            )

        self.data_path = data_path
        self.soul_name = soul_name or "sloughgpt"
        self._best_val_loss = float("inf")
        self._train_loss_at_best = 0.0

        # Setup device
        self.device = self._setup_device()
        self.config.device = self.device

        # Initialize DDP state
        self.ddp_model: Optional[DDP] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.accumulation_step = 0

        print(f"Using device: {self.device}")

        # Prepare data
        self.data, self.vocab_size, self.stoi, self.itos = prepare_data(data_path, self.config.block_size)
        if self.config.vocab_size:
            self.vocab_size = self.config.vocab_size

        # Split data
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

        # Create model
        self._create_model()

        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Setup mixed precision
        self._setup_mixed_precision()

        # Setup checkpointing
        self.checkpoint_manager = CheckpointManager(
            self.config.checkpoint_dir,
            self.config.max_checkpoints,
            self.config.save_best_only,
        )

        # Training state
        self.global_step = 0
        self.current_epoch = 0

        print(f"\nTrain: {len(self.train_data)}, Val: {len(self.val_data)}")

    def _setup_device(self) -> str:
        """Setup training device."""
        if self.config.device != "auto":
            return self.config.device
        if torch.cuda.is_available():
            if self.config.use_distributed:
                torch.cuda.set_device(self.config.local_rank)
            return "cuda"
        return "cpu"

    def _create_model(self):
        """Create and setup the model."""
        print("\n=== Creating Model ===")
        self.model = SloughGPTModel(
            vocab_size=self.vocab_size,
            n_embed=self.config.n_embed,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            block_size=self.config.block_size,
            dropout=self.config.dropout,
        ).to(self.device)

        print(f"Model type: SloughGPTModel")
        print(f"  - RoPE position embeddings")
        print(f"  - SwiGLU activation")
        print(f"  - RMSNorm")
        print(f"  - SDPA attention")
        print(f"Base model: {self.model.num_parameters():,} params")

        # Apply LoRA
        if self.config.use_lora:
            print("\n=== Applying LoRA ===")
            lora_config = LoRAConfig(
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"],
            )
            self.model = apply_lora_to_model(self.model, config=lora_config)
            lora_params = sum(p.numel() for n, p in self.model.named_parameters() if "lora_" in n)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"LoRA params: {lora_params:,} ({100 * lora_params / total:.1f}%)")

        # Setup DDP
        if self.config.use_distributed:
            self._setup_distributed()

    def _setup_distributed(self):
        """Setup distributed training (DDP or FSDP)."""
        try:
            import torch.distributed as dist

            if self.config.use_fsdp and torch.cuda.is_available():
                self._setup_fsdp()
            else:
                self.ddp_model = DDP(
                    self.model,
                    device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
                    find_unused_parameters=False,
                )
                logger.info(f"DDP: rank={self.config.rank}, world_size={self.config.world_size}")
        except Exception as e:
            logger.error(f"Failed to setup distributed: {e}")
            self.config.use_distributed = False
            self.config.use_fsdp = False
            self.ddp_model = None

    def _setup_fsdp(self):
        """Setup Fully Sharded Data Parallel (FSDP)."""
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.api import ShardingStrategy

            strategy_map = {
                "FULL_SHARD": ShardingStrategy.FULL_SHARD,
                "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
                "NO_SHARD": ShardingStrategy.NO_SHARD,
            }
            strategy = strategy_map.get(self.config.sharding_strategy, ShardingStrategy.FULL_SHARD)

            self.ddp_model = FSDP(
                self.model,
                sharding_strategy=strategy,
                device_id=self.config.local_rank if torch.cuda.is_available() else None,
            )

            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"FSDP: rank={self.config.rank}, world_size={self.config.world_size}, "
                f"strategy={self.config.sharding_strategy}, params={total_params:,}"
            )
        except ImportError:
            logger.warning("FSDP not available. Requires PyTorch 2.0+ with distributed support.")
            logger.warning("Falling back to DDP.")
            self.config.use_fsdp = False
            self._setup_distributed()
        except Exception as e:
            logger.error(f"Failed to setup FSDP: {e}")
            logger.warning("Falling back to DDP.")
            self.config.use_fsdp = False
            self._setup_distributed()

    def _setup_mixed_precision(self):
        """Setup mixed precision training."""
        if not self.config.use_mixed_precision or not torch.cuda.is_available():
            return

        self.scaler = torch.cuda.amp.GradScaler()
        self.mixed_precision_dtype = torch.bfloat16 if self.config.mixed_precision_dtype == "bf16" else torch.float16
        print(f"Mixed precision: {self.mixed_precision_dtype}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        decay_params = []
        no_decay_params = []

        for n, p in self.model.named_parameters():
            if "bias" in n or "norm" in n:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
        )

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        from domains.training.lr_schedulers import create_scheduler

        if self.config.max_steps:
            total_steps = self.config.max_steps
        else:
            steps_per_epoch = len(self.train_data) // self.config.block_size // self.config.batch_size
            total_steps = steps_per_epoch * self.config.epochs

        return create_scheduler(
            self.optimizer,
            scheduler_type=self.config.scheduler_type,
            total_steps=total_steps,
            warmup_steps=self.config.warmup_steps,
            min_lr=self.config.min_lr,
        )

    @property
    def training_model(self) -> torch.nn.Module:
        """Get the model for training (handles DDP wrapping)."""
        return self.ddp_model if self.ddp_model is not None else self.model

    def get_batch(self, split: str = "train") -> tuple:
        """Get a batch of data."""
        data = self.train_data if split == "train" else self.val_data
        idx = torch.randint(0, len(data) - self.config.block_size, (self.config.batch_size,))
        x = torch.stack([data[i : i + self.config.block_size] for i in idx])
        y = torch.stack([data[i + 1 : i + self.config.block_size + 1] for i in idx])
        return x.to(self.device), y.to(self.device)

    def train_step(self) -> Dict[str, float]:
        """Execute a single training step."""
        model = self.training_model
        model.train()

        x, y = self.get_batch("train")
        scale_factor = 1.0 / self.config.gradient_accumulation_steps

        if self.scaler is not None:
            with torch.cuda.amp.autocast(dtype=self.mixed_precision_dtype):
                logits, loss = model(x, y)
                loss = loss * scale_factor

            self.scaler.scale(loss).backward()
        else:
            logits, loss = model(x, y)
            (loss * scale_factor).backward()

        self.accumulation_step += 1

        metrics = {"loss": loss.item() / scale_factor}

        if self.accumulation_step >= self.config.gradient_accumulation_steps:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)

            if self.scheduler is not None:
                self.scheduler.step()

            self.accumulation_step = 0

        return metrics

    @torch.no_grad()
    def evaluate(self, num_batches: int = 50) -> Dict[str, float]:
        """Evaluate the model."""
        model = self.training_model
        model.eval()

        total_loss = 0.0
        steps = 0

        for _ in range(num_batches):
            x, y = self.get_batch("val")
            _, loss = model(x, y)
            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        return {"eval_loss": avg_loss, "eval_ppl": torch.exp(torch.tensor(avg_loss)).item()}

    def train(self, resume: bool = False, resume_path: Optional[str] = None) -> Dict[str, Any]:
        """Full training loop.

        Args:
            resume: If True, load checkpoint from ``resume_path`` or latest in ``checkpoint_dir``.
            resume_path: Optional explicit ``.pt`` file (``train_sloughgpt`` / periodic / manual saves).
        """
        if resume:
            checkpoint = None
            if resume_path:
                checkpoint = CheckpointManager.load_from_path(resume_path, map_location="cpu")
            if checkpoint is None:
                checkpoint = self.checkpoint_manager.load_latest()
            if checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if self.scheduler and checkpoint.get("scheduler_state_dict"):
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                self.global_step = checkpoint.get("step", 0)
                self.current_epoch = int(checkpoint.get("epoch", 0))
                logger.info(
                    "Resumed from step %s epoch %s",
                    self.global_step,
                    self.current_epoch,
                )

        is_main = not self.config.use_distributed or self.config.rank == 0

        if is_main:
            logger.info(f"Training config: {self.config}")
            logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch

            if is_main:
                logger.info(f"\nEpoch {epoch + 1}/{self.config.epochs}")

            model = self.training_model
            model.train()

            train_loss = 0.0
            steps_per_epoch = len(self.train_data) // self.config.block_size // self.config.batch_size

            for step in range(steps_per_epoch):
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    break

                metrics = self.train_step()
                train_loss += metrics["loss"]
                self.global_step += 1

                if is_main and self.global_step % self.config.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
                    logger.info(
                        f"Step {self.global_step} | Loss: {metrics['loss']:.4f} | LR: {lr:.2e}"
                    )

                # Evaluation
                if self.global_step % self.config.eval_interval == 0:
                    eval_metrics = self.evaluate()
                    if is_main:
                        logger.info(
                            f"Eval | Loss: {eval_metrics['eval_loss']:.4f} | "
                            f"PPL: {eval_metrics['eval_ppl']:.2f}"
                        )

                        if eval_metrics["eval_loss"] < self._best_val_loss:
                            self._best_val_loss = eval_metrics["eval_loss"]
                            self.save_checkpoint(eval_metrics)

                # Checkpoint
                if self.global_step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint({"loss": metrics["loss"]})

            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        if is_main:
            self.save_checkpoint({"loss": 0.0}, is_final=True)

        return {"best_eval_loss": self._best_val_loss, "global_step": self.global_step}

    def save_checkpoint(self, metrics: Optional[Dict[str, float]] = None, is_final: bool = False):
        """Save a checkpoint."""
        metrics = metrics or {"loss": 0.0}
        self.checkpoint_manager.save(
            self.model,
            self.optimizer,
            self.scheduler,
            self.global_step,
            metrics,
            self.config,
            epoch=self.current_epoch,
            is_final=is_final,
        )

    def save(self, path: str, format: str = "sou"):
        """Save model in specified format."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        metadata = {
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            "itos": self.itos,
            "config": {
                "n_embed": self.config.n_embed,
                "n_layer": self.config.n_layer,
                "n_head": self.config.n_head,
                "block_size": self.config.block_size,
                "model_type": "sloughgpt",
            },
            "training_dataset": self.data_path,
            "epochs_trained": self.config.epochs,
            "final_val_loss": self._best_val_loss,
        }

        if format == "sou":
            from domains.inference.sou_format import create_soul_profile, export_to_sou
            soul = create_soul_profile(
                name=self.soul_name,
                base_model="sloughgpt",
                training_dataset=self.data_path,
                epochs_trained=self.config.epochs,
                final_val_loss=self._best_val_loss,
                lineage="sloughgpt",
                tags=["sloughgpt", "trained", "soul"],
            )
            output_path = path + ".sou"
            export_to_sou(self.model, output_path, soul_profile=soul)
        elif format == "safetensors":
            from domains.training.export import export_to_safetensors
            output_path = path + ".safetensors"
            export_to_safetensors(self.model, output_path, metadata)
        elif format == "torch":
            output_path = path + ".pt"
            torch.save({"model": self.model.state_dict(), **metadata}, output_path)
        else:
            from domains.training.export import export_to_safetensors
            output_path = path + ".safetensors"
            export_to_safetensors(self.model, output_path, metadata)

        print(f"Model saved to {output_path} ({format})")

    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.8) -> str:
        """Generate text."""
        self.model.eval()
        idx = torch.tensor([[self.stoi.get(c, 0) for c in prompt[:1]]])

        with torch.no_grad():
            output = self.model.generate(idx, max_new_tokens=max_tokens, temperature=temperature)

        text = "".join([self.itos.get(int(i), "?") for i in output[0]])
        return text


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point for standalone training."""
    import argparse

    parser = argparse.ArgumentParser(description="SloughGPT Training")
    parser.add_argument("--data", default="datasets/shakespeare/input.txt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-embed", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", action="store_true", default=True)
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--save-best-only", action="store_true")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    args = parser.parse_args()

    trainer = SloughGPTTrainer(
        data_path=args.data,
        n_embed=args.n_embed,
        n_layer=args.n_layer,
        n_head=args.n_head,
        block_size=args.block_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        use_mixed_precision=args.mixed_precision,
        mixed_precision_dtype=args.precision,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        save_best_only=args.save_best_only,
        use_lora=args.lora,
        lora_rank=args.lora_rank,
    )

    trainer.train()
    print("\n=== Generated Text ===")
    print(trainer.generate("First"))


if __name__ == "__main__":
    main()
