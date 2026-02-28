"""Training script for the NanoGPT model using a live JSONL dataset.

The script demonstrates a minimal end‑to‑end pipeline:

1. Build a ``SimpleTokenizer`` from the first few lines of the data directory.
2. Create a :class:`LiveJSONLDataset` that streams token blocks of ``block_size``.
3. Wrap the dataset in a ``DataLoader`` (batch size configurable).
4. Instantiate ``NanoGPT`` (imported from ``domains.training.models.nanogpt``).
5. Set up an ``AdamW`` optimizer and a cosine learning‑rate scheduler with warm‑up.
6. Run a training loop with gradient clipping, periodic checkpointing, and optional
   validation (here we simply compute loss on a few batches from the same iterator).
7. Save the final checkpoint (model state, tokenizer, and config) using the helper
   functions from ``utils.py``.

The script can be invoked directly from the command line::

    python trainer.py --data_dir ./data --epochs 10 --batch_size 32 \
        --block_size 256 --lr 3e-4 --output_dir ./checkpoints

It is deliberately lightweight – the goal is to provide a clear, functional
example that can be extended (mixed‑precision, DDP, LoRA, etc.) without changing
the core logic.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Local imports
from utils import mkdir, save_json, load_json
from live_dataset import LiveJSONLDataset
from domains.training.models.nanogpt import NanoGPT

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def build_tokenizer_from_data(data_dir: str, sample_lines: int = 5) -> Any:
    """Create a ``SimpleTokenizer`` using the first ``sample_lines`` of data.

    The function reads a few lines from the first available ``*.jsonl`` file to
    collect enough characters for a reasonable vocabulary.  It returns the
    tokenizer instance.
    """
    for p in Path(data_dir).glob("*.jsonl"):
        with open(p, "r", encoding="utf-8") as fp:
            sample_text = ""
            for _ in range(sample_lines):
                line = fp.readline()
                if not line:
                    break
                try:
                    obj = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                sample_text += obj.get("prompt", "") + "\n" + obj.get("completion", "") + "\n"
            if sample_text:
                # Re‑use the SimpleTokenizer defined inside live_dataset for consistency
                from live_dataset import SimpleTokenizer
                return SimpleTokenizer(sample_text)
    raise RuntimeError("No JSONL files found to build tokenizer")


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    # auto fallback
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Simple cosine scheduler with warm‑up
def get_cosine_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int = 0) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)).item())
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    data_dir: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 32,
    block_size: int = 256,
    lr: float = 3e-4,
    device_str: str = "auto",
    checkpoint_interval: int = 1,
    amp: bool = False,
    distributed: bool = False,
    lora: bool = False,
    lora_rank: int = 4,
    lora_alpha: float = 1.0,
    log_dir: Optional[str] = None,
    grad_accum_steps: int = 1,
):
    # Prepare output directory
    mkdir(output_dir)

    # Device
    device = get_device(device_str)
    print(f"Using device: {device}")

    # Tokenizer (built from a small sample of the data)
    tokenizer = build_tokenizer_from_data(data_dir)
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {vocab_size}")

    # Dataset / DataLoader
    dataset = LiveJSONLDataset(data_dir=data_dir, block_size=block_size)
    # Distributed sampler if needed
    if distributed:
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = NanoGPT(vocab_size=vocab_size, block_size=block_size)
    # Apply LoRA if requested
    if lora:
        from lora import apply_lora_to_model
        model = apply_lora_to_model(model, rank=lora_rank, alpha=lora_alpha)
    model = model.to(device)
    # Distributed Data Parallel wrap if needed
    if distributed:
        # Initialize process group (expects env vars)
        if not dist.is_initialized():
            if isinstance(device, torch.device) and device.type == "cuda":
                torch.cuda.set_device(device)
            else:
                torch.cuda.set_device(0)
            dist.init_process_group(backend="nccl", init_method="env://")
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[device.index] if isinstance(device, torch.device) and device.type == "cuda" else None)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer & scheduler
    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
-    total_steps = epochs * (len(loader) if hasattr(loader, '__len__') else 1000)
-    scheduler = get_cosine_scheduler(optimizer, total_steps=total_steps, warmup_steps=int(0.1 * total_steps))
+    # Determine total steps for scheduler – fall back if dataset has no length (IterableDataset)
+    try:
+        total_steps = epochs * len(loader)
+    except TypeError:
+        total_steps = epochs * 1000
+    scheduler = get_cosine_scheduler(optimizer, total_steps=total_steps, warmup_steps=int(0.1 * total_steps))
*** End Patch
    # Mixed‑precision (AMP) setup
    scaler = GradScaler() if amp else None
    # Logging (TensorBoard) setup
    if log_dir:
        from torch.utils.tensorboard.writer import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None


    # Training
    model.train()
    global_step = 0
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batch_count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            # Mixed precision forward
            with autocast(enabled=amp):
                logits, loss = model(xb, yb)
            # Backward with scaling if using AMP
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            # Gradient accumulation
            if (global_step + 1) % grad_accum_steps == 0:
                # Unscale gradients and clip
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
            epoch_loss += loss.item()
            batch_count += 1
            global_step += 1
            # Logging
            if writer:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
        avg_loss = epoch_loss / max(1, batch_count)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{epochs} – loss: {avg_loss:.4f} – time: {elapsed:.1f}s")


        # Checkpoint
        # Checkpoint (only rank 0 writes when distributed)
        if not distributed or (distributed and dist.get_rank() == 0):
            ckpt_path = Path(output_dir) / f"checkpoint_epoch{epoch}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "tokenizer": tokenizer,
                    "config": {
                        "vocab_size": vocab_size,
                        "block_size": block_size,
                        "n_embed": model.blocks[0].attn.n_embed if hasattr(model, "blocks") else None,
                    },
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")
        else:
            # Non-main processes skip checkpointing
            pass

    # Final model save (compatible with inference example)
    if not distributed or (distributed and dist.get_rank() == 0):
        final_path = Path(output_dir) / "model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "tokenizer": tokenizer,
                "config": {
                    "vocab_size": vocab_size,
                    "block_size": block_size,
                    "n_embed": model.blocks[0].attn.n_embed if hasattr(model, "blocks") else None,
                    "n_layer": len(model.blocks) if hasattr(model, "blocks") else None,
                    "n_head": model.blocks[0].attn.n_head if hasattr(model, "blocks") else None,
                },
            },
            final_path,
        )
        print(f"Training complete. Final model saved to {final_path}")
    else:
        # Non-main process does not write final checkpoint
        pass
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train NanoGPT on live JSONL data")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing *.jsonl files")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Where to store checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda or mps (auto detects)" )
    parser.add_argument("--amp", action="store_true", help="Enable mixed‑precision (AMP) training")
    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard log directory")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Save every N epochs")
    parser.add_argument("--distributed", action="store_true", help="Enable torch DistributedDataParallel training")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA low‑rank adapters")
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank (default 4)")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA scaling factor (default 1.0)")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        block_size=args.block_size,
        lr=args.lr,
        device_str=args.device,
        checkpoint_interval=args.checkpoint_interval,
        distributed=args.distributed,
        lora=args.lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )


if __name__ == "__main__":
    main()
