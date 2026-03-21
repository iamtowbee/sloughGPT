#!/usr/bin/env python3
"""
SloughGPT Training Pipeline
Uses existing infrastructure: NanoGPT model, training features
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math

# GPU optimizations
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
elif torch.backends.mps.is_available():
    import torch.mps

# Use existing NanoGPT from our infrastructure
from domains.training.models.nanogpt import NanoGPT
from domains.training.export import export_to_safetensors, export_to_gguf


class TextDataset(Dataset):
    """Character-level text dataset."""

    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx : idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1 : idx + self.block_size + 1], dtype=torch.long)
        return x, y


def prepare_data(data_path, block_size=128):
    """Prepare training data from text file."""

    # Read text
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Character-level encoding
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encode entire text
    data = [stoi[ch] for ch in text]

    print(f"Dataset: {len(data)} tokens, {vocab_size} unique characters")

    return data, vocab_size, stoi, itos


def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_gpu_available():
    """Check if any GPU is available (CUDA or MPS)."""
    return torch.cuda.is_available() or torch.backends.mps.is_available()


def _save_model(model, base_path: str, format: str = "safetensors", quantization=None, metadata=None):
    """Save model in standard format with metadata."""
    os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)

    meta = metadata or {}
    meta["model_type"] = "nanogpt"
    meta["format"] = format

    formats = format.split(",") if "," in format else [format]

    for fmt in formats:
        fmt = fmt.strip().lower()

        if fmt == "safetensors":
            path = base_path + ".safetensors"
            export_to_safetensors(model, path, meta)

        elif fmt == "safetensors_bf16":
            path = base_path + "-bf16.safetensors"
            export_to_safetensors(model, path, meta, dtype="bf16")

        elif fmt in ("gguf", "gguf_q4_0", "gguf_q4_1", "gguf_q5_0", "gguf_q5_1", "gguf_q8_0", "gguf_f16", "gguf_f32"):
            quant_map = {
                "gguf": quantization or "Q4_0",
                "gguf_q4_0": "Q4_0",
                "gguf_q4_1": "Q4_1",
                "gguf_q5_0": "Q5_0",
                "gguf_q5_1": "Q5_1",
                "gguf_q8_0": "Q8_0",
                "gguf_f16": "F16",
                "gguf_f32": "F32",
            }
            quant = quant_map.get(fmt, "Q4_0")
            path = base_path + f"-{quant}.gguf"
            export_to_gguf(model, path, quant)

        elif fmt in ("torch", "pytorch"):
            path = base_path + ".pt"
            torch.save({"model_state_dict": model.state_dict(), "metadata": meta}, path)

        else:
            path = base_path + ".safetensors"
            export_to_safetensors(model, path, meta)


def train_sloughgpt(
    data_path="datasets/shakespeare/input.txt",
    vocab_size=None,
    n_embed=256,
    n_layer=6,
    n_head=8,
    block_size=128,
    batch_size=64,
    epochs=10,
    lr=1e-3,
    device=None,
    resume_from=None,
    max_steps=None,
    use_lora=False,
    lora_rank=8,
    lora_alpha=16,
    gradient_accumulation=1,
    mixed_precision=False,
    compile_model=False,
    save_format="safetensors",
    save_quantized=None,
    save_path=None,
):
    """Train SloughGPT model."""

    use_amp = mixed_precision and is_gpu_available()

    if device is None:
        device = get_device()

    print(f"Using device: {device}")
    print("=" * 50)

    # Prepare data FIRST to get vocab_size
    data, vocab_size, stoi, itos = prepare_data(data_path, block_size)

    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from {resume_from}...")
        checkpoint = torch.load(resume_from, map_location=device)

        # Handle different checkpoint formats
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Get config from training_info or top-level keys
        training_info = checkpoint.get("training_info", {})

        if "chars" in checkpoint:
            checkpoint_vocab_size = len(checkpoint["chars"])
        else:
            checkpoint_vocab_size = training_info.get(
                "vocab_size", checkpoint.get("vocab_size", vocab_size)
            )

        n_embed = training_info.get("n_embed", checkpoint.get("n_embed", n_embed))
        n_layer = training_info.get("n_layer", checkpoint.get("n_layer", n_layer))
        n_head = training_info.get("n_head", checkpoint.get("n_head", n_head))
        block_size = training_info.get("block_size", checkpoint.get("block_size", block_size))

        print(
            f"Resuming with: vocab={checkpoint_vocab_size}, embed={n_embed}, "
            f"layers={n_layer}, heads={n_head}, block={block_size}"
        )

        model = NanoGPT(
            vocab_size=checkpoint_vocab_size,
            n_embed=n_embed,
            n_layer=n_layer,
            n_head=n_head,
            block_size=block_size,
        )
        model.load_state_dict(state_dict)
        stoi = checkpoint.get("stoi")
        itos = checkpoint.get("itos")
        print(f"Loaded model with {model.num_parameters:,} parameters")
    else:
        # Create model using existing infrastructure
        model = NanoGPT(
            vocab_size=vocab_size,
            n_embed=n_embed,
            n_layer=n_layer,
            n_head=n_head,
            block_size=block_size,
        )

    model = model.to(device)

    # Apply LoRA if requested
    if use_lora:
        from domains.training.lora import apply_lora_to_model, LoRAConfig, print_lora_summary

        lora_config = LoRAConfig(rank=lora_rank, alpha=lora_alpha)
        model = apply_lora_to_model(model, lora_config)
        print_lora_summary(model)

    print(f"\nModel parameters: {model.num_parameters:,}")

    # Split train/val (use block_size from model for correct data splitting)
    model_block_size = model.block_size
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Create datasets
    train_dataset = TextDataset(train_data, model_block_size)
    val_dataset = TextDataset(val_data, model_block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Optimizer - either LoRA params only or all params
    if use_lora:
        from domains.training.lora import get_lora_parameters

        lora_params = get_lora_parameters(model)
        if not lora_params:
            print("Warning: No LoRA parameters found. Training all parameters instead.")
            train_params = model.parameters()
        else:
            print(f"Training {len(lora_params)} LoRA parameter groups")
            train_params = lora_params.values()
    else:
        train_params = model.parameters()

    optimizer = torch.optim.AdamW(train_params, lr=lr, weight_decay=0.01)

    # Mixed precision scaler (only for CUDA, MPS doesn't support it well)
    use_amp = mixed_precision and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Compile model if requested (only on CUDA, MPS support is experimental)
    if compile_model and hasattr(torch, "compile") and device == "cuda":
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Print optimization info
    print("Optimizations:")
    eff_batch = batch_size * gradient_accumulation
    print(f"  - Gradient accumulation: {gradient_accumulation}x (effective batch: {eff_batch})")
    print(f"  - Mixed precision: {'ON' if use_amp else 'OFF'}")
    print(f"  - torch.compile: {'ON' if compile_model and hasattr(torch, 'compile') else 'OFF'}")
    print(f"  - LoRA: {'ON' if use_lora else 'OFF'}")

    # Cosine learning rate scheduler with warmup
    total_steps = len(train_loader) * epochs
    warmup_steps = min(total_steps // 10, 500)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")

    # Training loop
    print(f"\nTraining on {device}...")
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
    print("-" * 50)

    # Training loop
    print(f"\nTraining on {device}...")
    print("-" * 50)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, (x, y) in enumerate(pbar):
            if max_steps and batch_idx >= max_steps:
                break
            x, y = x.to(device), y.to(device)

            # Forward pass with mixed precision (CUDA only)
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    logits, loss = model(x, y)
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation
                scaler.scale(loss).backward()

                # Optimizer step every gradient_accumulation steps
                if (batch_idx + 1) % gradient_accumulation == 0 or (batch_idx + 1) == len(
                    train_loader
                ):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                logits, loss = model(x, y)
                loss = loss / gradient_accumulation
                loss.backward()

                if (batch_idx + 1) % gradient_accumulation == 0 or (batch_idx + 1) == len(
                    train_loader
                ):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            scheduler.step()

            train_loss += loss.item() * gradient_accumulation
            pbar.set_postfix(
                {
                    "loss": f"{loss.item() * gradient_accumulation:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        if max_steps:
            break

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_model(
                model,
                "models/sloughgpt_best",
                format=save_format,
                quantization=save_quantized,
                metadata={
                    "vocab_size": vocab_size,
                    "n_embed": n_embed,
                    "n_layer": n_layer,
                    "n_head": n_head,
                    "block_size": block_size,
                    "stoi": stoi,
                    "itos": itos,
                    "is_best": True,
                },
            )
            print(f"  -> New best model saved ({save_format})!")

        print("-" * 50)

    # Save model
    os.makedirs("models", exist_ok=True)
    final_path = save_path or "models/sloughgpt"

    _save_model(
        model,
        final_path,
        format=save_format,
        quantization=save_quantized,
        metadata={
            "vocab_size": vocab_size,
            "n_embed": n_embed,
            "n_layer": n_layer,
            "n_head": n_head,
            "block_size": block_size,
            "stoi": stoi,
            "itos": itos,
        },
    )

    print(f"\nModel saved ({save_format}) to {final_path}")

    return model, stoi, itos


def generate_text(model, stoi, itos, prompt="First", max_new_tokens=200, temperature=0.8):
    """Generate text from trained model."""

    model.eval()

    # Encode prompt
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long)

    # Generate
    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)

    # Decode
    text = "".join([itos.get(i.item(), "?") for i in output[0]])

    return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SloughGPT")
    parser.add_argument("--data", type=str, default="datasets/shakespeare/input.txt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per iteration")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_embed", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None, help="auto-detect: cuda/mps/cpu")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps per epoch")

    # Optimization options
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for efficient training")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch_size * gradient_accumulation)",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Use mixed precision training (faster on GPU)",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Compile model with torch.compile (PyTorch 2.0+)"
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="safetensors",
        choices=["safetensors", "safetensors_bf16", "torch", "gguf", "all"],
        help="Model save format (default: safetensors)",
    )
    parser.add_argument(
        "--save_quantized",
        type=str,
        default=None,
        choices=["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "F16", "F32"],
        help="Quantization type for GGUF export",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="models/sloughgpt",
        help="Output path (without extension, format added automatically)",
    )

    args = parser.parse_args()

    # Train
    model, stoi, itos = train_sloughgpt(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_embed=args.n_embed,
        n_layer=args.n_layer,
        n_head=args.n_head,
        block_size=args.block_size,
        device=args.device,
        resume_from=args.resume,
        max_steps=args.max_steps,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        gradient_accumulation=args.gradient_accumulation,
        mixed_precision=args.mixed_precision,
        compile_model=args.compile,
        save_format=args.save_format,
        save_quantized=args.save_quantized,
        save_path=args.save_path,
    )

    # Generate sample
    print("\n" + "=" * 50)
    print("SAMPLE GENERATION")
    print("=" * 50)

    text = generate_text(model, stoi, itos, prompt="First")
    print(text)
