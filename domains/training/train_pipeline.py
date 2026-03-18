#!/usr/bin/env python3
"""
SloughGPT Training Pipeline
Uses EXISTING infrastructure: lora.py, efficient_inference.py, lr_schedulers.py
"""

import torch
from torch.utils.data import Dataset
from utils import get_device

# === USE EXISTING INFRASTRUCTURE ===
from domains.training.models.nanogpt import NanoGPT
from domains.training.lora import apply_lora_to_model, LoRAConfig
from domains.training.efficient_inference import Quantizer
from domains.training.lr_schedulers import create_scheduler


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
    """Prepare training data."""
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    print(f"Data: {len(data)} tokens, {len(chars)} chars")
    return data, len(chars), stoi, itos


class SloughGPTTrainer:
    """
    Training pipeline using EXISTING infrastructure.
    """

    def __init__(
        self,
        data_path,
        # Model config
        vocab_size=None,
        n_embed=256,
        n_layer=6,
        n_head=8,
        block_size=128,
        # LoRA config
        use_lora=False,
        lora_rank=8,
        lora_alpha=16,
        # Training config
        batch_size=64,
        epochs=10,
        lr=1e-3,
        max_steps=None,
        # Quantization
        quantize=False,
        quantize_bits=8,
        device=None,
    ):
        self.data_path = data_path
        self.block_size = block_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.max_steps = max_steps
        self.device = device or get_device()

        print(f"Using device: {self.device}")

        # Prepare data
        self.data, self.vocab_size, self.stoi, self.itos = prepare_data(data_path, block_size)

        # Create model using existing NanoGPT
        print("\n=== Creating Model ===")
        self.model = NanoGPT(
            vocab_size=self.vocab_size,
            n_embed=n_embed,
            n_layer=n_layer,
            n_head=n_head,
            block_size=block_size,
        ).to(self.device)

        print(f"Base model: {self.model.num_parameters:,} params")

        # Apply LoRA using existing lora.py
        if use_lora:
            print("\n=== Applying LoRA ===")
            lora_config = LoRAConfig(
                rank=lora_rank,
                alpha=lora_alpha,
                target_modules=["c_attn", "c_proj", "c_fc"],  # NanoGPT modules
            )
            self.model = apply_lora_to_model(self.model, config=lora_config)

            # Count LoRA params
            lora_params = sum(p.numel() for n, p in self.model.named_parameters() if "lora_" in n)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"LoRA params: {lora_params:,} ({100 * lora_params / total:.1f}%)")

        # Apply quantization using existing efficient_inference.py
        if quantize:
            print("\n=== Applying Quantization ===")
            dtype = torch.qint8 if quantize_bits == 8 else torch.float16
            self.model = Quantizer.dynamic_quantize(self.model, dtype=dtype)
            print(f"Quantized to INT{quantize_bits}")

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        # Split data FIRST
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

        # Use existing LR scheduler (needs total_steps)
        total_steps = (len(self.train_data) // self.block_size // self.batch_size) * self.epochs
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type="cosine",
            total_steps=total_steps,
            warmup_steps=total_steps // 10,
        )

        print(f"\nTrain: {len(self.train_data)}, Val: {len(self.val_data)}")

    def get_batch(self, split="train"):
        """Get a batch of data."""
        data = self.train_data if split == "train" else self.val_data
        idx = torch.randint(0, len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in idx])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in idx])
        return x.to(self.device), y.to(self.device)

    def train_step(self):
        """Single training step."""
        x, y = self.get_batch("train")

        self.optimizer.zero_grad()
        logits, loss = self.model(x, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def train(self):
        """Full training loop."""
        print("\n=== Training ===")

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0

            steps_per_epoch = len(self.train_data) // self.block_size // self.batch_size

            for step in range(steps_per_epoch):
                if self.max_steps and step >= self.max_steps:
                    break

                loss = self.train_step()
                train_loss += loss

                if step % 50 == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.epochs} | Step {step}/{steps_per_epoch} | Loss: {loss:.4f}"
                    )

            if self.max_steps and sum(1 for _ in range(epoch * steps_per_epoch)) >= self.max_steps:
                break

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for _ in range(10):
                    x, y = self.get_batch("val")
                    _, loss = self.model(x, y)
                    val_loss += loss.item()

            print(
                f"Epoch {epoch + 1} | Train: {train_loss / steps_per_epoch:.4f} | Val: {val_loss / 10:.4f}"
            )

        return self.model

    def generate(self, prompt, max_tokens=200, temperature=0.8):
        """Generate text."""
        self.model.eval()
        idx = torch.tensor([[self.stoi.get(c, 0) for c in prompt[:1]]])

        with torch.no_grad():
            output = self.model.generate(idx, max_new_tokens=max_tokens, temperature=temperature)

        text = "".join([self.itos.get(int(i), "?") for i in output[0]])
        return text

    def save(self, path):
        """Save model checkpoint."""
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            "itos": self.itos,
            "config": {
                "n_embed": self.model.n_embed,
                "n_layer": self.model.n_layer,
                "n_head": self.model.n_head,
                "block_size": self.model.block_size,
            },
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        """Load model checkpoint and recreate trainer."""
        checkpoint = torch.load(path, weights_only=False)

        # Recreate trainer with same config
        # For now just return the model
        return checkpoint


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/shakespeare/input.txt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_embed", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Max training steps (for quick testing)"
    )
    parser.add_argument("--lora", action="store_true", help="Use LoRA")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--quantize", action="store_true", help="Quantize model")
    args = parser.parse_args()

    # Create trainer
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
        use_lora=args.lora,
        lora_rank=args.lora_rank,
        quantize=args.quantize,
    )

    # Train
    trainer.train()

    # Generate
    print("\n=== Generated Text ===")
    print(trainer.generate("First"))


if __name__ == "__main__":
    main()
