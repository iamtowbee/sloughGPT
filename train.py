#!/usr/bin/env python3
"""
SloughGPT Training Script - Optimized for Mac

Usage:
    python train.py                    # Train with defaults
    python train.py --epochs 10      # More epochs
    python train.py --gpu             # Force GPU usage
"""

import argparse
import time
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Training data - embedded directly so it works without downloads
DEFAULT_DATA = """First Citizen:
We are accounted poor citizens, the patricians good.
If wealthy scion, meet the ancient marke
Some to the wars, to seek their lives have gone,
And from the Register their names razed.
"""

class SimpleTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
    
    def encode(self, text: str) -> np.ndarray:
        return np.array([self.stoi[c] for c in text if c in self.stoi], dtype=np.int64)
    
    def decode(self, ids):
        return ''.join([self.itos.get(i, '') for i in ids])


class TextDataset:
    """Simple text dataset."""
    
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return max(0, len(self.data) - self.block_size)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        return x, y


class SloughGPT(nn.Module):
    """Simple GPT model."""
    
    def __init__(self, vocab_size, n_embed=128, n_layer=4, n_head=4, block_size=128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, n_head)
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        self.vocab_size = vocab_size
        
    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )
        return logits, loss


class TransformerBlock(nn.Module):
    """Transformer block."""
    
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_embed, n_head, batch_first=True)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(0.1),
        )
    
    def forward(self, x):
        # Attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.ln1(x)
        # MLP with residual
        x = x + self.mlp(x)
        x = self.ln2(x)
        return x


def get_device():
    """Get best available device - prefers MPS on Mac."""
    # Try CUDA first
    if torch.cuda.is_available():
        return "cuda", "NVIDIA GPU"
    
    # Try MPS on Mac
    if torch.backends.mps.is_available():
        return "mps", "Apple Silicon (MPS)"
    
    return "cpu", "CPU"


def train(
    data_path: str = "",
    epochs: int = 5,
    batch_size: int = 32,
    n_embed: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    block_size: int = 128,
    lr: float = 1e-3,
    device_str: str = "auto",
):
    """Train the model."""
    
    # Get device
    if device_str == "auto":
        device, device_name = get_device()
    else:
        device = device_str
        device_name = device_str.upper()
    
    print("=" * 60)
    print("  SLOUGHGPT TRAINING")
    print("=" * 60)
    print(f"Device:     {device_name}")
    print(f"Epochs:     {epochs}")
    print(f"Batch:      {batch_size}")
    print(f"Layers:     {n_layer}")
    print(f"Embed:      {n_embed}")
    print("=" * 60)
    
    # Load or create data
    if data_path and Path(data_path).exists():
        text = Path(data_path).read_text()
    else:
        print("Using embedded sample data...")
        text = DEFAULT_DATA
    
    print(f"\nData: {len(text):,} characters")
    
    # Tokenize
    tokenizer = SimpleTokenizer(text)
    data = tokenizer.encode(text)
    print(f"Vocabulary: {tokenizer.vocab_size} chars")
    
    # Dataset
    dataset = TextDataset(data, block_size)
    print(f"Batches per epoch: {len(dataset) // batch_size}")
    
    # Model
    model = SloughGPT(
        vocab_size=tokenizer.vocab_size,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} parameters")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Training loop
    print("\n" + "=" * 60)
    print("  TRAINING")
    print("=" * 60)
    
    start_time = time.time()
    batches_per_epoch = max(1, len(dataset) // batch_size)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        batches = 0
        
        # Simple random sampling
        for _ in range(batches_per_epoch):
            # Random indices
            idx = torch.randint(0, max(1, len(dataset) - block_size), (batch_size,))
            
            x = torch.stack([dataset[i.item()][0] for i in idx]).to(device)
            y = torch.stack([dataset[i.item()][1] for i in idx]).to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, y)
            
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batches += 1
                
                # Progress
                if batches % 20 == 0:
                    elapsed = time.time() - start_time
                    pct = (batches / batches_per_epoch) * 100
                    eta = (elapsed / batches) * (batches_per_epoch - batches)
                    print(f"Epoch {epoch+1}/{epochs} [{'█'*int(pct/5)}{'░'*(20-int(pct/5))}] {pct:.0f}% | Loss: {loss.item():.4f} | ETA: {eta:.0f}s")
        
        avg_loss = total_loss / max(batches, 1)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} done | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    
    # Save
    output_dir = PROJECT_ROOT / "models" / "sloughgpt"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'config': {
            'vocab_size': tokenizer.vocab_size,
            'n_embed': n_embed,
            'n_layer': n_layer,
            'n_head': n_head,
            'block_size': block_size,
        }
    }, output_dir / "sloughgpt.pt")
    
    print("\n" + "=" * 60)
    print(f"✅ DONE! Time: {total_time:.1f}s | Saved: {output_dir / 'sloughgpt.pt'}")
    print("=" * 60)
    
    return model, tokenizer


def generate(model, tokenizer, prompt="", max_tokens=100, device="cpu"):
    """Generate text."""
    model.eval()
    
    if prompt:
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    else:
        idx = torch.randint(0, tokenizer.vocab_size, (1, 1), dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = idx[:, -128:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / 0.8
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat([idx, idx_next], dim=1)
    
    return tokenizer.decode(idx[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Train SloughGPT")
    parser.add_argument("--data", type=str, default="", help="Data file path")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--layers", type=int, default=4, help="Layers")
    parser.add_argument("--embed", type=int, default=128, help="Embedding size")
    parser.add_argument("--head", type=int, default=4, help="Attention heads")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gpu", action="store_true", help="Force GPU usage")
    parser.add_argument("--generate", action="store_true", help="Generate after training")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for generation")
    parser.add_argument("--tokens", type=int, default=200, help="Tokens to generate")
    
    args = parser.parse_args()
    
    device = "cuda" if args.gpu else "auto"
    
    model, tokenizer = train(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        n_embed=args.embed,
        n_layer=args.layers,
        n_head=args.head,
        lr=args.lr,
        device_str=device,
    )
    
    if args.generate:
        print("\n" + "=" * 60)
        print("  GENERATING TEXT")
        print("=" * 60)
        result = generate(model, tokenizer, args.prompt, args.tokens, "cpu")
        print(result)


if __name__ == "__main__":
    main()
