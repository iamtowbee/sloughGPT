#!/usr/bin/env python3
"""
Example: Train a custom NanoGPT model
"""

import sys
sys.path.insert(0, "..")

from domains.training.models.nanogpt import NanoGPT
from domains.training.optimized_trainer import TrainingConfig, OptimizedTrainer, Presets
import torch

def main():
    print("=" * 60)
    print("Training Example: Custom NanoGPT")
    print("=" * 60)
    
    # Get optimal config for this device
    config = Presets.auto()
    print(f"\nUsing preset: {config.device}")
    
    # Create small model for demo
    vocab_size = 1000
    model = NanoGPT(
        vocab_size=vocab_size,
        n_embed=128,
        n_layer=4,
        n_head=4,
        block_size=64
    )
    
    print(f"\nModel parameters: {model.num_parameters:,}")
    
    # Dummy training data
    train_data = torch.randint(0, vocab_size, (1000, 64))
    
    # Simple training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    
    print("\nTraining for 10 steps...")
    for i in range(10):
        idx = torch.randint(0, 1000 - 64, (4,))
        x = train_data[idx]
        y = train_data[idx]
        
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i % 5 == 0:
            print(f"  Step {i}: loss = {loss.item():.4f}")
    
    print("\nTraining complete!")
    print(f"Final loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
