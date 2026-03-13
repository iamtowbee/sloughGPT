#!/usr/bin/env python3
"""
SloughGPT Training Pipeline
Uses existing infrastructure: NanoGPT model, training features
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import math

# Use existing NanoGPT from our infrastructure
from domains.training.models.nanogpt import NanoGPT


class TextDataset(Dataset):
    """Character-level text dataset."""
    
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        return x, y


def prepare_data(data_path, block_size=128):
    """Prepare training data from text file."""
    
    # Read text
    with open(data_path, 'r', encoding='utf-8') as f:
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


def train_sloughgpt(
    data_path='datasets/shakespeare/input.txt',
    vocab_size=None,
    n_embed=256,
    n_layer=6,
    n_head=8,
    block_size=128,
    batch_size=64,
    epochs=10,
    lr=1e-3,
    device='cpu',
    resume_from=None,
    max_steps=None,
):
    """Train SloughGPT model."""
    
    print("=" * 50)
    print("SLOUGHGPT TRAINING PIPELINE")
    print("=" * 50)
    
    # Load existing model or create new one
    stoi = None
    itos = None
    
    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from {resume_from}...")
        checkpoint = torch.load(resume_from, map_location=device)
        
        # Handle different checkpoint formats
        if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Get config from training_info or top-level keys
        training_info = checkpoint.get('training_info', {})
        
        if 'chars' in checkpoint:
            checkpoint_vocab_size = len(checkpoint['chars'])
        else:
            checkpoint_vocab_size = training_info.get('vocab_size', checkpoint.get('vocab_size', vocab_size))
        
        n_embed = training_info.get('n_embed', checkpoint.get('n_embed', n_embed))
        n_layer = training_info.get('n_layer', checkpoint.get('n_layer', n_layer))
        n_head = training_info.get('n_head', checkpoint.get('n_head', n_head))
        block_size = training_info.get('block_size', checkpoint.get('block_size', block_size))
        
        print(f"Resuming with: vocab={checkpoint_vocab_size}, embed={n_embed}, layers={n_layer}, heads={n_head}, block={block_size}")
        
        model = NanoGPT(
            vocab_size=checkpoint_vocab_size,
            n_embed=n_embed,
            n_layer=n_layer,
            n_head=n_head,
            block_size=block_size
        )
        model.load_state_dict(state_dict)
        stoi = checkpoint.get('stoi')
        itos = checkpoint.get('itos')
        print(f"Loaded model with {model.num_parameters:,} parameters")
    else:
        # Create model using existing infrastructure
        model = NanoGPT(
            vocab_size=vocab_size,
            n_embed=n_embed,
            n_layer=n_layer,
            n_head=n_head,
            block_size=block_size
        )
    
    model = model.to(device)
    
    # Prepare data (use block_size from model)
    data, vocab_size, stoi, itos = prepare_data(data_path, model.block_size)
    
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
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
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
    
    best_val_loss = float('inf')
    
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
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (x, y) in enumerate(pbar):
            if max_steps and batch_idx >= max_steps:
                break
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
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
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model': model.state_dict(),
                'chars': list(range(vocab_size)),
                'stoi': stoi,
                'itos': itos,
                'training_info': {
                    'vocab_size': vocab_size,
                    'n_embed': n_embed,
                    'n_layer': n_layer,
                    'n_head': n_head,
                    'block_size': block_size,
                }
            }, 'models/sloughgpt_best.pt')
            print(f"  -> New best model saved!")
        
        print("-" * 50)
    
    # Save model
    save_path = 'models/sloughgpt.pt'
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'n_embed': n_embed,
        'n_layer': n_layer,
        'n_head': n_head,
        'block_size': block_size,
        'stoi': stoi,
        'itos': itos,
    }, save_path)
    
    print(f"\nModel saved to {save_path}")
    
    return model, stoi, itos


def generate_text(model, stoi, itos, prompt='First', max_new_tokens=200, temperature=0.8):
    """Generate text from trained model."""
    
    model.eval()
    
    # Encode prompt
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long)
    
    # Generate
    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    
    # Decode
    text = ''.join([itos.get(i.item(), '?') for i in output[0]])
    
    return text


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SloughGPT')
    parser.add_argument('--data', type=str, default='datasets/shakespeare/input.txt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_embed', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
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
    )
    
    # Generate sample
    print("\n" + "=" * 50)
    print("SAMPLE GENERATION")
    print("=" * 50)
    
    text = generate_text(model, stoi, itos, prompt="First")
    print(text)