#!/usr/bin/env python3
"""
Simple GPT Model for Distributed Training
A minimal transformer implementation compatible with distributed training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal mask."""
    
    def __init__(self, n_embed, n_head, dropout=0.1):
        super().__init__()
        assert n_embed % n_head == 0
        
        # Key, query, value projections
        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=False)
        
        # Output projection
        self.c_proj = nn.Linear(n_embed, n_embed, bias=False)
        
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        self.n_embed = n_embed
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        
    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        
        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention computation
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        if x.is_cuda:
            causal_mask = causal_mask.cuda()
        att = att.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Simple feed-forward network."""
    
    def __init__(self, n_embed, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(n_embed, 4 * n_embed, bias=False)
        self.c_proj = nn.Linear(4 * n_embed, n_embed, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with attention and feed-forward."""
    
    def __init__(self, n_embed, n_head, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed, dropout)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Simple GPT model for distributed training."""
    
    def __init__(self, vocab_size, n_embed=384, n_layer=6, n_head=6, dropout=0.1, block_size=256):
        super().__init__()
        
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        # Token embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        
        # Position embeddings
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(n_embed, n_head, dropout) for _ in range(n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(n_embed)
        
        # Language model head
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special initialization for output projection
        self.lm_head.weight.data = self.token_embedding_table.weight.data.clone()
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        assert t <= self.block_size, f"Sequence length {t} exceeds block size {self.block_size}"
        
        # Position indices
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        
        # Token and position embeddings
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb
        
        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language model logits
        logits = self.lm_head(x)
        
        # Calculate loss if targets provided
        if targets is not None:
            # Ensure targets have same sequence length as logits
            if targets.size(1) != logits.size(1):
                # If targets are shorter, trim logits
                if targets.size(1) < logits.size(1):
                    logits = logits[:, :targets.size(1), :]
                # If targets are longer, trim targets  
                else:
                    targets = targets[:, :logits.size(1)]
            
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
            return logits, loss
        else:
            return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text tokens."""
        for _ in range(max_new_tokens):
            # Crop context to block size
            idx_cond = idx[:, -self.block_size:]
            
            # Get predictions
            logits = self(idx_cond)
            
            # Focus on last time step
            logits = logits[:, -1, :]
            
            # Apply temperature
            logits = logits / temperature
            
            # Optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


def load_dataset(dataset_name):
    """Load dataset in the expected format."""
    import pickle
    from pathlib import Path
    
    dataset_path = Path(f"datasets/{dataset_name}")
    
    # Check if dataset exists
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_name}")
    
    # Load metadata
    meta_file = dataset_path / "meta.pkl"
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata not found for dataset: {dataset_name}")
    
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
    
    # Load training data
    train_file = dataset_path / "train.bin"
    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found for dataset: {dataset_name}")
    
    train_data = np.fromfile(train_file, dtype=np.uint16)
    
    # Load validation data
    val_file = dataset_path / "val.bin"
    if val_file.exists():
        val_data = np.fromfile(val_file, dtype=np.uint16)
    else:
        # Use last 10% of training data for validation
        split_idx = int(0.9 * len(train_data))
        val_data = train_data[split_idx:]
        train_data = train_data[:split_idx]
    
    return train_data, val_data, meta


def count_parameters(model):
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Test the model
    model = GPT(vocab_size=100, n_embed=128, n_layer=4, n_head=4)
    
    print(f"Model created with {count_parameters(model)} parameters")
    
    # Test forward pass
    x = torch.randint(0, 100, (2, 16))
    targets = x[:, 1:]  # targets is one shorter than input
    logits, loss = model(x, targets)
    
    print(f"Forward pass successful. Logits shape: {logits.shape}, Loss: {loss.item():.4f}")
    
    # Test generation
    generated = model.generate(torch.randint(0, 100, (1, 1)), max_new_tokens=10)
    print(f"Generation successful. Shape: {generated.shape}")