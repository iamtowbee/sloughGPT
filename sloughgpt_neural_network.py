#!/usr/bin/env python3
"""
SloughGPT Neural Network Architecture
Our Custom Transformer-Based Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple
import json
import time
from dataclasses import dataclass
from collections import deque

from .core.config import ModelConfig

@dataclass
class ModelConfig:
    """Configuration for our SloughGPT model"""
    vocab_size: int = 50000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 1024
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    quantized: bool = False
    compile: bool = False
    attention_type: str = "scaled_dot_product_attention"  # "flash_attention", "efficient_attention"

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer with optimization support"""
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, seq_len, d_model = query.size()
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention calculation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        
        output = self.w_o(context)
        return output, attention_weights

class TransformerBlock(nn.Module):
    """Single transformer block with optimization support"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(
            config.d_model, config.n_heads, config.dropout,
            attention_type=config.attention_type
        )
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class SloughGPT(nn.Module):
    """Our custom GPT transformer model with optimization support"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_encoding = PositionalEncoding(config.d_model, config.max_seq_length)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config, attention_type=config.attention_type) for _ in range(config.n_layers)
        ])
        
        # Output layers
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        
        # Create attention mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, device=input_ids.device)
        
        # Embeddings and positional encoding
        x = self.token_embedding(input_ids)
        x = self.position_encoding(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, 
                temperature: float = 1.0, do_sample: bool = True,
                top_k: int = 50) -> torch.Tensor:
        """Generate text from the model"""
        
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                logits = self(input_ids)
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_logits, top_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, -float('inf'))
                    logits.scatter_(1, top_indices, top_logits)
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids

# Test our SloughGPT architecture
def test_sloughgpt_architecture():
    """Test the neural network architecture"""
    print("🧠 SloughGPT Neural Network Architecture")
    print("=" * 50)
    
    # Configuration
    config = ModelConfig(
        vocab_size=1000,  # Small for testing
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        dropout=0.1,
        max_seq_length=512
    )
    
    # Create model
    model = SloughGPT(config)
    device = torch.device(config.device)
    model.to(device)
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"✅ Device: {device}")
    print(f"✅ Config: vocab={config.vocab_size}, d_model={config.d_model}, layers={config.n_layers}")
    
    # Test forward pass
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    
    print(f"\n🔄 Testing forward pass...")
    start_time = time.time()
    
    with torch.no_grad():
        logits = model(input_ids)
    
    forward_time = time.time() - start_time
    print(f"✅ Forward pass: {forward_time:.4f}s")
    print(f"✅ Output shape: {logits.shape}")
    print(f"✅ Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")
    
    # Test generation
    print(f"\n🎲 Testing generation...")
    start_time = time.time()
    
    with torch.no_grad():
        generated = model.generate(input_ids[:1], max_length=20, temperature=0.8)
    
    gen_time = time.time() - start_time
    print(f"✅ Generation: {gen_time:.4f}s")
    print(f"✅ Generated sequence shape: {generated.shape}")
    print(f"✅ Generated tokens: {generated[0].tolist()}")
    
    # Model summary
    print(f"\n📊 Model Summary:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    return model

if __name__ == "__main__":
    model = test_sloughgpt_architecture()