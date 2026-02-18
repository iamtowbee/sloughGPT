"""
NanoGPT Model Implementation
Based on recovered simple_gpt_model.py
"""

import torch
import torch.nn
import torch.nn.functional
import math
from typing import Optional, Tuple

nn = torch.nn
F = torch.nn.functional


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal mask."""
    
    def __init__(self, n_embed: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert n_embed % n_head == 0
        
        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.c_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        self.n_embed = n_embed
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        if x.is_cuda:
            causal_mask = causal_mask.cuda()
        att = att.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Simple feed-forward network."""
    
    def __init__(self, n_embed: int, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(n_embed, 4 * n_embed, bias=False)
        self.c_proj = nn.Linear(4 * n_embed, n_embed, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with attention and feed-forward."""
    
    def __init__(self, n_embed: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed, dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NanoGPT(nn.Module):
    """NanoGPT - A minimal GPT model."""
    
    def __init__(
        self,
        vocab_size: int,
        n_embed: int = 384,
        n_layer: int = 6,
        n_head: int = 6,
        dropout: float = 0.1,
        block_size: int = 256,
    ):
        super().__init__()
        
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        
        self.blocks = nn.ModuleList([
            Block(n_embed, n_head, dropout) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        
        self.apply(self._init_weights)
        self.lm_head.weight.data = self.token_embedding_table.weight.data.clone()
        
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        
        assert t <= self.block_size, f"Sequence length {t} exceeds block size {self.block_size}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            if targets.size(1) != logits.size(1):
                if targets.size(1) < logits.size(1):
                    logits = logits[:, :targets.size(1), :]
                else:
                    targets = targets[:, :logits.size(1)]
            
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
            return logits, loss
        else:
            return logits, None
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text tokens."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["NanoGPT", "Block", "CausalSelfAttention", "MLP"]
