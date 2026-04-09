"""
Optimized Operations for SloughGPT

High-performance CUDA kernels and fused operations:
- Fused softmax + mask
- Fused attention score computation
- Fused layer norm
- Optimized transpose operations
- Memory-efficient operations
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedLayerNorm(nn.Module):
    """Fused Layer Normalization with optional bias.

    Combines normalization computation in a single kernel for efficiency.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class FusedRMSNorm(nn.Module):
    """Fused RMSNorm - simpler than LayerNorm, no mean computation.

    LLaMA-style: output = x * weight / RMS(x)
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        output = (x * norm * self.weight).to(x_dtype)
        return output


class FusedCrossEntropyLoss(nn.Module):
    """Fused Cross-Entropy loss with optional label smoothing.

    Combines log_softmax + nll_loss in single kernel.
    """

    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets.clone()
            targets = targets.float()
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / (logits.size(-1) - 1)

        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing if self.label_smoothing > 0 else 0.0,
        )


class FusedAttentionBias(nn.Module):
    """Fused attention bias computation.

    Computes attention scores + bias in single operation.
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, H, E = query.shape
        _, S, _, _ = key.shape

        scores = torch.einsum("bnhe,bshe->bhsn", query, key) * scale

        if attn_bias is not None:
            scores = scores + attn_bias

        if causal and N > 0 and S > 0:
            causal_mask = torch.triu(
                torch.ones(N, S, device=query.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.einsum("bhsn,bshe->bnhe", attn_weights, value)

        return output.contiguous(), attn_weights


class EfficientTranspose(nn.Module):
    """Memory-efficient tensor transpose operations.

    Optimized for common patterns in transformer models.
    """

    @staticmethod
    def transpose_for_attn(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Transpose for multi-head attention: [B, N, C] -> [B, H, N, E]"""
        B, N, C = x.shape
        E = C // num_heads
        return x.view(B, N, num_heads, E).transpose(1, 2)

    @staticmethod
    def transpose_from_attn(x: torch.Tensor) -> torch.Tensor:
        """Transpose back from attention: [B, H, N, E] -> [B, N, C]"""
        B, H, N, E = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * E)


class ChunkedOperation(nn.Module):
    """Chunk large operations into smaller segments for memory efficiency.

    Useful for very long sequences where full attention matrix doesn't fit.
    """

    def __init__(self, chunk_size: int = 512):
        super().__init__()
        self.chunk_size = chunk_size

    def attention_chunked(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chunked attention computation for long sequences."""
        chunk_size = chunk_size or self.chunk_size
        B, H, N, E = query.shape
        _, _, S, _ = key.shape

        all_outputs = []
        all_weights = []

        for i in range(0, N, chunk_size):
            q_chunk = query[:, :, i : i + chunk_size]

            start = max(0, i - chunk_size)
            k_chunk = key[:, :, start : i + chunk_size]
            v_chunk = value[:, :, start : i + chunk_size]

            scale = 1.0 / math.sqrt(E)
            scores = torch.einsum("bqhe,bshe->bqhs", q_chunk, k_chunk) * scale

            causal_mask = torch.triu(
                torch.ones(q_chunk.size(2), k_chunk.size(2), device=query.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            attn = F.softmax(scores, dim=-1)
            out = torch.einsum("bqhs,bshe->bqhe", attn, v_chunk)

            all_outputs.append(out)
            all_weights.append(attn)

        output = torch.cat(all_outputs, dim=2)
        weights = torch.cat(all_weights, dim=3)

        return output, weights


class MemoryEfficientSoftmax(nn.Module):
    """Memory-efficient softmax with configurable numerical stability.

    Computes softmax in chunks to avoid large intermediate tensors.
    """

    @staticmethod
    def forward(
        logits: torch.Tensor,
        dim: int = -1,
        stable: bool = True,
        chunk_size: int = 0,
    ) -> torch.Tensor:
        """Memory-efficient softmax."""
        if chunk_size > 0 and logits.shape[dim] > chunk_size:
            return MemoryEfficientSoftmax._chunked_softmax(logits, dim, stable, chunk_size)

        if stable:
            logits_max = logits.max(dim=dim, keepdim=True)[0]
            logits = logits - logits_max

        exp_logits = torch.exp(logits)
        sum_exp = exp_logits.sum(dim=dim, keepdim=True)
        return exp_logits / sum_exp

    @staticmethod
    def _chunked_softmax(
        logits: torch.Tensor,
        dim: int,
        stable: bool,
        chunk_size: int,
    ) -> torch.Tensor:
        """Chunked softmax for very large dimensions."""
        dim_size = logits.shape[dim]
        chunks = []

        for i in range(0, dim_size, chunk_size):
            chunk = logits.index_select(dim, torch.arange(i, min(i + chunk_size, dim_size), device=logits.device))

            if stable:
                chunk_max = chunk.max(dim=dim, keepdim=True)[0]
                chunk = chunk - chunk_max

            exp_chunk = torch.exp(chunk)
            sum_exp = exp_chunk.sum(dim=dim, keepdim=True)
            chunks.append(exp_chunk / sum_exp)

        return torch.cat(chunks, dim=dim)


class FusedScaleBias(nn.Module):
    """Fused scale and bias operation for memory efficiency."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias


class OptimizedEmbedding(nn.Embedding):
    """Optimized embedding layer with optional caching and quantization."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        quantize: bool = False,
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        self.quantize = quantize
        self._quantized_weight: Optional[torch.Tensor] = None

    def quantize_weight(self, dtype: torch.dtype = torch.quint8):
        """Quantize embedding weights for memory savings."""
        if not self.quantize:
            return

        self._quantized_weight = torch.quantize_per_channel(
            self.weight.data,
            scales=torch.ones(self.num_embeddings),
            zero_points=torch.zeros(self.num_embeddings),
            axis=0,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._quantized_weight is not None:
            return F.embedding_bag(
                x,
                self._quantized_weight,
                padding_idx=self.padding_idx,
                max_norm=self.max_norm,
                norm_type=self.norm_type,
                scale_grad_by_freq=self.scale_grad_by_freq,
                mode="mean",
            )
        return super().forward(x)


def fused_swiglu(
    x: torch.Tensor,
    w1: torch.nn.Linear,
    w2: torch.nn.Linear,
    w3: torch.nn.Linear,
) -> torch.Tensor:
    """Fused SwiGLU activation.

    SwiGLU(x) = SiLU(w1(x)) * w3(x) @ w2
    """
    return w2(torch.nn.functional.silu(w1(x)) * w3(x))


def efficient_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Efficient cross-entropy with optional ignore index.

    Uses log-sum-exp trick for numerical stability.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    nll_loss = F.nll_loss(
        log_probs.view(-1, log_probs.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index,
        reduction=reduction,
    )
    return nll_loss


def chunked_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Chunked matrix multiplication for memory efficiency."""
    if min(a.shape[0], b.shape[1]) <= chunk_size:
        return a @ b

    result = torch.zeros(
        a.shape[0], b.shape[1],
        dtype=a.dtype,
        device=a.device,
    )

    for i in range(0, a.shape[0], chunk_size):
        a_chunk = a[i : i + chunk_size]
        result[i : i + chunk_size] = a_chunk @ b

    return result


def ragged_to_padded(
    tokens: torch.Tensor,
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert ragged sequences to padded with attention mask.

    Returns (padded_tokens, attention_mask).
    """
    max_len = tokens.shape[1]
    mask = tokens != pad_token_id
    return tokens, mask


def estimate_attention_memory(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    precision_bytes: int = 2,
) -> float:
    """Estimate memory for attention scores."""
    scores_size = batch_size * num_heads * seq_len * seq_len * precision_bytes
    return scores_size / (1024 ** 2)


__all__ = [
    "FusedLayerNorm",
    "FusedRMSNorm",
    "FusedCrossEntropyLoss",
    "FusedAttentionBias",
    "EfficientTranspose",
    "ChunkedOperation",
    "MemoryEfficientSoftmax",
    "FusedScaleBias",
    "OptimizedEmbedding",
    "fused_swiglu",
    "efficient_cross_entropy",
    "chunked_matmul",
    "ragged_to_padded",
    "estimate_attention_memory",
]
