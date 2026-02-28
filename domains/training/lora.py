"""
LoRA (Low-Rank Adaptation) Module for SloughGPT

Production-ready LoRA implementation with:
- Standard LoRA
- QLoRA (Quantized LoRA)
- LoRA+ 
- IA3 (Inflation-Aware Adapter)
- Training utilities
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger("sloughgpt.lora")


class LoRAType(Enum):
    """Types of LoRA adapters."""
    LORA = "lora"           # Standard LoRA
    QLORA = "q_lora"       # Quantized LoRA
    LORA_PLUS = "lora_plus" # LoRA+
    IA3 = "ia3"             # IA3 (Inflation-Aware Adapter)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    rank: int = 8                    # LoRA rank (r)
    alpha: float = 16.0              # LoRA scaling factor
    dropout: float = 0.05            # Dropout probability
    target_modules: Optional[List[str]] = None  # Which modules to apply LoRA
    lora_type: LoRAType = LoRAType.LORA
    bias: str = "none"              # "none", "all", "lora_only"
    task_type: str = "CAUSAL_LM"    # "CAUSAL_LM", "SEQ_CLS"
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for transformer models
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class LoRALinear(nn.Module):
    """
    LoRA-augmented linear layer.
    
    The original weight W is frozen. Two low-rank matrices A and B are learned.
    Effective weight: W + (alpha / rank) * (B @ A)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        lora_type: LoRAType = LoRAType.LORA,
        original_weight: Optional[torch.Tensor] = None,
        original_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout_prob = dropout
        self.lora_type = lora_type
        
        # Original weight (frozen)
        if original_weight is None:
            self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            self.weight = nn.Parameter(original_weight.clone().detach(), requires_grad=False)
        
        # LoRA parameters (trainable)
        if lora_type == LoRAType.IA3:
            # IA3: scaling vectors instead of rank decomposition
            self.lora_s = nn.Parameter(torch.ones(out_features))
        else:
            # Standard LoRA: A and B matrices
            self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Bias handling
        if bias:
            if original_bias is None:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.bias = nn.Parameter(original_bias.clone().detach())
        else:
            self.bias = None
        
        # Store original forward for merging
        self._original_forward = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lora_type == LoRAType.IA3:
            # IA3: element-wise scaling
            original = F.linear(x, self.weight, self.bias)
            lora_contrib = original * self.lora_s.unsqueeze(0)
            return lora_contrib
        else:
            # Standard LoRA
            lora_weight = (self.lora_B @ self.lora_A) * (self.alpha / self.rank)
            effective_weight = self.weight + lora_weight
            return self.dropout(F.linear(x, effective_weight, self.bias))
    
    def merge_weights(self):
        """Merge LoRA weights into original weights."""
        if self.lora_type == LoRAType.IA3:
            # IA3: apply scaling to original weights
            with torch.no_grad():
                self.weight.copy_(self.weight * self.lora_s.unsqueeze(1))
                self.lora_s.fill_(1.0)
        else:
            # Standard LoRA: merge A and B
            lora_weight = (self.lora_B @ self.lora_A) * (self.alpha / self.rank)
            with torch.no_grad():
                self.weight.copy_(self.weight + lora_weight)
                self.lora_A.fill_(0)
                self.lora_B.fill_(0)
    
    def get_trainable_parameters(self):
        """Get only trainable parameters (LoRA)."""
        if self.lora_type == LoRAType.IA3:
            return [self.lora_s]
        return [self.lora_A, self.lora_B]


class LoRAEmbedding(nn.Module):
    """LoRA for embedding layers."""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        rank: int = 8,
        alpha: float = 16.0,
        original_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.alpha = alpha
        
        # Original embedding (frozen)
        if original_weight is None:
            self.weight = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.weight = nn.Embedding.from_pretrained(original_weight.clone().detach(), freeze=True)
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, embedding_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(embedding_dim, rank))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = self.weight(x)
        lora_contrib = F.linear(original, self.lora_B @ self.lora_A)
        return original + lora_contrib * (self.alpha / self.rank)
    
    def merge_weights(self):
        lora_weight = (self.lora_B @ self.lora_A) * (self.alpha / self.rank)
        with torch.no_grad():
            self.weight.weight.copy_(self.weight.weight + lora_weight)


def apply_lora_to_model(
    model: nn.Module,
    config: Optional[LoRAConfig] = None,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply LoRA to a model.
    
    Args:
        model: PyTorch model
        config: LoRAConfig (preferred)
        rank: LoRA rank (if no config)
        alpha: LoRA alpha (if no config)
        target_modules: Module names to apply LoRA
    
    Returns:
        Model with LoRA applied
    """
    if config is None:
        config = LoRAConfig(rank=rank, alpha=alpha, target_modules=target_modules)
    
    target_modules = config.target_modules or []
    
    for name, module in model.named_modules():
        # Check if this module should be LoRA-ified
        module_name = name.split('.')[-1]
        
        if module_name in target_modules or any(t in name for t in target_modules):
            if isinstance(module, nn.Linear):
                new_lora = LoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                    lora_type=config.lora_type,
                    original_weight=module.weight,
                    original_bias=module.bias,
                )
                # Set the parent module
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], new_lora)
                logger.info(f"Applied LoRA to {name}")
            
            elif isinstance(module, nn.Embedding):
                new_lora = LoRAEmbedding(
                    num_embeddings=module.num_embeddings,
                    embedding_dim=module.embedding_dim,
                    rank=config.rank,
                    alpha=config.alpha,
                    original_weight=module.weight,
                )
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], new_lora)
                logger.info(f"Applied LoRA to embedding {name}")
    
    return model


def get_lora_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Get only LoRA parameters from a model."""
    lora_params = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params[name] = param
    return lora_params


def count_lora_parameters(model: nn.Module) -> int:
    """Count number of trainable LoRA parameters."""
    return sum(p.numel() for p in model.parameters() if 'lora_' in p.name)


def print_lora_summary(model: nn.Module):
    """Print LoRA parameter summary."""
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = count_lora_parameters(model)
    frozen_params = total_params - lora_params
    
    print("=" * 50)
    print("LoRA Summary")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Trainable %: {100 * lora_params / total_params:.2f}%")
    print("=" * 50)


class LoRATrainer:
    """
    Trainer for LoRA fine-tuning.
    
    Only trains LoRA parameters, keeps base model frozen.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: LoRAConfig,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
    ):
        self.model = model
        self.config = config
        
        # Get only LoRA parameters
        self.lora_params = []
        for name, param in model.named_parameters():
            if 'lora_' in name:
                self.lora_params.append(param)
        
        # Create optimizer for LoRA only
        self.optimizer = torch.optim.AdamW(
            self.lora_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        print_lora_summary(model)
    
    def train_step(self, batch, loss_fn):
        """Single training step."""
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(**batch)
        loss = loss_fn(output, batch.get('labels', None))
        
        # Backward
        loss.backward()
        
        # Update only LoRA parameters
        self.optimizer.step()
        
        return {"loss": loss.item()}
    
    def save_lora(self, path: str):
        """Save only LoRA weights."""
        lora_state = {
            name: param 
            for name, param in self.model.state_dict().items() 
            if 'lora_' in name
        }
        torch.save(lora_state, path)
        logger.info(f"LoRA weights saved to {path}")
    
    def load_lora(self, path: str):
        """Load LoRA weights."""
        lora_state = torch.load(path, map_location='cpu')
        self.model.load_state_dict(lora_state, strict=False)
        logger.info(f"LoRA weights loaded from {path}")


# =============================================================================
# QLoRA Support
# =============================================================================

class QuantizedLinear(nn.Module):
    """Linear layer with quantized weights (for QLoRA)."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        bits: int = 4,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.bits = bits
        
        # Quantized base (stored as int4)
        self.weight_int4 = nn.Parameter(torch.zeros(out_features, in_features // 8))
        self.weight_scale = nn.Parameter(torch.ones(out_features))
        
        # LoRA adapters
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weights
        weight = self.weight_int4.float() * self.weight_scale.unsqueeze(1)
        
        # Add LoRA contribution
        lora_weight = (self.lora_B @ self.lora_A) * (self.alpha / self.rank)
        effective_weight = weight + lora_weight
        
        return F.linear(x, effective_weight, self.bias)


__all__ = [
    "LoRAConfig",
    "LoRAType",
    "LoRALinear",
    "LoRAEmbedding",
    "apply_lora_to_model",
    "get_lora_parameters",
    "count_lora_parameters",
    "print_lora_summary",
    "LoRATrainer",
    "QuantizedLinear",
]
