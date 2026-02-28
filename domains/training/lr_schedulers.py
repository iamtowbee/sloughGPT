"""
Learning Rate Schedulers - Industry Standard Implementations

Based on best practices from:
- PyTorch Lightning, HuggingFace Accelerate, DeepSpeed
- Karpathy's minGPT, Meta's LLaMA training
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class SchedulerConfig:
    """Base scheduler configuration."""
    name: str = "none"
    initial_lr: float = 1e-4


@dataclass
class CosineAnnealingConfig(SchedulerConfig):
    """Cosine Annealing with Warm restarts."""
    name: str = "cosine"
    min_lr: float = 1e-6
    warmup_steps: int = 0
    total_steps: int = 10000
    num_cycles: float = 0.5


@dataclass
class WarmupConfig(SchedulerConfig):
    """Linear warmup then constant or decay."""
    name: str = "warmup"
    warmup_steps: int = 500
    warmup_start_lr: float = 1e-7


@dataclass  
class OneCycleConfig(SchedulerConfig):
    """OneCycle LR - peak then decay (best for LLMs)."""
    name: str = "onecycle"
    max_lr: float = 1e-3
    pct_start: float = 0.1
    anneal_strategy: str = "cos"


@dataclass
class CyclicConfig(SchedulerConfig):
    """Cyclic LR - oscillate between bounds."""
    name: str = "cyclic"
    base_lr: float = 1e-5
    max_lr: float = 1e-3
    step_size_up: int = 1000
    step_size_down: Optional[int] = None
    mode: str = "triangular2"


# =============================================================================
# Custom Scheduler Implementations
# =============================================================================

class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing with linear warmup.
    
    Best practice for LLM training:
    - Linear warmup (avoid training instability early)
    - Cosine decay (smooth learning rate decay)
    - Optional warm restarts
    
    LR Curve:
           ╭──────╮
          ╱        ╲
         ╱          ╲
        ╱            ╲
    ────╱              ╲───────
        warmup       min_lr
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        num_cycles: float = 0.5,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.num_cycles = num_cycles
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            progress = float(self.last_epoch - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * self.num_cycles * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


class PolynomialDecayScheduler(_LRScheduler):
    """
    Polynomial learning rate decay.
    
    Used in many production LLM trainings (e.g., some LLaMA configs).
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        min_lr: float = 0.0,
        power: float = 1.0,
        last_epoch: int = -1
    ):
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.power = power
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch >= self.total_steps:
            return [self.min_lr for _ in self.base_lrs]
        
        decay_ratio = (self.last_epoch / self.total_steps) ** self.power
        return [
            base_lr * (1 - decay_ratio) + self.min_lr * decay_ratio
            for base_lr in self.base_lrs
        ]


class LinearWarmupScheduler(_LRScheduler):
    """Linear warmup then hold or decay."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        base_lr: float,
        hold_steps: int = 0,
        decay_type: str = "none",
        min_lr: float = 0.0,
        total_steps: Optional[int] = None,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.hold_steps = hold_steps
        self.decay_type = decay_type
        self.min_lr = min_lr
        self.total_steps = total_steps or warmup_steps + hold_steps + 10000
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            factor = self.last_epoch / self.warmup_steps
            return [self.base_lr * factor for _ in self.base_lrs]
        elif self.last_epoch < self.warmup_steps + self.hold_steps:
            return self.base_lrs
        else:
            if self.decay_type == "cosine":
                progress = (self.last_epoch - self.warmup_steps - self.hold_steps) / (
                    self.total_steps - self.warmup_steps - self.hold_steps
                )
                cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
                return [
                    self.min_lr + (self.base_lr - self.min_lr) * cosine_factor
                    for _ in self.base_lrs
                ]
            elif self.decay_type == "linear":
                progress = (self.last_epoch - self.warmup_steps - self.hold_steps) / (
                    self.total_steps - self.warmup_steps - self.hold_steps
                )
                return [
                    self.base_lr * (1 - progress) + self.min_lr * progress
                    for _ in self.base_lrs
                ]
            else:
                return self.base_lrs


# =============================================================================
# Factory Function
# =============================================================================

def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    total_steps: Optional[int] = None,
    warmup_steps: int = 0,
    min_lr: float = 1e-6,
    max_lr: float = 1e-3,
    **kwargs
) -> _LRScheduler:
    """
    Factory function to create LR scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: One of: cosine, warmup, onecycle, cyclic, polynomial, none
        total_steps: Total training steps
        warmup_steps: Steps for warmup
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate (for cyclic/onecycle)
        **kwargs: Additional scheduler-specific arguments
    
    Returns:
        _LRScheduler: Configured scheduler
    
    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = create_scheduler(
        ...     optimizer, "cosine", 
        ...     total_steps=10000, warmup_steps=500
        ... )
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == "none":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, last_epoch=-1)
    
    elif scheduler_type == "cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps or 10000,
            min_lr=min_lr,
            num_cycles=kwargs.get("num_cycles", 0.5)
        )
    
    elif scheduler_type == "warmup":
        return LinearWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            base_lr=max_lr,
            decay_type=kwargs.get("decay_type", "cosine"),
            min_lr=min_lr,
            total_steps=total_steps
        )
    
    elif scheduler_type == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=kwargs.get("pct_start", 0.1),
            anneal_strategy=kwargs.get("anneal_strategy", "cos"),
            div_factor=kwargs.get("div_factor", 25.0),
            final_div_factor=kwargs.get("final_div_factor", 1e4)
        )
    
    elif scheduler_type == "cyclic":
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=min_lr,
            max_lr=max_lr,
            step_size_up=kwargs.get("step_size_up", 2000),
            step_size_down=kwargs.get("step_size_down", None),
            mode=kwargs.get("mode", "triangular2"),
            gamma=kwargs.get("gamma", 0.5)
        )
    
    elif scheduler_type == "polynomial":
        return PolynomialDecayScheduler(
            optimizer,
            total_steps=total_steps or 10000,
            min_lr=min_lr,
            power=kwargs.get("power", 1.0)
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Choose from: none, cosine, warmup, onecycle, cyclic, polynomial")


# =============================================================================
# Best Practices Summary
# =============================================================================

BEST_PRACTICES = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     LR SCHEDULER BEST PRACTICES                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. WARMUP IS ESSENTIAL                                                     ║
║     • Always use warmup (100-1000 steps) for LLMs                          ║
║     • Prevents training instability from random init                        ║
║     • Recommended: 0.1-0.3 of total training steps                          ║
║                                                                              ║
║  2. COSINE ANNEALING IS DEFAULT                                             ║
║     • Smoother decay than step/exponential                                  ║
║     • Better final convergence                                              ║
║     • LLaMA, GPT-3, and most modern LLMs use this                          ║
║                                                                              ║
║  3. ONECYCLE FOR FAST CONVERGENCE                                           ║
║     • Peak LR at ~30% of training, then decay                               ║
║     • Good for quick experiments / prototyping                              ║
║     • Can achieve similar results to cosine with less tuning               ║
║                                                                              ║
║  4. LR RANGES                                                               ║
║     • Embeddings: 1e-5 to 1e-4                                             ║
║     • Full model finetune: 1e-5 to 1e-4                                     ║
║     • LoRA: 1e-4 to 1e-3                                                   ║
║     • Base model pretraining: 1e-4 to 1e-3                                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

__all__ = [
    "SchedulerConfig",
    "CosineAnnealingConfig", 
    "WarmupConfig",
    "OneCycleConfig",
    "CyclicConfig",
    "WarmupCosineScheduler",
    "PolynomialDecayScheduler", 
    "LinearWarmupScheduler",
    "create_scheduler",
    "BEST_PRACTICES",
]
