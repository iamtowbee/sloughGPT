#!/usr/bin/env python3
"""Configuration loader for SloughGPT."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str = "sloughgpt"
    vocab_size: Optional[int] = None
    n_embed: int = 256
    n_layer: int = 6
    n_head: int = 8
    block_size: int = 128
    dropout: float = 0.1


@dataclass
class DataConfig:
    dataset: str = "shakespeare"
    data_path: str = "datasets/shakespeare/input.txt"
    val_split: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    warmup_steps: int = 500
    scheduler: str = "cosine"


@dataclass
class LoRAConfig:
    enabled: bool = False
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["c_attn", "c_proj", "c_fc"])


@dataclass
class QuantizationConfig:
    enabled: bool = False
    bits: int = 8


@dataclass
class CheckpointConfig:
    save_every: int = 5
    keep_last: int = 3
    save_dir: str = "models"


@dataclass
class TrackingConfig:
    enabled: bool = False
    backend: str = "wandb"  # wandb, mlflow, none
    project: str = "sloughgpt"
    entity: Optional[str] = None
    log_every: int = 10


@dataclass
class DeviceConfig:
    type: str = "auto"  # auto, cpu, mps, cuda


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    path = Path(config_path)
    
    if not path.exists():
        return Config()
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    if data is None:
        return Config()
    
    return Config(
        model=ModelConfig(**data.get('model', {})),
        data=DataConfig(**data.get('data', {})),
        training=TrainingConfig(**data.get('training', {})),
        lora=LoRAConfig(**data.get('lora', {})),
        quantization=QuantizationConfig(**data.get('quantization', {})),
        checkpoint=CheckpointConfig(**data.get('checkpoint', {})),
        tracking=TrackingConfig(**data.get('tracking', {})),
        device=DeviceConfig(**data.get('device', {})),
    )


def get_device(config: DeviceConfig) -> str:
    """Get device string based on config."""
    if config.type == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return config.type


def merge_args_with_config(config: Config, args) -> Config:
    """Merge CLI args with config."""
    import argparse
    
    if hasattr(args, 'dataset') and args.dataset:
        config.data.dataset = args.dataset
        config.data.data_path = f"datasets/{args.dataset}/input.txt"
    
    if hasattr(args, 'epochs') and args.epochs:
        config.training.epochs = args.epochs
    
    if hasattr(args, 'batch_size') and args.batch_size:
        config.training.batch_size = args.batch_size
    
    if hasattr(args, 'lr') and args.lr:
        config.training.learning_rate = args.lr
    
    if hasattr(args, 'use_lora') and args.use_lora:
        config.lora.enabled = True
    
    if hasattr(args, 'resume') and args.resume:
        config.checkpoint.resume = args.resume
    
    if hasattr(args, 'max_steps') and args.max_steps:
        config.training.max_steps = args.max_steps
    
    return config


import torch
