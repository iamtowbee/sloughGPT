#!/usr/bin/env python3
"""
SloughGPT Configuration
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class Config:
    """SloughGPT configuration"""
    api_url: str = "http://localhost:8000"
    api_key: str = ""
    default_model: str = "gpt2"
    model_cache_dir: str = "./models"
    default_epochs: int = 3
    default_batch_size: int = 8
    default_lr: float = 1e-5
    output_format: str = "text"
    verbose: bool = False


_config: Optional[Config] = None


def load_config() -> Config:
    """Load configuration from environment"""
    global _config
    
    if _config is None:
        _config = Config(
            api_url=os.environ.get("SLOUGHGPT_API_URL", "http://localhost:8000"),
            api_key=os.environ.get("SLOUGHGPT_API_KEY", ""),
            default_model=os.environ.get("SLOUGHGPT_DEFAULT_MODEL", "gpt2"),
            model_cache_dir=os.environ.get("SLOUGHGPT_MODEL_CACHE_DIR", "./models"),
            default_epochs=int(os.environ.get("SLOUGHGPT_DEFAULT_EPOCHS", "3")),
            default_batch_size=int(os.environ.get("SLOUGHGPT_DEFAULT_BATCH_SIZE", "8")),
            default_lr=float(os.environ.get("SLOUGHGPT_DEFAULT_LR", "1e-5")),
            output_format=os.environ.get("SLOUGHGPT_OUTPUT_FORMAT", "text"),
            verbose=os.environ.get("SLOUGHGPT_VERBOSE", "false").lower() == "true",
        )
    
    return _config


def get_config() -> Config:
    """Get current configuration"""
    return load_config()


def reset_config():
    """Reset configuration (mainly for testing)"""
    global _config
    _config = None
