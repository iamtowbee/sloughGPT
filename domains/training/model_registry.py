"""
Model Registry - Dynamically loadable models for SloughGPT

Models are auto-discovered from the models directory only.
"""

from typing import Dict, Any, Optional, Type, List
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import logging

logger = logging.getLogger("sloughgpt.models")


def _get_nn():
    """Lazily get torch.nn module."""
    import torch
    return torch.nn


@dataclass
class ModelInfo:
    """Information about a model."""
    id: str
    name: str
    description: str
    model_class: Type
    default_config: Dict[str, Any]
    tags: List[str]


def BaseModel(config: Dict[str, Any]):
    """Factory function to create base model class (lazy torch import)."""
    nn = _get_nn()
    
    class _BaseModel(nn.Module):
        def __init__(self, config: Dict[str, Any]):
            super().__init__()
            self.config = config
        
        def forward(self, x, targets=None):
            raise NotImplementedError
        
        def get_num_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters())
    
    return _BaseModel(config)


class ModelRegistry:
    """Registry for dynamically loadable models."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
            cls._instance._discovered = False
        return cls._instance
    
    def _ensure_discovered(self):
        """Lazily discover models when needed."""
        if not self._discovered:
            self._discover_models()
            self._discovered = True
    
    def _discover_models(self):
        """Discover models from domains/training/models/ only."""
        
        models_dir = Path("domains/training/models")
        
        if not models_dir.exists():
            return
        
        for model_file in models_dir.glob("*.py"):
            if model_file.name.startswith("_"):
                continue
            
            self._load_from_file(model_file)
    
    def _load_from_file(self, model_file: Path):
        """Load model classes from a file."""
        nn = _get_nn()
        
        try:
            spec = importlib.util.spec_from_file_location(
                f"models.{model_file.stem}",
                model_file
            )
            
            if not spec or not spec.loader:
                return
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find model classes
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                
                attr = getattr(module, attr_name)
                
                if not isinstance(attr, type):
                    continue
                
                if not issubclass(attr, nn.Module):
                    continue
                
                if attr is nn.Module or attr is BaseModel:
                    continue
                
                if not hasattr(attr, 'forward'):
                    continue
                
                # Create ID
                model_id = f"{model_file.stem}-{attr_name.lower()}"
                
                if model_id in self._models:
                    continue
                
                # Get config
                config = {}
                if hasattr(module, 'DEFAULT_CONFIG'):
                    config = getattr(module, 'DEFAULT_CONFIG')
                
                # Get description
                desc = attr.__doc__ or f"Model from {model_file.name}"
                desc = desc.strip().split('\n')[0][:80]
                
                self._models[model_id] = ModelInfo(
                    id=model_id,
                    name=attr_name,
                    description=desc,
                    model_class=attr,
                    default_config=config,
                    tags=[model_file.stem]
                )
        
        except Exception as e:
            logger.debug(f"Error loading {model_file}: {e}")
    
    def get(self, model_id: str) -> Optional[ModelInfo]:
        self._ensure_discovered()
        return self._models.get(model_id)
    
    def list_models(self) -> List[ModelInfo]:
        self._ensure_discovered()
        return list(self._models.values())
    
    def create_model(self, model_id: str, config: Optional[Dict[str, Any]] = None):
        self._ensure_discovered()
        model_info = self.get(model_id)
        
        if not model_info:
            available = ", ".join(self._models.keys())
            raise ValueError(f"Model '{model_id}' not found. Available: {available}")
        
        final_config = {**model_info.default_config, **(config or {})}
        # Pass config as kwargs since models expect individual parameters
        return model_info.model_class(**final_config)


registry = ModelRegistry()


def get_available_models() -> List[ModelInfo]:
    return registry.list_models()


def create_model(model_id: str, config: Optional[Dict[str, Any]] = None):
    return registry.create_model(model_id, config)


__all__ = [
    "ModelRegistry",
    "ModelInfo", 
    "BaseModel",
    "registry",
    "get_available_models",
    "create_model",
]
