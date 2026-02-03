#!/usr/bin/env python3
"""
SloughGPT Model Zoo
Pre-trained models and model management system
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import requests
from datetime import datetime

from sloughgpt.config import ModelConfig
from sloughgpt.neural_network import SloughGPT
from sloughgpt.optimizations import OptimizedSloughGPT, create_optimized_model
from sloughgpt.core.exceptions import SloughGPTError, create_error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Metadata for a model in the model zoo"""
    name: str
    description: str
    version: str
    config: ModelConfig
    file_size: int = 0
    parameters: int = 0
    created_at: str = ""
    tags: List[str] = None
    checksum: str = ""
    download_url: Optional[str] = None
    performance_metrics: Dict[str, float] = None
    training_data: Optional[str] = None
    license: str = "MIT"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

class ModelZoo:
    """Model zoo for managing pre-trained SloughGPT models"""
    
    def __init__(self, zoo_dir: str = "./model_zoo"):
        self.zoo_dir = Path(zoo_dir)
        self.zoo_dir.mkdir(exist_ok=True)
        self.models_file = self.zoo_dir / "models.json"
        self.models_dir = self.zoo_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize models database
        self.models = self._load_models()
        
        # Default models
        self._ensure_default_models()
    
    def _load_models(self) -> Dict[str, ModelMetadata]:
        """Load models metadata from JSON file"""
        if self.models_file.exists():
            try:
                with open(self.models_file, 'r') as f:
                    data = json.load(f)
                    return {name: ModelMetadata(**model_data) 
                           for name, model_data in data.items()}
            except Exception as e:
                logger.warning(f"Failed to load models database: {e}")
        
        return {}
    
    def _save_models(self):
        """Save models metadata to JSON file"""
        try:
            with open(self.models_file, 'w') as f:
                models_dict = {name: asdict(model) for name, model in self.models.items()}
                json.dump(models_dict, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save models database: {e}")
            raise create_error(SloughGPTError, "Failed to save models database", 
                            details={"error": str(e)})
    
    def _ensure_default_models(self):
        """Ensure default models are available"""
        default_models = {
            "sloughgpt-small": ModelMetadata(
                name="sloughgpt-small",
                description="Small SloughGPT model for quick experiments",
                version="1.0.0",
                config=ModelConfig(
                    vocab_size=10000,
                    d_model=256,
                    n_heads=8,
                    n_layers=6
                ),
                tags=["small", "experimental", "fast"]
            ),
            
            "sloughgpt-medium": ModelMetadata(
                name="sloughgpt-medium",
                description="Medium-sized SloughGPT model for general use",
                version="1.0.0",
                config=ModelConfig(
                    vocab_size=30000,
                    d_model=512,
                    n_heads=8,
                    n_layers=6
                ),
                tags=["medium", "balanced"]
            ),
            
            "sloughgpt-large": ModelMetadata(
                name="sloughgpt-large",
                description="Large SloughGPT model for high-quality generation",
                version="1.0.0",
                config=ModelConfig(
                    vocab_size=50000,
                    d_model=1024,
                    n_heads=16,
                    n_layers=12
                ),
                tags=["large", "high-quality"]
            ),
            
            "sloughgpt-code": ModelMetadata(
                name="sloughgpt-code",
                description="SloughGPT model optimized for code generation",
                version="1.0.0",
                config=ModelConfig(
                    vocab_size=40000,
                    d_model=768,
                    n_heads=12,
                    n_layers=8
                ),
                tags=["code", "programming"]
            )
        }
        
        # Add default models if they don't exist
        for name, model in default_models.items():
            if name not in self.models:
                self.models[name] = model
                logger.info(f"Added default model: {name}")
        
        self._save_models()
    
    def add_model(self, 
                  name: str,
                  description: str,
                  config: ModelConfig,
                  model_file: Union[str, Path],
                  tags: List[str] = None,
                  training_data: str = None) -> str:
        """Add a new model to the zoo"""
        try:
            model_file = Path(model_file)
            
            # Calculate file info
            file_size = model_file.stat().st_size if model_file.exists() else 0
            
            # Create model metadata
            metadata = ModelMetadata(
                name=name,
                description=description,
                version="1.0.0",
                config=config,
                file_size=file_size,
                parameters=self._estimate_parameters(config),
                tags=tags or [],
                training_data=training_data
            )
            
            # Copy model file to zoo
            target_path = self.models_dir / f"{name}.pt"
            if model_file != target_path:
                import shutil
                shutil.copy2(model_file, target_path)
            
            # Calculate checksum
            metadata.checksum = self._calculate_checksum(target_path)
            
            # Add to models database
            self.models[name] = metadata
            self._save_models()
            
            logger.info(f"Added model to zoo: {name}")
            return name
            
        except Exception as e:
            logger.error(f"Failed to add model {name}: {e}")
            raise create_error(SloughGPTError, f"Failed to add model {name}", 
                            details={"error": str(e)})
    
    def list_models(self, tags: List[str] = None) -> List[ModelMetadata]:
        """List all models in the zoo"""
        models = list(self.models.values())
        
        if tags:
            models = [model for model in models 
                     if any(tag in model.tags for tag in tags)]
        
        return sorted(models, key=lambda x: x.created_at, reverse=True)
    
    def get_model(self, name: str) -> Optional[ModelMetadata]:
        """Get model metadata by name"""
        return self.models.get(name)
    
    def load_model(self, 
                   name: str, 
                   optimize: bool = True) -> Union[SloughGPT, OptimizedSloughGPT]:
        """Load a model from the zoo"""
        metadata = self.get_model(name)
        if not metadata:
            raise create_error(SloughGPTError, f"Model not found: {name}")
        
        model_path = self.models_dir / f"{name}.pt"
        if not model_path.exists():
            # Try to download if URL is available
            if metadata.download_url:
                self._download_model(metadata.download_url, model_path)
            else:
                raise create_error(SloughGPTError, f"Model file not found: {model_path}")
        
        try:
            # Create model
            if optimize:
                model = create_optimized_model(
                    metadata.config,
                    enable_quantization=True,
                    enable_mixed_precision=True
                )
            else:
                model = SloughGPT(metadata.config)
            
            # Load weights
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            
            logger.info(f"Loaded model: {name} (parameters: {metadata.parameters:,})")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")
            raise create_error(SloughGPTError, f"Failed to load model {name}", 
                            details={"error": str(e)})
    
    def delete_model(self, name: str) -> bool:
        """Delete a model from the zoo"""
        if name not in self.models:
            logger.warning(f"Model not found: {name}")
            return False
        
        try:
            # Delete model file
            model_path = self.models_dir / f"{name}.pt"
            if model_path.exists():
                model_path.unlink()
            
            # Remove from database
            del self.models[name]
            self._save_models()
            
            logger.info(f"Deleted model: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {name}: {e}")
            return False
    
    def update_model(self, 
                    name: str,
                    description: str = None,
                    tags: List[str] = None,
                    performance_metrics: Dict[str, float] = None) -> bool:
        """Update model metadata"""
        if name not in self.models:
            logger.warning(f"Model not found: {name}")
            return False
        
        try:
            metadata = self.models[name]
            
            if description is not None:
                metadata.description = description
            
            if tags is not None:
                metadata.tags = tags
            
            if performance_metrics is not None:
                metadata.performance_metrics = performance_metrics
            
            self.models[name] = metadata
            self._save_models()
            
            logger.info(f"Updated model: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update model {name}: {e}")
            return False
    
    def search_models(self, 
                     query: str = "",
                     min_parameters: int = None,
                     max_parameters: int = None,
                     tags: List[str] = None) -> List[ModelMetadata]:
        """Search for models matching criteria"""
        models = list(self.models.values())
        
        # Filter by query (name/description)
        if query:
            query_lower = query.lower()
            models = [model for model in models 
                     if query_lower in model.name.lower() or 
                        query_lower in model.description.lower()]
        
        # Filter by parameters
        if min_parameters is not None:
            models = [model for model in models if model.parameters >= min_parameters]
        
        if max_parameters is not None:
            models = [model for model in models if model.parameters <= max_parameters]
        
        # Filter by tags
        if tags:
            models = [model for model in models 
                     if any(tag in model.tags for tag in tags)]
        
        return models
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get zoo statistics"""
        models = list(self.models.values())
        
        total_models = len(models)
        total_size = sum(model.file_size for model in models)
        total_parameters = sum(model.parameters for model in models)
        
        # Parameter distribution
        small_models = len([m for m in models if m.parameters < 100_000_000])
        medium_models = len([m for m in models if 100_000_000 <= m.parameters < 500_000_000])
        large_models = len([m for m in models if m.parameters >= 500_000_000])
        
        # Tag distribution
        all_tags = []
        for model in models:
            all_tags.extend(model.tags)
        tag_counts = {tag: all_tags.count(tag) for tag in set(all_tags)}
        
        return {
            "total_models": total_models,
            "total_size_mb": total_size / (1024 * 1024),
            "total_parameters": total_parameters,
            "parameter_distribution": {
                "small": small_models,
                "medium": medium_models,
                "large": large_models
            },
            "tag_distribution": tag_counts,
            "latest_model": models[0].name if models else None,
            "zoo_directory": str(self.zoo_dir.absolute())
        }
    
    def export_models(self, output_file: Union[str, Path]) -> bool:
        """Export models database to file"""
        try:
            output_path = Path(output_file)
            models_dict = {name: asdict(model) for name, model in self.models.items()}
            
            with open(output_path, 'w') as f:
                json.dump(models_dict, f, indent=2, default=str)
            
            logger.info(f"Exported {len(self.models)} models to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export models: {e}")
            return False
    
    def import_models(self, input_file: Union[str, Path]) -> int:
        """Import models from file"""
        try:
            input_path = Path(input_file)
            with open(input_path, 'r') as f:
                models_dict = json.load(f)
            
            imported_count = 0
            for name, model_data in models_dict.items():
                if name not in self.models:
                    self.models[name] = ModelMetadata(**model_data)
                    imported_count += 1
            
            if imported_count > 0:
                self._save_models()
                logger.info(f"Imported {imported_count} new models")
            
            return imported_count
            
        except Exception as e:
            logger.error(f"Failed to import models: {e}")
            return 0
    
    def _estimate_parameters(self, config: ModelConfig) -> int:
        """Estimate parameter count from config"""
        # Simplified parameter count estimation
        embed_params = config.vocab_size * config.d_model
        attention_params = config.d_model * config.d_model
        ffn_params = config.d_model * config.d_model
        
        per_layer = attention_params + ffn_params
        total_params = embed_params + per_layer * config.n_layers
        
        return total_params
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""
    
    def _download_model(self, url: str, target_path: Path):
        """Download model from URL"""
        try:
            logger.info(f"Downloading model from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Downloaded model to {target_path}")
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

# Global model zoo instance
_model_zoo = None

def get_model_zoo() -> ModelZoo:
    """Get global model zoo instance"""
    global _model_zoo
    if _model_zoo is None:
        _model_zoo = ModelZoo()
    return _model_zoo

# Factory functions
def create_model(name: str, 
                optimize: bool = True) -> Union[SloughGPT, OptimizedSloughGPT]:
    """Create model from zoo"""
    zoo = get_model_zoo()
    return zoo.load_model(name, optimize)

def list_available_models(tags: List[str] = None) -> List[ModelMetadata]:
    """List available models"""
    zoo = get_model_zoo()
    return zoo.list_models(tags)

def search_models(query: str = "", 
                 tags: List[str] = None) -> List[ModelMetadata]:
    """Search models"""
    zoo = get_model_zoo()
    return zoo.search_models(query=query, tags=tags)

if __name__ == "__main__":
    # Demo model zoo functionality
    zoo = get_model_zoo()
    
    print("🦁 SloughGPT Model Zoo")
    print("=" * 40)
    
    # List models
    models = zoo.list_models()
    print(f"Available models ({len(models)}):")
    for model in models:
        print(f"  📊 {model.name}: {model.description}")
        print(f"     Parameters: {model.parameters:,}")
        print(f"     Tags: {', '.join(model.tags)}")
        print()
    
    # Show statistics
    stats = zoo.get_statistics()
    print("📈 Zoo Statistics:")
    print(f"  Total models: {stats['total_models']}")
    print(f"  Total size: {stats['total_size_mb']:.1f} MB")
    print(f"  Total parameters: {stats['total_parameters']:,}")
    
    # Demo loading a model
    if models:
        print("\n🔄 Demo: Loading sloughgpt-small...")
        try:
            model = zoo.load_model("sloughgpt-small")
            print(f"✅ Loaded model with {model.count_parameters():,} parameters")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")