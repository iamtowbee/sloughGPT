"""
SloughGPT Config Tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
import yaml


class TestConfigLoader:
    """Tests for configuration loader."""
    
    def test_load_default_config(self):
        """Test loading default config."""
        from config_loader import load_config, Config
        
        config = load_config("nonexistent.yaml")
        
        assert isinstance(config, Config)
        assert config.model.n_embed == 256
        assert config.training.epochs == 10
        assert config.lora.enabled is False
    
    def test_load_yaml_config(self):
        """Test loading from YAML file."""
        from config_loader import load_config
        
        config_data = {
            'model': {'n_embed': 128, 'n_layer': 4},
            'training': {'epochs': 5, 'batch_size': 32},
            'lora': {'enabled': True, 'rank': 4},
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            
            assert config.model.n_embed == 128
            assert config.model.n_layer == 4
            assert config.training.epochs == 5
            assert config.training.batch_size == 32
            assert config.lora.enabled is True
            assert config.lora.rank == 4
        finally:
            os.unlink(config_path)
    
    def test_model_config(self):
        """Test model config defaults."""
        from config_loader import ModelConfig
        
        cfg = ModelConfig()
        
        assert cfg.name == "sloughgpt"
        assert cfg.n_embed == 256
        assert cfg.n_layer == 6
        assert cfg.n_head == 8
        assert cfg.block_size == 128
    
    def test_training_config(self):
        """Test training config defaults."""
        from config_loader import TrainingConfig
        
        cfg = TrainingConfig()
        
        assert cfg.epochs == 10
        assert cfg.batch_size == 64
        assert cfg.learning_rate == 1e-3
        assert cfg.weight_decay == 0.01
        assert cfg.scheduler == "cosine"
    
    def test_lora_config(self):
        """Test LoRA config defaults."""
        from config_loader import LoRAConfig
        
        cfg = LoRAConfig()
        
        assert cfg.enabled is False
        assert cfg.rank == 8
        assert cfg.alpha == 16
        assert "c_attn" in cfg.target_modules
    
    def test_tracking_config(self):
        """Test tracking config defaults."""
        from config_loader import TrackingConfig
        
        cfg = TrackingConfig()
        
        assert cfg.enabled is False
        assert cfg.backend == "wandb"
        assert cfg.project == "sloughgpt"
        assert cfg.log_every == 10


class TestInferenceConfig:
    """Tests for inference configuration."""
    
    def test_generation_config_defaults(self):
        """Test GenerationConfig defaults."""
        from domains.training.inference_engine import GenerationConfig
        
        cfg = GenerationConfig()
        
        assert cfg.max_new_tokens == 100
        assert cfg.temperature == 0.8
        assert cfg.top_k == 50
        assert cfg.top_p == 0.9
        assert cfg.repetition_penalty == 1.0
        assert cfg.do_sample is True
    
    def test_generation_config_custom(self):
        """Test custom GenerationConfig."""
        from domains.training.inference_engine import GenerationConfig
        
        cfg = GenerationConfig(
            max_new_tokens=200,
            temperature=0.5,
            top_k=20,
            top_p=0.95,
            repetition_penalty=1.2,
        )
        
        assert cfg.max_new_tokens == 200
        assert cfg.temperature == 0.5
        assert cfg.top_k == 20
        assert cfg.top_p == 0.95
        assert cfg.repetition_penalty == 1.2
