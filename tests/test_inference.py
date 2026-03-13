"""
SloughGPT Inference Tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from domains.training.models.nanogpt import NanoGPT


class TestInferenceEngine:
    """Tests for InferenceEngine."""
    
    @pytest.fixture
    def model(self):
        """Create a small test model."""
        model = NanoGPT(
            vocab_size=50,
            n_embed=32,
            n_layer=2,
            n_head=2,
            block_size=32
        )
        return model
    
    @pytest.fixture
    def engine(self, model):
        """Create inference engine."""
        from domains.training.inference_engine import InferenceEngine
        
        stoi = {chr(i+65): i for i in range(50)}
        itos = {i: chr(i+65) for i in range(50)}
        
        return InferenceEngine(model, stoi, itos, device="cpu")
    
    def test_encode(self, engine):
        """Test text encoding."""
        ids = engine.encode("ABC")
        assert ids.shape == (1, 3)
        assert ids[0, 0].item() == 0  # A = 0
        assert ids[0, 1].item() == 1  # B = 1
        assert ids[0, 2].item() == 2  # C = 2
    
    def test_decode(self, engine):
        """Test tensor decoding."""
        ids = torch.tensor([[0, 1, 2]])
        text = engine.decode(ids)
        assert text == "ABC"
    
    def test_generate_config_defaults(self, engine):
        """Test generation with default config."""
        from domains.training.inference_engine import GenerationConfig
        
        config = GenerationConfig(max_new_tokens=10)
        result = engine.generate("A", config)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_generate_with_temperature(self, engine):
        """Test generation with custom temperature."""
        from domains.training.inference_engine import GenerationConfig
        
        config = GenerationConfig(
            max_new_tokens=10,
            temperature=0.5
        )
        result = engine.generate("A", config)
        assert isinstance(result, str)
    
    def test_generate_stream(self, engine):
        """Test streaming generation."""
        from domains.training.inference_engine import GenerationConfig
        
        config = GenerationConfig(max_new_tokens=5)
        chars = list(engine.generate_stream("A", config))
        assert len(chars) > 0
        assert all(isinstance(c, str) for c in chars)
    
    def test_generate_batch(self, engine):
        """Test batch generation."""
        from domains.training.inference_engine import GenerationConfig
        
        config = GenerationConfig(max_new_tokens=5)
        prompts = ["A", "B", "C"]
        results = engine.generate_batch(prompts, config)
        
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)
    
    def test_repetition_penalty(self, engine):
        """Test generation with repetition penalty."""
        from domains.training.inference_engine import GenerationConfig
        
        config = GenerationConfig(
            max_new_tokens=10,
            repetition_penalty=1.5
        )
        result = engine.generate("A", config)
        assert isinstance(result, str)
    
    def test_top_k_filtering(self, engine):
        """Test top-k filtering."""
        from domains.training.inference_engine import GenerationConfig
        
        config = GenerationConfig(
            max_new_tokens=10,
            top_k=10
        )
        result = engine.generate("A", config)
        assert isinstance(result, str)
    
    def test_top_p_filtering(self, engine):
        """Test nucleus (top-p) filtering."""
        from domains.training.inference_engine import GenerationConfig
        
        config = GenerationConfig(
            max_new_tokens=10,
            top_p=0.9
        )
        result = engine.generate("A", config)
        assert isinstance(result, str)


class TestLoadModel:
    """Tests for model loading."""
    
    def test_load_model_for_inference(self):
        """Test loading model for inference."""
        from domains.training.inference_engine import load_model_for_inference
        import tempfile
        
        # Create a small test model
        model = NanoGPT(
            vocab_size=50,
            n_embed=32,
            n_layer=2,
            n_head=2,
            block_size=32
        )
        
        stoi = {chr(i+65): i for i in range(50)}
        itos = {i: chr(i+65) for i in range(50)}
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            torch.save({
                'model': model.state_dict(),
                'stoi': stoi,
                'itos': itos,
                'training_info': {
                    'vocab_size': 50,
                    'n_embed': 32,
                    'n_layer': 2,
                    'n_head': 2,
                    'block_size': 32,
                }
            }, checkpoint_path)
            
            # Load
            engine = load_model_for_inference(checkpoint_path, device="cpu")
            assert engine is not None
            assert engine.vocab_size == 50
        finally:
            os.unlink(checkpoint_path)
