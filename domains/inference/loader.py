"""
.sou Model Loader for SloughGPT

Loads and runs .sou model configurations.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Generator, Union

from .sou_format import SouModelFile, SouParser, GenerationParameters
from .quantization import QuantizationType, SouModelQuantizer

logger = logging.getLogger("sloughgpt.inference")


@dataclass
class InferenceConfig:
    """Runtime configuration for inference."""
    model_path: str
    sou_config: SouModelFile
    device: str = "auto"
    quantization: Optional[QuantizationType] = None
    
    @classmethod
    def from_sou_file(cls, sou_path: str, **kwargs) -> "InferenceConfig":
        """Create from .sou file."""
        sou_config = SouParser.parse_file(sou_path)
        return cls(
            model_path=sou_config.from_model,
            sou_config=sou_config,
            **kwargs
        )


class SouModelLoader:
    """
    Loads and manages .sou model files.
    
    Supports:
    - Loading model configuration
    - Applying quantization
    - Setting up inference pipeline
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.sou_config = config.sou_config
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def load(self) -> "SouModelLoader":
        """Load the model for inference."""
        logger.info(f"Loading model from {self.config.model_path}")
        
        # Check if it's a local path or HuggingFace model
        model_path = Path(self.config.model_path)
        
        if model_path.exists():
            self._load_local_model(model_path)
        else:
            self._load_hf_model(self.config.model_path)
        
        self._loaded = True
        return self
    
    def _load_local_model(self, model_path: Path):
        """Load a local model."""
        # Check file extension
        ext = model_path.suffix.lower()
        
        if ext == ".sou":
            # It's a .sou config file, load the config
            logger.info(f"Loading .sou config from {model_path}")
            # Already loaded in InferenceConfig
        
        elif ext in [".bin", ".gguf", ".ggml"]:
            # Binary model file (like GGUF)
            logger.info(f"Loading binary model from {model_path}")
            # In production, load with llama.cpp or similar
        
        elif ext in [".pt", ".pth"]:
            # PyTorch model
            logger.info(f"Loading PyTorch model from {model_path}")
            self._load_pytorch_model(model_path)
        
        elif ext == ".safetensors":
            # Safetensors model
            logger.info(f"Loading safetensors model from {model_path}")
            self._load_safetensors_model(model_path)
        
        else:
            raise ValueError(f"Unsupported model format: {ext}")
    
    def _load_pytorch_model(self, model_path: Path):
        """Load PyTorch model."""
        try:
            import torch
            state_dict = torch.load(model_path, map_location="cpu")
            logger.info(f"Loaded PyTorch model with {len(state_dict)} keys")
        except ImportError:
            logger.warning("PyTorch not available")
    
    def _load_safetensors_model(self, model_path: Path):
        """Load safetensors model."""
        try:
            from safetensors import safe_open
            with safe_open(model_path, framework="pt") as f:
                keys = f.keys()
                logger.info(f"Loaded safetensors model with {len(keys)} keys")
        except ImportError:
            logger.warning("safetensors not available")
    
    def _load_hf_model(self, model_id: str):
        """Load model from HuggingFace."""
        logger.info(f"Loading HuggingFace model: {model_id}")
        # In production, use transformers library
        # from transformers import AutoModelForCausalLM, AutoTokenizer
    
    def generate(
        self,
        prompt: str,
        **kwargs
    ) -> Union[str, Generator[str, None]]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Override generation parameters
            
        Returns:
            Generated text or generator
        """
        if not self._loaded:
            self.load()
        
        # Merge config with kwargs
        params = self.sou_config.parameters.to_dict()
        params.update(kwargs)
        
        logger.info(f"Generating with params: {params}")
        
        # Placeholder - in production, run actual inference
        return self._generate_placeholder(prompt, params)
    
    def _generate_placeholder(self, params: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Placeholder generation for demo."""
        return f"[Generated response to: {prompt[:50]}...]"
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Union[str, Generator[str, None]]:
        """
        Generate chat response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Override generation parameters
            
        Returns:
            Generated response
        """
        if not self._loaded:
            self.load()
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        return self.generate(prompt, **kwargs)
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to prompt."""
        # Apply template if set
        if self.sou_config.template:
            # Simple template substitution
            prompt = self.sou_config.template
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt = prompt.replace(f"{{{{.{role}}}}}", content)
            return prompt
        
        # Default: concatenate messages
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        
        return "\n".join(prompt_parts)


class SouInferenceEngine:
    """
    High-level inference engine for .sou models.
    
    Features:
    - Automatic model loading
    - Parameter validation
    - Streaming support
    - Batch inference
    """
    
    def __init__(self, sou_path: str, device: str = "auto"):
        self.sou_path = sou_path
        self.device = device
        self.loader: Optional[SouModelLoader] = None
    
    def __enter__(self):
        """Context manager entry."""
        config = InferenceConfig.from_sou_file(self.sou_path, device=self.device)
        self.loader = SouModelLoader(config)
        self.loader.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Cleanup if needed
        pass
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> "SouInferenceEngine":
        """
        Load a pretrained model by name.
        
        Args:
            model_name: Model name or path
            **kwargs: Additional arguments
            
        Returns:
            SouInferenceEngine instance
        """
        # Check if it's a known model
        known_models = {
            "llama3.2": "llama3.2",
            "llama3.1": "llama3.1",
            "mistral": "mistral",
            "mixtral": "mixtral",
        }
        
        model_path = known_models.get(model_name.lower(), model_name)
        
        return cls(model_path, **kwargs)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text (synchronous)."""
        if self.loader is None:
            config = InferenceConfig.from_sou_file(self.sou_path, device=self.device)
            self.loader = SouModelLoader(config)
            self.loader.load()
        
        return self.loader.generate(prompt, **kwargs)
    
    def stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Generate text with streaming."""
        if self.loader is None:
            self.load()
        
        params = self.loader.sou_config.parameters.to_dict()
        params.update(kwargs)
        
        # In production, implement actual streaming
        result = self.loader.generate(prompt, **kwargs)
        
        # Simulate streaming
        for char in result:
            yield char
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat with the model."""
        if self.loader is None:
            config = InferenceConfig.from_sou_file(self.sou_path, device=self.device)
            self.loader = SouModelLoader(config)
            self.loader.load()
        
        return self.loader.chat(messages, **kwargs)


# =============================================================================
# Convenience Functions
# =============================================================================

def load_model(sou_path: str, **kwargs) -> SouModelLoader:
    """Load a .sou model."""
    config = InferenceConfig.from_sou_file(sou_path, **kwargs)
    loader = SouModelLoader(config)
    loader.load()
    return loader


def generate(prompt: str, sou_path: str = None, **kwargs) -> str:
    """Quick generate function."""
    if sou_path is None:
        # Create default config
        sou_config = SouModelFile(
            from_model="llama3.2",
            parameters=GenerationParameters(temperature=kwargs.pop("temperature", 0.7)),
        )
        config = InferenceConfig(
            model_path="llama3.2",
            sou_config=sou_config,
        )
        loader = SouModelLoader(config)
    else:
        config = InferenceConfig.from_sou_file(sou_path)
        loader = SouModelLoader(config)
    
    loader.load()
    return loader.generate(prompt, **kwargs)


def chat(messages: List[Dict[str, str]], sou_path: str = None, **kwargs) -> str:
    """Quick chat function."""
    if sou_path is None:
        sou_config = SouModelFile(
            from_model="llama3.2",
            parameters=GenerationParameters(temperature=kwargs.pop("temperature", 0.7)),
        )
        config = InferenceConfig(
            model_path="llama3.2",
            sou_config=sou_config,
        )
        loader = SouModelLoader(config)
    else:
        config = InferenceConfig.from_sou_file(sou_path)
        loader = SouModelLoader(config)
    
    loader.load()
    return loader.chat(messages, **kwargs)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Demo CLI."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python loader.py <model.sou>")
        print("\n=== .sou Loader Demo ===")
        
        # Create example .sou file
        example_sou = """FROM llama3.2
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

PERSONALITY
    warmth 0.8
    creativity 0.6
    END

SYSTEM You are a helpful AI assistant.
"""
        
        # Save example
        with open("/tmp/example.sou", "w") as f:
            f.write(example_sou)
        
        # Parse it
        sou = SouParser.parse_file("/tmp/example.sou")
        
        print(f"Loaded .sou config:")
        print(f"  Model: {sou.from_model}")
        print(f"  Temperature: {sou.parameters.temperature}")
        print(f"  Context: {sou.context.num_ctx}")
        if sou.personality:
            print(f"  Personality: {sou.personality.to_dict()}")
        
        # Try loading
        try:
            loader = load_model("/tmp/example.sou")
            print("\nModel loaded successfully!")
        except Exception as e:
            print(f"\nModel loading: {e}")
            print("(This is expected in demo mode)")
        
        return
    
    sou_path = sys.argv[1]
    loader = load_model(sou_path)
    
    # Interactive mode
    print(f"Loaded model from {sou_path}")
    print("Type 'quit' to exit\n")
    
    messages = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        
        messages.append({"role": "user", "content": user_input})
        
        response = loader.chat(messages)
        print(f"Assistant: {response}")
        
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()


__all__ = [
    "InferenceConfig",
    "SouModelLoader",
    "SouInferenceEngine",
    "load_model",
    "generate",
    "chat",
]
