# HuggingFace Integration - Local Loader

Download and load HuggingFace models locally.
"""

import os
import shutil
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Generator
from urllib.parse import urlparse

logger = logging.getLogger("sloughgpt.huggingface")


@dataclass
class HFLocalConfig:
    """Configuration for local HuggingFace model."""
    model: str = "meta-llama/Llama-2-7b-chat-hf"
    cache_dir: str = "~/.cache/huggingface"
    revision: str = "main"
    local_files_only: bool = False
    
    # Quantization (for GGUF models)
    quantization: Optional[str] = None  # "q4_k_m", "q8_0", "f16", etc.
    
    # Device mapping
    device_map: str = "auto"  # "auto", "cuda", "cpu", or custom
    
    # Loading options
    low_cpu_mem_usage: bool = True
    torch_dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"
    
    def __post_init__(self):
        self.cache_dir = os.path.expanduser(self.cache_dir)


class HuggingFaceLocalLoader:
    """
    Download and load HuggingFace models locally.
    
    Features:
    - Download models from HF Hub
    - Cache management
    - Multiple format support (safetensors, pytorch, gguf)
    - Memory-efficient loading
    - Device mapping
    """
    
    # Model format detection
    FORMAT_EXTENSIONS = {
        ".safetensors": "safetensors",
        ".bin": "pytorch",
        ".pt": "pytorch",
        ".pth": "pytorch",
        ".gguf": "gguf",
        ".ggml": "ggml",
    }
    
    def __init__(self, config: HFLocalConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def download(self, force: bool = False) -> Path:
        """
        Download model from HuggingFace Hub.
        
        Args:
            force: Force re-download even if cached
            
        Returns:
            Path to downloaded model
        """
        import urllib.request
        import tarfile
        
        model_id = self.config.model
        cache_dir = Path(self.config.cache_dir) / "models" / model_id.replace("/", "--")
        
        if cache_dir.exists() and not force:
            logger.info(f"Model already cached at {cache_dir}")
            return cache_dir
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Download model files
        logger.info(f"Downloading {model_id}...")
        
        # Get model info from HF
        model_info = self._get_model_info()
        
        # Download safetensors file if available
        siblings = model_info.get("siblings", [])
        
        for sibling in siblings:
            filename = sibling.get("rfilename")
            if not filename:
                continue
                
            # Check if it's a model file
            if any(filename.endswith(ext) for ext in self.FORMAT_EXTENSIONS):
                self._download_file(model_id, filename, cache_dir)
        
        # Create metadata file
        metadata = {
            "model_id": model_id,
            "revision": self.config.revision,
            "downloaded_at": str(Path(__file__).stat().st_mtime),
        }
        
        with open(cache_dir / "metadata.json", "w") as f:
            import json
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model downloaded to {cache_dir}")
        return cache_dir
    
    def _download_file(self, model_id: str, filename: str, cache_dir: Path):
        """Download a single file from HF."""
        import urllib.request
        
        url = f"https://huggingface.co/{model_id}/resolve/{self.config.revision}/{filename}"
        dest = cache_dir / filename
        
        if dest.exists():
            logger.info(f"File already exists: {filename}")
            return
        
        logger.info(f"Downloading {filename}...")
        
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get('content-length', 0))
                
                with open(dest, 'wb') as f:
                    downloaded = 0
                    block_size = 8192
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        downloaded += len(buffer)
                        f.write(buffer)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024) == 0:  # Log every MB
                                logger.info(f"  {filename}: {progress:.1f}%")
                                
        except Exception as e:
            logger.warning(f"Failed to download {filename}: {e}")
            if dest.exists():
                dest.unlink()
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get model info from HuggingFace API."""
        import urllib.request
        import json
        import os
        
        url = f"https://huggingface.co/api/models/{self.config.model}"
        
        # Try to get token
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        
        req = urllib.request.Request(url)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        
        try:
            with urllib.request.urlopen(req) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            logger.warning(f"Could not fetch model info: {e}")
            return {"siblings": []}
    
    def load(self, download: bool = True) -> "HuggingFaceLocalLoader":
        """
        Load model into memory.
        
        Args:
            download: Download if not cached
            
        Returns:
            Self for chaining
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            logger.error("transformers not installed: pip install transformers")
            raise
        
        model_path = None
        
        # Check cache first
        cache_dir = Path(self.config.cache_dir) / "models" / self.config.model.replace("/", "--")
        
        if cache_dir.exists():
            model_path = cache_dir
        elif download:
            model_path = self.download()
        else:
            raise FileNotFoundError(f"Model not found in cache: {self.config.model}")
        
        # Determine torch dtype
        torch_dtype = self._get_torch_dtype()
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )
        
        # Load model
        logger.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            device_map=self.config.device_map,
            low_cpu_mem_usage=self.config.low_cpu_mem_usage,
            trust_remote_code=True,
        )
        
        self._loaded = True
        logger.info(f"Model loaded successfully!")
        
        return self
    
    def _get_torch_dtype(self):
        """Get torch dtype from config."""
        import torch
        
        dtype_map = {
            "auto": None,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        
        return dtype_map.get(self.config.torch_dtype, None)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Max tokens to generate
            **kwargs: Generation parameters
            
        Returns:
            Generated text
        """
        if not self._loaded:
            self.load()
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Chat completion.
        
        Args:
            messages: List of message dicts
            **kwargs: Generation parameters
            
        Returns:
            Assistant response
        """
        if not self._loaded:
            self.load()
        
        # Format messages
        prompt = self._format_chat(messages)
        
        return self.generate(prompt, **kwargs)
    
    def _format_chat(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages."""
        # Use chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        # Simple format
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        
        prompt_parts.append("assistant:")
        return "\n".join(prompt_parts)
    
    def unload(self):
        """Unload model from memory."""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    
    def get_memory_requirements(self) -> Dict[str, float]:
        """Get estimated memory requirements in GB."""
        if not self._loaded or self.model is None:
            return {"model": 0, "total": 0}
        
        import torch
        
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        
        total = param_size + buffer_size
        
        return {
            "model_gb": total / (1024 ** 3),
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }


class HuggingFaceLocalClient:
    """High-level client for local HF models."""
    
    def __init__(self, model: str, **kwargs):
        config = HFLocalConfig(model=model, **kwargs)
        self.loader = HuggingFaceLocalLoader(config)
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Quick generate."""
        return self.loader.generate(prompt, **kwargs)
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> "HuggingFaceLocalClient":
        """Create client from pretrained model."""
        return cls(model=model_name, **kwargs)


# =============================================================================
# Convenience Functions
# =============================================================================

def download_model(model: str, cache_dir: str = None, **kwargs) -> Path:
    """Download model from HuggingFace."""
    config = HFLocalConfig(model=model, cache_dir=cache_dir or "~/.cache/huggingface", **kwargs)
    loader = HuggingFaceLocalLoader(config)
    return loader.download()


def load_model(model: str, **kwargs) -> HuggingFaceLocalLoader:
    """Load model locally."""
    config = HFLocalConfig(model=model, **kwargs)
    loader = HuggingFaceLocalLoader(config)
    loader.load()
    return loader


def generate_local(prompt: str, model: str, **kwargs) -> str:
    """Quick generate from local model."""
    loader = load_model(model, **kwargs)
    return loader.generate(prompt)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Demo CLI."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python local_loader.py <model> [prompt]")
        print("\nExamples:")
        print("   python local_loader.py facebook/opt-125m 'Hello world'")
        print("   python local_loader.py meta-llama/Llama-2-7b 'Tell me a story'")
        return
    
    model = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello, how are you?"
    
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print("-" * 40)
    
    try:
        loader = load_model(model)
        print("Generating...")
        result = loader.generate(prompt, max_new_tokens=100)
        print(f"\nResult: {result}")
        
    except ImportError:
        print("Error: transformers not installed")
        print("  pip install transformers")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()


__all__ = [
    "HFLocalConfig",
    "HuggingFaceLocalLoader",
    "HuggingFaceLocalClient",
    "download_model",
    "load_model",
    "generate_local",
]
