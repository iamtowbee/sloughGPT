#!/usr/bin/env python3
"""
SloughGPT Model Loader
Unified interface for loading models: local, trained, or HuggingFace.
Supports: .pt, .safetensors, .gguf, .onnx
"""

import torch
from pathlib import Path
from typing import Tuple, Dict, Any, Optional


class ModelLoader:
    """Unified model loader for SloughGPT."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

    def _find_model_path(self, name: str) -> Path:
        """Find model path by trying different extensions."""
        name_path = Path(name)

        if name_path.suffix in (".pt", ".safetensors", ".gguf", ".onnx"):
            return name_path

        candidates = [
            self.models_dir / f"{name}.safetensors",
            self.models_dir / f"{name}.pt",
            self.models_dir / f"{name}-bf16.safetensors",
            self.models_dir / f"{name}.gguf",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Model not found: {name}. "
            f"Searched: {', '.join(str(c) for c in candidates)}"
        )

    def _load_safetensors(self, path: Path) -> Tuple[Dict, Optional[Dict]]:
        """Load model from SafeTensors format."""
        try:
            from safetensors import safe_open
        except ImportError:
            raise ImportError("pip install safetensors")

        state_dict = {}
        metadata = None
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = dict(f.metadata()) if f.metadata() else {}
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

        return state_dict, metadata

    def _load_pt(self, path: Path) -> Tuple[Dict, Dict]:
        """Load model from PyTorch checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        metadata = checkpoint.get("metadata") or checkpoint.get("config") or {}
        return state_dict, metadata

    def load_local(self, name: str = "sloughgpt") -> Tuple[Any, Dict]:
        """
        Load local trained model.

        Supports: .safetensors, .pt, .gguf

        Args:
            name: Model name (with or without extension)

        Returns:
            (model, config) tuple
        """
        path = self._find_model_path(name)

        if path.suffix == ".safetensors":
            state_dict, metadata = self._load_safetensors(path)
        elif path.suffix == ".gguf":
            state_dict, metadata = self._load_gguf(path)
        else:
            state_dict, metadata = self._load_pt(path)

        from domains.models import SloughGPTModel

        config = metadata or {}
        vocab_size = config.get(
            "vocab_size",
            config.get("vocab_size", 0) or 256,
        )
        if isinstance(vocab_size, dict):
            vocab_size = vocab_size.get("vocab_size", 256)

        model = SloughGPTModel(
            vocab_size=vocab_size,
            n_embed=config.get("n_embed", 128),
            n_layer=config.get("n_layer", 4),
            n_head=config.get("n_head", 4),
            block_size=config.get("block_size", 64),
        )

        model_state = {}
        model_keys = set(model.state_dict().keys())
        for k, v in state_dict.items():
            if k in model_keys:
                model_state[k] = v

        if model_state:
            model.load_state_dict(model_state, strict=False)

        return model, config

    def _load_gguf(self, path: Path) -> Tuple[Dict, Dict]:
        """Load model from GGUF file."""
        try:
            from gguf import GGUFReader
        except ImportError:
            raise ImportError("pip install gguf")

        reader = GGUFReader(str(path), "r")
        state_dict = {}
        for key in reader.tensors:
            tensor = reader.tensors[key]
            state_dict[key] = tensor.data

        metadata = {}
        for k, v in reader.metadata.items():
            if hasattr(v, "parts"):
                metadata[k] = str(v)
            else:
                metadata[k] = v

        return state_dict, metadata

    def load_huggingface(self, model_name: str = "gpt2") -> Tuple[Any, Any]:
        """
        Load model from HuggingFace.

        Args:
            model_name: HuggingFace model name

        Returns:
            (model, tokenizer) tuple
        """
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
        except ImportError:
            raise ImportError("transformers not installed: pip install transformers")

        print(f"Loading {model_name} from HuggingFace...")

        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def load(
        self, source: str = "local", name: str = "sloughgpt"
    ) -> Tuple[Any, Any]:
        """
        Unified load method.

        Args:
            source: "local" or "huggingface"
            name: Model name

        Returns:
            (model, config/tokenizer)
        """
        if source == "local":
            return self.load_local(name)
        elif source == "huggingface":
            return self.load_huggingface(name)
        else:
            raise ValueError(f"Unknown source: {source}")

    def list_local(self) -> list:
        """List available local models."""
        models = set()
        for ext in ("*.safetensors", "*.pt", "*.gguf", "*-bf16.safetensors"):
            for p in self.models_dir.glob(ext):
                name = p.stem
                if name.endswith("-bf16"):
                    name = name.replace("-bf16", "")
                models.add(name)
        return sorted(models)

    def generate(
        self,
        model: Any,
        prompt: str,
        tokenizer: Any = None,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        itos: Dict = None,
        stoi: Dict = None,
    ) -> str:
        """Generate text from model."""

        model.eval()

        if hasattr(model, "generate"):
            if itos and stoi:
                idx = torch.tensor([[stoi.get(c, 0) for c in prompt[:1]]])
                with torch.no_grad():
                    output = model.generate(
                        idx, max_new_tokens=max_new_tokens, temperature=temperature
                    )
                return "".join([itos.get(int(i), "?") for i in output[0]])
            else:
                return "No tokenizer available"

        elif hasattr(model, "generate"):
            inputs = tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            return tokenizer.decode(output[0])

        return "Unknown model type"


def main():
    """CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(description="SloughGPT Model Loader")
    parser.add_argument("--source", choices=["local", "huggingface"], default="local")
    parser.add_argument("--model", default="sloughgpt")
    parser.add_argument("--prompt", default="First")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--list", action="store_true", help="List available models")

    args = parser.parse_args()

    loader = ModelLoader()

    if args.list:
        print("Available local models:")
        for m in loader.list_local():
            print(f"  - {m}")
        return

    print(f"Loading {args.model} from {args.source}...")

    try:
        model, config = loader.load(args.source, args.model)

        print("Model loaded!")

        print("\nGenerating...")
        result = loader.generate(
            model,
            args.prompt,
            tokenizer=config if args.source == "huggingface" else None,
            itos=config.get("itos") if args.source == "local" else None,
            stoi=config.get("stoi") if args.source == "local" else None,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
        )

        print("\n=== Generated ===")
        print(result)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
