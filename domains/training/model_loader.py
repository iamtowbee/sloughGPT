#!/usr/bin/env python3
"""
SloughGPT Model Loader
Unified interface for loading models: local, trained, or HuggingFace
"""

import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


class ModelLoader:
    """Unified model loader for SloughGPT."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
    
    def load_local(self, name: str = "sloughgpt") -> Tuple[Any, Dict]:
        """
        Load local trained model.
        
        Args:
            name: Model name (without .pt extension)
        
        Returns:
            (model, config) tuple
        """
        path = self.models_dir / f"{name}.pt"
        
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        checkpoint = torch.load(path, weights_only=False)
        
        # Import here to avoid circular imports
        from domains.training.models.nanogpt import NanoGPT
        
        config = checkpoint.get('config', {})
        vocab_size = checkpoint.get('vocab_size', len(checkpoint.get('chars', [])))
        
        model = NanoGPT(
            vocab_size=vocab_size,
            n_embed=config.get('n_embed', 128),
            n_layer=config.get('n_layer', 4),
            n_head=config.get('n_head', 4),
            block_size=config.get('block_size', 64)
        )
        
        # Load state dict (handle both 'model' and 'model_state_dict' keys)
        state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
        
        # Filter keys that match
        model_state = {}
        for k, v in state_dict.items():
            if k in model.state_dict():
                model_state[k] = v
        
        if model_state:
            model.load_state_dict(model_state, strict=False)
        
        return model, checkpoint
    
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
        
        # Add padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    def load(self, source: str = "local", name: str = "sloughgpt") -> Tuple[Any, Any]:
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
        return [p.stem for p in self.models_dir.glob("*.pt")]
    
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
        
        # Check if it's a HuggingFace model or our NanoGPT
        if hasattr(model, 'generate'):
            # Our NanoGPT
            if itos and stoi:
                idx = torch.tensor([[stoi.get(c, 0) for c in prompt[:1]]])
                with torch.no_grad():
                    output = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
                return ''.join([itos.get(int(i), '?') for i in output[0]])
            else:
                # No tokenizer, can't generate
                return "No tokenizer available"
        
        elif hasattr(model, 'generate'):
            # HuggingFace model
            inputs = tokenizer(prompt, return_tensors='pt')
            
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
    parser.add_argument('--source', choices=['local', 'huggingface'], default='local')
    parser.add_argument('--model', default='sloughgpt')
    parser.add_argument('--prompt', default='First')
    parser.add_argument('--tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--list', action='store_true', help='List available models')
    
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
        
        print(f"✓ Model loaded!")
        
        # Generate
        print(f"\nGenerating...")
        result = loader.generate(
            model,
            args.prompt,
            tokenizer=config if args.source == 'huggingface' else None,
            itos=config.get('itos') if args.source == 'local' else None,
            stoi=config.get('stoi') if args.source == 'local' else None,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
        )
        
        print(f"\n=== Generated ===")
        print(result)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
