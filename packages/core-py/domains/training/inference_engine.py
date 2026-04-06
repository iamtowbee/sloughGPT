#!/usr/bin/env python3
"""
SloughGPT Inference Engine
Supports streaming, batch processing, and advanced sampling.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Iterator, Dict
from dataclasses import dataclass

from domains.errors import require_non_empty_prompt


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    do_sample: bool = True
    seed: Optional[int] = None


class InferenceEngine:
    """
    High-performance inference engine for SloughGPT.

    Features:
    - Streaming generation
    - Batch processing
    - Advanced sampling (temperature, top-k, top-p)
    - Repetition penalty
    """

    def __init__(
        self,
        model,
        stoi: Dict[str, int],
        itos: Dict[int, str],
        device: str = "cpu",
    ):
        self.model = model
        self.stoi = stoi
        self.itos = itos
        self.device = device
        self.vocab_size = len(stoi)
        self.model.eval()

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to tensor."""
        ids = [self.stoi.get(c, 0) for c in text]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def decode(self, ids: torch.Tensor) -> str:
        """Decode tensor to text."""
        return "".join([self.itos.get(i, "?") for i in ids[0].tolist()])

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate text from prompt (non-streaming)."""
        if config is None:
            config = GenerationConfig()

        prompt = require_non_empty_prompt(prompt)

        if config.seed is not None:
            torch.manual_seed(config.seed)

        self.model.eval()
        idx = self.encode(prompt)

        for _ in range(config.max_new_tokens):
            # Get logits
            logits, _ = self.model(idx)
            logits = logits[:, -1, :]  # Last token

            # Apply temperature
            if config.temperature > 0:
                logits = logits / config.temperature

            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                for token_id in idx[0].tolist():
                    logits[0, token_id] /= config.repetition_penalty

            # Apply top-k filtering
            if config.top_k > 0:
                top_k = min(config.top_k, logits.size(-1))
                values, indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(1, indices, values)

            # Apply top-p (nucleus) filtering
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample
            if config.do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            idx = torch.cat([idx, next_token], dim=1)

            # Stop if EOS token (if we have one)
            if next_token.item() == self.stoi.get("\n", -1):
                break

        return self.decode(idx)

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Generate text with streaming output."""
        if config is None:
            config = GenerationConfig()

        prompt = require_non_empty_prompt(prompt)

        if config.seed is not None:
            torch.manual_seed(config.seed)

        self.model.eval()
        idx = self.encode(prompt)

        for i in range(config.max_new_tokens):
            logits, _ = self.model(idx)
            logits = logits[:, -1, :]

            if config.temperature > 0:
                logits = logits / config.temperature

            if config.repetition_penalty != 1.0:
                for token_id in idx[0].tolist():
                    logits[0, token_id] /= config.repetition_penalty

            if config.top_k > 0:
                top_k = min(config.top_k, logits.size(-1))
                values, indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(1, indices, values)

            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            if config.do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            idx = torch.cat([idx, next_token], dim=1)

            # Yield new token
            new_char = self.itos.get(next_token.item(), "?")
            yield new_char

            if next_token.item() == self.stoi.get("\n", -1):
                break

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        """Generate text for multiple prompts at once."""
        if config is None:
            config = GenerationConfig()

        if config.seed is not None:
            torch.manual_seed(config.seed)

        self.model.eval()

        # Encode all prompts
        max_len = max(len(p) for p in prompts)
        input_ids = []
        for p in prompts:
            ids = [self.stoi.get(c, 0) for c in p]
            ids.extend([0] * (max_len - len(ids)))  # Pad
            input_ids.append(ids)

        idx = torch.tensor(input_ids, dtype=torch.long, device=self.device)

        # Generate
        for _ in range(config.max_new_tokens):
            logits, _ = self.model(idx)
            logits = logits[:, -1, :]

            if config.temperature > 0:
                logits = logits / config.temperature

            if config.top_k > 0:
                for i in range(logits.size(0)):
                    top_k = min(config.top_k, logits.size(-1))
                    values, indices = torch.topk(logits[i], top_k)
                    logits[i] = torch.full_like(logits[i], float("-inf"))
                    logits[i, indices] = values

            if config.do_sample:
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(logits, dim=-1)

            idx = torch.cat([idx, next_tokens.unsqueeze(1)], dim=1)

        # Decode
        results = []
        for i in range(len(prompts)):
            text = "".join([self.itos.get(j, "?") for j in idx[i].tolist()])
            results.append(text)

        return results


def load_model_for_inference(checkpoint_path: str, device: str = "cpu") -> InferenceEngine:
    """Load a trained model for inference."""
    from domains.models import SloughGPTModel

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    if "training_info" in checkpoint:
        info = checkpoint["training_info"]
        vocab_size = info.get("vocab_size", 65)
        n_embed = info.get("n_embed", 128)
        n_layer = info.get("n_layer", 4)
        n_head = info.get("n_head", 4)
        block_size = info.get("block_size", 64)
    else:
        vocab_size = checkpoint.get("vocab_size", 65)
        n_embed = checkpoint.get("n_embed", 128)
        n_layer = checkpoint.get("n_layer", 4)
        n_head = checkpoint.get("n_head", 4)
        block_size = checkpoint.get("block_size", 64)

    model = SloughGPTModel(
        vocab_size=vocab_size,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
    )

    # Load state dict
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        model.load_state_dict(checkpoint["model"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)

    stoi = checkpoint.get("stoi", {})
    itos = checkpoint.get("itos", {})

    # Convert itos keys to int if needed
    if itos and isinstance(next(iter(itos.keys())), str):
        itos = {int(k): v for k, v in itos.items()}

    return InferenceEngine(model, stoi, itos, device)


if __name__ == "__main__":
    # Example usage
    print("Loading model...")
    engine = load_model_for_inference("models/sloughgpt.pt")

    # Single generation
    print("\n=== Single Generation ===")
    config = GenerationConfig(
        max_new_tokens=100,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )
    result = engine.generate("First", config)
    print(result)

    # Streaming generation
    print("\n=== Streaming Generation ===")
    for char in engine.generate_stream("First", config):
        print(char, end="", flush=True)
    print()

    # Batch generation
    print("\n=== Batch Generation ===")
    prompts = ["First", "Second", "Hello"]
    results = engine.generate_batch(prompts, config)
    for i, r in enumerate(results):
        print(f"{prompts[i]}: {r[:50]}...")
