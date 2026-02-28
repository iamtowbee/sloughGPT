"""Simple inference script for a trained NanoGPT checkpoint.

Usage::

    python inference_example.py --checkpoint ./checkpoints/model.pt \
        --prompt "Once upon a time" --max_new_tokens 100

The script loads the checkpoint (model state dict, tokenizer, and config),
instantiates the ``NanoGPT`` model with the saved configuration, and generates
text using the model's ``generate`` method.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from domains.training.models.nanogpt import NanoGPT

def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    model_state = ckpt["model_state_dict"]
    tokenizer = ckpt["tokenizer"]
    config = ckpt["config"]
    return model_state, tokenizer, config

def main():
    parser = argparse.ArgumentParser(description="Generate text from a NanoGPT checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text to condition on")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top‑k sampling (optional)")
    args = parser.parse_args()

    model_state, tokenizer, config = load_checkpoint(args.checkpoint)

    # Re‑create the model with the saved configuration
    model = NanoGPT(
        vocab_size=config["vocab_size"],
        n_embed=config.get("n_embed", 384),
        n_layer=config.get("n_layer", 6),
        n_head=config.get("n_head", 6),
        block_size=config.get("block_size", 256),
        dropout=0.0,
    )
    model.load_state_dict(model_state)
    model.eval()

    device = torch.device("cpu")
    model.to(device)

    # Encode prompt
    if args.prompt:
        idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long).to(device)
    else:
        idx = torch.randint(0, config["vocab_size"], (1, 1), dtype=torch.long).to(device)

    # Generation loop (mirrors NanoGPT.generate but allows temperature/top_k)
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            idx_cond = idx[:, -model.block_size :]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / args.temperature
            if args.top_k is not None:
                v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

    generated = tokenizer.decode(idx[0].tolist())
    print(generated)

if __name__ == "__main__":
    main()
