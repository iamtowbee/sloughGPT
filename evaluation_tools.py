#!/usr/bin/env python3
"""Model evaluation tools for SloughGPT."""

import json
import math
import torch
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    perplexity: float
    loss: float
    accuracy: float
    samples: int


def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from loss."""
    return math.exp(loss)


def evaluate_model(model, data: List[str], stoi: Dict, itos: Dict, block_size: int = 128) -> EvaluationResult:
    """Evaluate model on text data."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for text in data[:100]:  # Limit to 100 samples
        if len(text) < 2:
            continue
            
        encoded = [stoi.get(c, 0) for c in text]
        
        for i in range(1, len(encoded) - 1):
            if i >= block_size:
                break
                
            x = torch.tensor([encoded[:i]], dtype=torch.long)
            y = torch.tensor([encoded[i]], dtype=torch.long)
            
            with torch.no_grad():
                logits, _ = model(x)
                loss = torch.nn.functional.cross_entropy(logits[:, -1, :], y)
                total_loss += loss.item()
                
                pred = logits[:, -1, :].argmax().item()
                if pred == y.item():
                    total_correct += 1
                    
            total_samples += 1
    
    avg_loss = total_loss / max(total_samples, 1)
    perplexity = calculate_perplexity(avg_loss)
    accuracy = total_correct / max(total_samples, 1)
    
    return EvaluationResult(
        perplexity=perplexity,
        loss=avg_loss,
        accuracy=accuracy,
        samples=total_samples
    )


def compare_models(results: Dict[str, EvaluationResult]) -> str:
    """Compare multiple model evaluations."""
    output = ["=== Model Comparison ===\n"]
    
    for name, result in results.items():
        output.append(f"{name}:")
        output.append(f"  Perplexity: {result.perplexity:.2f}")
        output.append(f"  Loss: {result.loss:.4f}")
        output.append(f"  Accuracy: {result.accuracy*100:.1f}%")
        output.append(f"  Samples: {result.samples}")
        output.append("")
    
    return "\n".join(output)


if __name__ == "__main__":
    import sys
    
    print("Model Evaluation Tools")
    print("Usage: Import evaluate_model() from this module")
    print("")
    print("Example:")
    print("  from evaluation_tools import evaluate_model")
    print("  result = evaluate_model(model, data, stoi, itos)")
    print(f"  Perplexity: {result.perplexity:.2f}")
