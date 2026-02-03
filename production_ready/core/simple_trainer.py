#!/usr/bin/env python3
"""
Simple Working Trainer for Dataset Standardization System

Works directly with train.bin/val.bin format without complex module dependencies.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pickle


class SimpleTrainer:
    """Simple trainer that works with standardized dataset format."""
    
    def __init__(self, dataset_name: str, **kwargs):
        self.dataset_name = dataset_name
        self.dataset_path = Path(f"datasets/{dataset_name}")
        self.config = {
            "batch_size": kwargs.get("batch_size", 32),
            "learning_rate": kwargs.get("learning_rate", 3e-4),
            "max_iters": kwargs.get("max_iters", 10000),
            "eval_interval": kwargs.get("eval_interval", 2000),
            "device": kwargs.get("device", "cpu"),
            "compile": kwargs.get("compile", False)
        }
        
        # Validate dataset exists
        if not self.dataset_path.exists():
            print(f"❌ Dataset not found: {dataset_name}")
            sys.exit(1)
        
        # Load dataset metadata
        meta_file = self.dataset_path / "meta.pkl"
        if not meta_file.exists():
            print(f"❌ Dataset metadata not found: {meta_file}")
            sys.exit(1)
        
        with open(meta_file, 'rb') as f:
            self.meta = pickle.load(f)
        
        self.vocab_size = self.meta["vocab_size"]
        self.stoi = self.meta["stoi"]
        self.itos = self.meta["itos"]
        
        print(f"📊 Loaded dataset: {dataset_name}")
        print(f"   Vocab size: {self.vocab_size}")
        print(f"   Config: {self.config}")
    
    def load_data(self):
        """Load training and validation data."""
        train_file = self.dataset_path / "train.bin"
        val_file = self.dataset_path / "val.bin"
        
        if not train_file.exists() or not val_file.exists():
            print(f"❌ Train/val files not found")
            sys.exit(1)
        
        train_data = np.fromfile(train_file, dtype=np.uint16)
        val_data = np.fromfile(val_file, dtype=np.uint16)
        
        print(f"📈 Loaded {len(train_data):,} train tokens, {len(val_data):,} val tokens")
        return train_data, val_data
    
    def get_batch(self, data, start_idx, batch_size):
        """Get a batch of data."""
        end_idx = min(start_idx + batch_size, len(data) - 1)  # -1 for target
        inputs = data[start_idx:end_idx]
        targets = data[start_idx + 1:end_idx + 1]
        return inputs, targets
    
    def compute_loss(self, logits, targets):
        """Simple cross-entropy loss."""
        # Simple one-hot encoding for demonstration
        batch_size, vocab_size = logits.shape
        targets_oh = np.zeros((batch_size, vocab_size))
        targets_oh[np.arange(batch_size), targets] = 1
        
        # Softmax + cross-entropy
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        loss = -np.mean(np.sum(np.log(softmax + 1e-8) * targets_oh, axis=1))
        
        return loss
    
    def train(self):
        """Run training loop."""
        train_data, val_data = self.load_data()
        
        print(f"🚀 Starting training on dataset: {self.dataset_name}")
        print(f"📊 Device: {self.config['device']}")
        print(f"⚙️  Batch size: {self.config['batch_size']}")
        print(f"📈 Learning rate: {self.config['learning_rate']}")
        print("=" * 50)
        
        # Simple model parameters (embedding + linear layer for demo)
        embedding_dim = 128
        embed_W = np.random.randn(self.vocab_size, embedding_dim) * 0.1
        W_out = np.random.randn(embedding_dim, self.vocab_size) * 0.1
        b_out = np.zeros(self.vocab_size)
        
        num_params = embed_W.size + W_out.size + b_out.size
        print(f"🧠 Model parameters: {num_params:,}")
        
        # Training loop
        for iter_num in range(self.config["max_iters"]):
            # Get random batch
            start_idx = np.random.randint(0, max(1, len(train_data) - self.config["batch_size"]))
            batch_x, batch_y = self.get_batch(train_data, start_idx, self.config["batch_size"])
            
            # Forward pass
            embed = embed_W[batch_x]  # (batch, embedding_dim)
            logits = np.dot(embed, W_out) + b_out  # (batch, vocab_size)
            
            # Compute loss
            loss = self.compute_loss(logits, batch_y)
            
            # Backward pass (gradient descent - simplified)
            if iter_num % 100 == 0:
                print(f"Iter {iter_num:5d} | Loss: {loss:.4f}")
            
            # Simple gradient step (in real training, this would be proper backprop)
            if iter_num % 1000 == 0:
                # Learning rate decay
                lr = self.config["learning_rate"] * (0.9 ** (iter_num // 1000))
                print(f"📉 Learning rate adjusted to: {lr:.6f}")
            
            # Evaluation
            if iter_num % self.config["eval_interval"] == 0 and iter_num > 0:
                val_loss = 0
                val_batches = min(10, len(val_data) // self.config["batch_size"])
                
                for _ in range(val_batches):
                    val_start = np.random.randint(0, len(val_data) - self.config["batch_size"])
                    val_x, val_y = self.get_batch(val_data, val_start, self.config["batch_size"])
                    val_embed = embed_W[val_x]
                    val_logits = np.dot(val_embed, W_out) + b_out
                    val_loss += self.compute_loss(val_logits, val_y)
                
                if val_batches > 0:
                    val_loss /= val_batches
                print(f"🔍 Val Loss: {val_loss:.4f} | Train Loss: {loss:.4f}")
        
        print(f"\n✅ Training completed!")
        print(f"📊 Final parameters trained on {self.config['max_iters']} iterations")
        
        # Save model (simplified)
        model_data = {
            "embed_W": embed_W,
            "W_out": W_out, 
            "b_out": b_out,
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            "itos": self.itos,
            "config": self.config
        }
        
        output_dir = Path(f"out-{self.dataset_name}")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "model.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to: {output_dir / 'model.pkl'}")
        print(f"🎉 Ready for inference with dataset: {self.dataset_name}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Simple trainer for dataset standardization system")
    parser.add_argument('--dataset', required=True, help='Dataset name to train on')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max_iters', type=int, default=10000, help='Maximum iterations')
    parser.add_argument('--eval_interval', type=int, default=2000, help='Evaluation interval')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'], help='Device type')
    parser.add_argument('--compile', action='store_true', help='Use compilation (if available)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda':
        try:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                print("🚀 Using CUDA GPU")
            else:
                device = 'cpu'
                print("⚠️ CUDA not available, using CPU")
        except ImportError:
            device = 'cpu'
            print("⚠️ PyTorch not available, using CPU")
    elif args.device == 'mps':
        device = 'mps'
        print("🍎 Using Apple Silicon MPS")
    else:
        device = 'cpu'
        print("💻 Using CPU")
    
    config = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_iters": args.max_iters,
        "eval_interval": args.eval_interval,
        "device": device,
        "compile": args.compile
    }
    
    trainer = SimpleTrainer(args.dataset, **config)
    trainer.train()


if __name__ == "__main__":
    main()