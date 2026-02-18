#!/usr/bin/env python3
"""
Unified Training Module - OOP Implementation
All training logic in ONE file with proper OOP principles.

Classes:
- TrainingConfig: Configuration dataclass
- DataLoader: Abstract base for data loading
- ModelWrapper: Abstract base for models
- Trainer: Main training orchestrator
- TrainingPipeline: Complete end-to-end pipeline
"""

import json
import time
import random
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator, Union
from datetime import datetime

import numpy as np

logger = logging.getLogger("sloughgpt.training")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Immutable training configuration."""
    
    # Data
    data_path: str = ""
    vocab_size: int = 500
    block_size: int = 128
    
    # Model
    model_id: str = "nanogpt-nanogpt"
    n_embed: int = 128
    n_layer: int = 3
    n_head: int = 4
    dropout: float = 0.1
    
    # Training
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-4
    max_batches: int = 100
    
    # Output
    output_path: Optional[str] = None
    checkpoint_interval: int = 100
    
    # Device
    device: str = "auto"
    
    def __post_init__(self):
        if self.device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_path": self.data_path,
            "vocab_size": self.vocab_size,
            "model_id": self.model_id,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "output_path": self.output_path,
        }


# ============================================================================
# DATA LOADING (Abstract + Implementations)
# ============================================================================

class DataLoader(ABC):
    """Abstract base class for data loading."""
    
    @abstractmethod
    def load(self) -> "DataLoader":
        """Load data from source."""
        pass
    
    @abstractmethod
    def tokenize(self, vocab_size: int) -> np.ndarray:
        """Convert data to tokens."""
        pass
    
    @abstractmethod
    def get_batch(self, batch_size: int, block_size: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield training batches."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return data size."""
        pass


class UniversalDataLoader(DataLoader):
    """Handles any data format: txt, jsonl, json, csv, bin, directories."""
    
    def __init__(self, data_path: str):
        self.path = Path(data_path)
        self.format = self._detect_format()
        self.data: Optional[np.ndarray] = None
        self.texts: List[str] = []
    
    def _detect_format(self) -> str:
        if self.path.is_dir():
            return "directory"
        
        ext = self.path.suffix.lower()
        format_map = {
            ".txt": "text",
            ".jsonl": "jsonl",
            ".json": "json",
            ".csv": "csv",
            ".bin": "binary",
        }
        return format_map.get(ext, "text")
    
    def load(self) -> "UniversalDataLoader":
        """Load data based on format."""
        if not self.path.exists():
            raise FileNotFoundError(f"Data not found: {self.path}")
        
        loaders = {
            "directory": self._load_directory,
            "text": self._load_text,
            "jsonl": self._load_jsonl,
            "json": self._load_json,
            "csv": self._load_csv,
            "binary": self._load_binary,
        }
        
        loader = loaders.get(self.format, self._load_text)
        loader()
        return self
    
    def _load_text(self):
        text = self.path.read_text(encoding="utf-8")
        self.texts = [text]
    
    def _load_directory(self):
        texts = []
        for pattern in ["*.txt", "*.md"]:
            for file in self.path.rglob(pattern):
                try:
                    texts.append(file.read_text(encoding="utf-8"))
                except Exception:
                    continue
        self.texts = texts if texts else [""]
    
    def _load_jsonl(self):
        texts = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                try:
                    import json
                    data = json.loads(line)
                    text = data.get('text') or data.get('content') or str(data)
                    texts.append(text)
                except Exception:
                    continue
        self.texts = texts if texts else [""]
    
    def _load_json(self):
        import json
        data = json.loads(self.path.read_text())
        
        if isinstance(data, list):
            self.texts = [str(item) for item in data]
        else:
            self.texts = [str(data)]
    
    def _load_csv(self):
        import csv
        texts = []
        for row in csv.DictReader(self.path.read_text().splitlines()):
            text = row.get('text') or row.get('content') or str(row)
            texts.append(text)
        self.texts = texts if texts else [""]
    
    def _load_binary(self):
        self.data = np.fromfile(self.path, dtype=np.uint16)
    
    def tokenize(self, vocab_size: int) -> np.ndarray:
        """Convert texts to token array."""
        if self.data is not None:
            return self.data
        
        all_text = "\n".join(self.texts)
        unique_chars = sorted(set(all_text))[:vocab_size - 4]
        
        char_to_idx = {c: i + 4 for i, c in enumerate(unique_chars)}
        char_to_idx['<PAD>'] = 0
        char_to_idx['<UNK>'] = 1
        char_to_idx['<BOS>'] = 2
        char_to_idx['<EOS>'] = 3
        
        tokens = [char_to_idx.get(c, 1) for c in all_text]
        self.data = np.array(tokens, dtype=np.uint16)
        
        return self.data
    
    def get_batch(self, batch_size: int, block_size: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield batches."""
        if self.data is None:
            self.tokenize(500)
        
        data = self.data
        max_idx = len(data) - block_size - 1
        
        while True:
            if max_idx <= 0:
                return
            
            n = min(batch_size, max_idx)
            indices = random.sample(range(max_idx), n)
            
            x = np.array([data[i:i+block_size] for i in indices])
            y = np.array([data[i+1:i+block_size+1] for i in indices])
            
            yield x, y
    
    def __len__(self) -> int:
        if self.data is not None:
            return len(self.data)
        return sum(len(t) for t in self.texts)


# ============================================================================
# MODEL WRAPPER (Abstract + Implementation)
# ============================================================================

class ModelWrapper(ABC):
    """Abstract wrapper for ML models."""
    
    @abstractmethod
    def train(self):
        """Set model to training mode."""
        pass
    
    @abstractmethod
    def eval(self):
        """Set model to evaluation mode."""
        pass
    
    @abstractmethod
    def forward(self, x: Any, y: Any) -> Tuple[Any, Any]:
        """Forward pass."""
        pass
    
    @abstractmethod
    def parameters(self) -> Any:
        """Get model parameters."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save model."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load model."""
        pass
    
    @abstractmethod
    def num_parameters(self) -> int:
        """Count parameters."""
        pass


class TorchModelWrapper(ModelWrapper):
    """Wrapper for PyTorch models."""
    
    def __init__(self, model: Any):
        self.model = model
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
    
    def forward(self, x: Any, y: Any) -> Tuple[Any, Any]:
        return self.model(x, y)
    
    def parameters(self) -> Any:
        return self.model.parameters()
    
    def save(self, path: str):
        import torch
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str):
        import torch
        self.model.load_state_dict(torch.load(path))
    
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


# ============================================================================
# TRAINER (Core Training Logic)
# ============================================================================

class Trainer:
    """Core training orchestrator with OOP principles."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model: Optional[ModelWrapper] = None
        self.data_loader: Optional[DataLoader] = None
        self.optimizer: Optional[Any] = None
        self.history: List[Dict[str, Any]] = []
        self._initialized = False
    
    def setup(self) -> "Trainer":
        """Initialize all components."""
        
        # Load data
        self.data_loader = UniversalDataLoader(self.config.data_path).load()
        tokens = self.data_loader.tokenize(self.config.vocab_size)
        
        logger.info(f"Data loaded: {len(tokens)} tokens")
        
        # Load model
        from domains.training.model_registry import create_model
        
        model = create_model(
            self.config.model_id,
            {
                "vocab_size": self.config.vocab_size,
                "n_embed": self.config.n_embed,
                "n_layer": self.config.n_layer,
                "n_head": self.config.n_head,
            }
        )
        
        self.model = TorchModelWrapper(model)
        logger.info(f"Model loaded: {self.model.num_parameters():,} parameters")
        
        # Setup optimizer
        import torch
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        self._initialized = True
        return self
    
    def train(self) -> Dict[str, Any]:
        """Execute training loop."""
        
        if not self._initialized:
            self.setup()
        
        self.model.train()
        
        total_loss = 0.0
        total_batches = 0
        
        logger.info(f"Training for {self.config.epochs} epochs")
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            batch_gen = self.data_loader.get_batch(
                self.config.batch_size,
                self.config.block_size
            )
            
            for batch_x, batch_y in batch_gen:
                # Convert to tensors
                import torch
                x = torch.tensor(batch_x.astype(np.int64), dtype=torch.long)
                y = torch.tensor(batch_y.astype(np.int64), dtype=torch.long)
                
                # Forward pass
                self.optimizer.zero_grad()
                logits, loss = self.model.forward(x, y)
                
                # Backward pass
                if loss is not None:
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    epoch_batches += 1
                
                # Safety limit
                if epoch_batches >= self.config.max_batches:
                    break
            
            avg_loss = epoch_loss / max(epoch_batches, 1)
            
            # Record history
            self.history.append({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "batches": epoch_batches,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"Epoch {epoch+1}/{self.config.epochs}: loss={avg_loss:.4f}")
            
            total_loss += epoch_loss
            total_batches += epoch_batches
        
        # Compile results
        results = {
            "model_id": self.config.model_id,
            "parameters": self.model.num_parameters(),
            "epochs": self.config.epochs,
            "final_loss": self.history[-1]["loss"] if self.history else 0.0,
            "total_batches": total_batches,
            "history": self.history,
            "config": self.config.to_dict(),
        }
        
        # Save if requested
        if self.config.output_path:
            self.model.save(self.config.output_path)
            
            # Save metadata
            meta_path = f"{self.config.output_path}.json"
            with open(meta_path, "w") as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Model saved to {self.config.output_path}")
        
        return results
    
    def evaluate(self, test_path: str) -> Dict[str, Any]:
        """Evaluate model on test data."""
        
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call setup() first.")
        
        self.model.eval()
        
        test_loader = UniversalDataLoader(test_path).load()
        test_tokens = test_loader.tokenize(self.config.vocab_size)
        
        total_loss = 0.0
        total_batches = 0
        
        batch_gen = test_loader.get_batch(
            self.config.batch_size,
            self.config.block_size
        )
        
        import torch
        
        for batch_x, batch_y in batch_gen:
            x = torch.tensor(batch_x.astype(np.int64), dtype=torch.long)
            y = torch.tensor(batch_y.astype(np.int64), dtype=torch.long)
            
            with torch.no_grad():
                logits, loss = self.model.forward(x, y)
                if loss is not None:
                    total_loss += loss.item()
                    total_batches += 1
        
        avg_loss = total_loss / max(total_batches, 1)
        perplexity = np.exp(avg_loss)
        
        return {
            "test_loss": avg_loss,
            "perplexity": perplexity,
            "test_batches": total_batches,
            "test_tokens": len(test_tokens),
        }


# ============================================================================
# TRAINING PIPELINE (High-level Interface)
# ============================================================================

class TrainingPipeline:
    """Complete end-to-end training pipeline."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.trainer = Trainer(config)
        self.results: Optional[Dict[str, Any]] = None
    
    def run(self) -> Dict[str, Any]:
        """Execute full pipeline."""
        
        print("🚀 Training Pipeline")
        print("=" * 40)
        print(f"Data: {self.config.data_path}")
        print(f"Model: {self.config.model_id}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Device: {self.config.device}")
        print()
        
        # Setup and train
        self.trainer.setup()
        self.results = self.trainer.train()
        
        print()
        print("✅ Training Complete")
        print(f"   Final loss: {self.results['final_loss']:.4f}")
        print(f"   Parameters: {self.results['parameters']:,}")
        
        if self.config.output_path:
            print(f"   Saved: {self.config.output_path}")
        
        return self.results
    
    def benchmark(self, test_path: str) -> Dict[str, Any]:
        """Run benchmark evaluation."""
        
        if self.results is None:
            raise RuntimeError("Pipeline not run. Call run() first.")
        
        return self.trainer.evaluate(test_path)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def train(
    data_path: str,
    model_id: str = "nanogpt-nanogpt",
    epochs: int = 5,
    output_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Train a model on any data format."""
    
    config = TrainingConfig(
        data_path=data_path,
        model_id=model_id,
        epochs=epochs,
        output_path=output_path,
        **kwargs
    )
    
    pipeline = TrainingPipeline(config)
    return pipeline.run()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Training")
    parser.add_argument("data_path", help="Path to training data")
    parser.add_argument("--model", default="nanogpt-nanogpt", help="Model ID")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs")
    parser.add_argument("--output", help="Output path")
    parser.add_argument("--vocab-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    results = train(
        data_path=args.data_path,
        model_id=args.model,
        epochs=args.epochs,
        output_path=args.output,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    
    print(f"\nResults: {json.dumps(results, indent=2)}")
