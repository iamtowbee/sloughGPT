#!/usr/bin/env python3
"""
SloughGPT Training Pipeline
Complete training and fine-tuning capabilities for the SloughGPT model
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np

from .config import ModelConfig, LearningConfig
from .neural_network import SloughGPT
from .core.exceptions import SloughGPTError, create_error

# Optimizations - optional import
try:
    from optimizations import OptimizedSloughGPT
except ImportError:
    OptimizedSloughGPT = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    save_interval: int = 1000
    eval_interval: int = 500
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = False
    checkpoint_dir: str = "./checkpoints"
    data_dir: str = "./data"
    device: str = "auto"

class TextDataset(Dataset):
    """Simple text dataset for training"""
    
    def __init__(self, data: Union[str, List[str]], block_size: int = 512, vocab_size: int = 50257):
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        if isinstance(data, str):
            # Load from file
            with open(data, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            # Join list of strings
            text = '\n'.join(data)
        
        # Simple character-level tokenization
        # In a real implementation, you'd use proper tokenization
        self.tokens = [ord(c) % vocab_size for c in text]
        
        # Create training sequences
        self.examples = []
        for i in range(len(self.tokens) - block_size):
            self.examples.append((
                torch.tensor(self.tokens[i:i+block_size], dtype=torch.long),
                torch.tensor(self.tokens[i+1:i+block_size+1], dtype=torch.long)
            ))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

class SloughGPTTrainer:
    """Training class for SloughGPT models"""
    
    def __init__(self, 
                 model,
                 training_config: TrainingConfig,
                 learning_config: Optional[LearningConfig] = None):
        self.model = model
        self.training_config = training_config
        self.learning_config = learning_config or LearningConfig()
        
        # Setup device
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if training_config.use_mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_stats = {
            'losses': [],
            'learning_rates': [],
            'gradient_norms': [],
            'evaluation_losses': []
        }
        
        # Create checkpoint directory
        os.makedirs(training_config.checkpoint_dir, exist_ok=True)
    
    def _setup_device(self) -> torch.device:
        """Setup training device"""
        if self.training_config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.training_config.device)
        
        logger.info(f"🖥️  Using device: {device}")
        return device
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        return CosineAnnealingLR(
            self.optimizer,
            T_max=self.training_config.num_epochs,
            eta_min=1e-6
        )
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.training_config.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    logits = outputs['logits']
                    
                    # Calculate loss
                    loss = nn.CrossEntropyLoss()(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )
                
                # Backward pass with gradient scaling
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.training_config.gradient_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.training_config.gradient_clip_norm
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    
                    # Gradient clipping
                    if self.training_config.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.training_config.gradient_clip_norm
                        )
                    
                    self.optimizer.step()
            else:
                outputs = self.model(inputs)
                logits = outputs['logits']
                
                # Calculate loss
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.training_config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            # Update stats
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log progress
            if batch_idx % 100 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                grad_norm = torch.norm(torch.stack([
                    torch.norm(p.grad) for p in self.model.parameters() 
                    if p.grad is not None
                ])).item() if any(p.grad is not None for p in self.model.parameters()) else 0.0
                
                logger.info(f"Step {self.global_step}: Loss={loss.item():.4f}, LR={lr:.6f}, GradNorm={grad_norm:.4f}")
                
                self.training_stats['losses'].append(loss.item())
                self.training_stats['learning_rates'].append(lr)
                self.training_stats['gradient_norms'].append(grad_norm)
            
            # Save checkpoint
            if self.global_step % self.training_config.save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}")
            
            # Evaluation
            if self.global_step % self.training_config.eval_interval == 0:
                eval_loss = self.evaluate(dataloader)  # Using same data for simplicity
                if not isinstance(eval_loss, dict):
                    eval_loss = {"loss": eval_loss}
                self.training_stats['evaluation_losses'].append(eval_loss)
                if isinstance(eval_loss, dict):
                    eval_loss_val = eval_loss.get("loss", eval_loss)
                else:
                    eval_loss_val = eval_loss
                logger.info(f"Eval Loss: {eval_loss_val:.4f}")
        
        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}
    
    def evaluate(self, dataloader: DataLoader):
        """Evaluate model on validation data"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                logits = outputs['logits']
                
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.training_config.checkpoint_dir, f"{name}.pt")
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'training_config': self.training_config,
            'training_stats': self.training_stats
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"📂 Loaded checkpoint: {checkpoint_path}")
        logger.info(f"   Epoch: {self.current_epoch}, Step: {self.global_step}, Best Loss: {self.best_loss:.4f}")
    
    def train(self, train_data: Union[str, List[str]], val_data: Optional[Union[str, List[str]]] = None):
        """Main training loop"""
        logger.info("🚀 Starting SloughGPT training...")
        
        # Create datasets
        train_dataset = TextDataset(train_data, block_size=512, vocab_size=self.model.config.vocab_size)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        if val_data is not None:
            val_dataset = TextDataset(val_data, block_size=512, vocab_size=self.model.config.vocab_size)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Training loop
        for epoch in range(self.current_epoch, self.training_config.num_epochs):
            self.current_epoch = epoch
            
            logger.info(f"📚 Epoch {epoch + 1}/{self.training_config.num_epochs}")
            
            # Train one epoch
            train_loss = self.train_epoch(train_loader)
            
            # Evaluate
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                logger.info(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint("best_model")
            else:
                logger.info(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}")
            
            # Step scheduler
            self.scheduler.step()
        
        # Final checkpoint
        self.save_checkpoint("final_model")
        logger.info("✅ Training completed!")
        
        return self.training_stats
    
    def fine_tune(self, 
                  data: Union[str, List[str]], 
                  learning_rate: float = 1e-5,
                  num_epochs: int = 5):
        """Fine-tune the model on specific data"""
        logger.info("🔧 Starting fine-tuning...")
        
        # Create temporary training config for fine-tuning
        fine_tune_config = TrainingConfig(
            learning_rate=learning_rate,
            batch_size=8,  # Smaller batch size for fine-tuning
            num_epochs=num_epochs,
            save_interval=500,
            eval_interval=100,
            checkpoint_dir="./fine_tune_checkpoints"
        )
        
        # Temporarily override training config
        original_config = self.training_config
        self.training_config = fine_tune_config
        
        # Re-setup optimizer with lower learning rate
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        try:
            # Fine-tune on the data
            stats = self.train(data)
            logger.info("✅ Fine-tuning completed!")
            return stats
        finally:
            # Restore original config
            self.training_config = original_config

def create_trainer(model_config: Optional[ModelConfig] = None,
                  training_config: Optional[TrainingConfig] = None) -> SloughGPTTrainer:
    """Factory function to create a trainer"""
    # Create model
    if model_config is None:
        model_config = ModelConfig()
    
    model = SloughGPT(model_config)
    
    # Create training config
    if training_config is None:
        training_config = TrainingConfig()
    
    # Create and return trainer
    return SloughGPTTrainer(model, training_config)

# Example usage
if __name__ == "__main__":
    # Create a simple example dataset
    sample_data = [
        "Hello world! This is a sample text for training.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating and powerful.",
        "Natural language processing helps computers understand text.",
        "SloughGPT is a custom language model."
    ] * 100  # Repeat for more data
    
    # Create trainer
    trainer = create_trainer()
    
    # Start training
    training_stats = trainer.train(sample_data)
    
    print("Training completed!")
    print(f"Final training loss: {training_stats['losses'][-1]:.4f}")
    print(f"Total steps: {len(training_stats['losses'])}")