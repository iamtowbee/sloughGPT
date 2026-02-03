#!/usr/bin/env python3
"""
Enhanced Distributed Training System for SloGPT
Integrates distributed training with the dataset standardization system.
"""

import os
import json
import time
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pickle

# Import dataset system
from create_dataset_fixed import create_dataset
from simple_gpt_model import GPT, load_dataset
from simple_distributed_training import SimpleDistributedTrainer, launch_multi_gpu_training


class DistributedSloGPTTrainer:
    """Enhanced trainer with distributed capabilities."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dataset_name = config.get('dataset', 'default')
        self.distributed_trainer = None
        self.model = None
        self.optimizer = None
        self.train_data = None
        self.val_data = None
        self.meta = None
        
        # Setup distributed training
        if config.get('use_distributed', False):
            self.distributed_trainer = SimpleDistributedTrainer(config)
            self.device = self.distributed_trainer.device
            self.rank = self.distributed_trainer.rank
            self.world_size = self.distributed_trainer.world_size
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.rank = 0
            self.world_size = 1
        
        print(f"🚀 Initialized trainer on device: {self.device}")
        if self.config.get('use_distributed', False):
            print(f"   Distributed: {self.world_size} processes, rank {self.rank}")
    
    def setup_dataset(self):
        """Load and setup dataset for training."""
        print(f"📊 Loading dataset: {self.dataset_name}")
        
        try:
            self.train_data, self.val_data, self.meta = load_dataset(self.dataset_name)
            vocab_size = self.meta['vocab_size']
            
            # Adjust batch size for distributed training
            batch_size = self.config.get('batch_size', 32)
            if self.world_size > 1:
                batch_size = max(1, batch_size // self.world_size)
                print(f"   Adjusted batch size for distributed: {batch_size}")
            
            print(f"   Train tokens: {len(self.train_data):,}")
            print(f"   Val tokens: {len(self.val_data):,}")
            print(f"   Vocab size: {vocab_size}")
            print(f"   Batch size: {batch_size}")
            
            return vocab_size, batch_size
            
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return None, None
    
    def create_model(self, vocab_size: int):
        """Create and setup the model."""
        print(f"🏗️ Creating model...")
        
        # Model configuration
        n_embed = self.config.get('n_embed', 384)
        n_layer = self.config.get('n_layer', 6)
        n_head = self.config.get('n_head', 6)
        
        print(f"   Architecture: {n_layer} layers, {n_embed} embed, {n_head} heads")
        
        # Create model
        self.model = GPT(vocab_size, n_embed, n_layer, n_head)
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Wrap for distributed training
        if self.distributed_trainer:
            self.model = self.distributed_trainer.wrap_model(self.model)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def create_optimizer(self):
        """Create optimizer with support for distributed training."""
        print(f"⚙️ Creating optimizer...")
        
        learning_rate = self.config.get('learning_rate', 3e-4)
        weight_decay = self.config.get('weight_decay', 0.1)
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        print(f"   Learning rate: {learning_rate}")
        print(f"   Weight decay: {weight_decay}")
        
        # Create learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get('max_steps', 5000),
            eta_min=self.config.get('min_lr', 1e-6)
        )
        
        return self.optimizer, self.scheduler
    
    def create_data_loader(self, data: np.ndarray, batch_size: int, split: str = 'train'):
        """Create data loader with distributed support."""
        print(f"📦 Creating {split} data loader...")
        
        # Calculate block size
        block_size = self.config.get('block_size', 256)
        
        # Calculate number of batches
        num_batches = len(data) // (batch_size * block_size)
        
        if self.rank == 0:
            print(f"   {split.capitalize()} batches: {num_batches:,}")
            print(f"   Block size: {block_size}")
            print(f"   Sequence length: {block_size}")
        
        return data, num_batches, block_size
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        
        batch_size = self.config.get('batch_size', 32)
        if self.world_size > 1:
            batch_size = max(1, batch_size // self.world_size)
        
        block_size = self.config.get('block_size', 256)
        
        # Shuffle data for each epoch
        np.random.shuffle(self.train_data)
        
        total_loss = 0
        num_batches = len(self.train_data) // (batch_size * block_size)
        
        for batch_idx in range(num_batches):
            # Get batch
            start_idx = batch_idx * batch_size * block_size
            end_idx = start_idx + batch_size * block_size
            
            if end_idx > len(self.train_data):
                break
            
            batch_data = self.train_data[start_idx:end_idx]
            batch_x = torch.from_numpy(batch_data[:-1]).reshape(batch_size, -1).to(self.device)
            batch_y = torch.from_numpy(batch_data[1:]).reshape(batch_size, -1).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, loss = self.model(batch_x, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % 100 == 0 and self.rank == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"   Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        
        batch_size = self.config.get('batch_size', 32)
        if self.world_size > 1:
            batch_size = max(1, batch_size // self.world_size)
        
        block_size = self.config.get('block_size', 256)
        
        total_loss = 0
        num_batches = len(self.val_data) // (batch_size * block_size)
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size * block_size
                end_idx = start_idx + batch_size * block_size
                
                if end_idx > len(self.val_data):
                    break
                
                batch_data = self.val_data[start_idx:end_idx]
                batch_x = torch.from_numpy(batch_data[:-1]).reshape(batch_size, -1).to(self.device)
                batch_y = torch.from_numpy(batch_data[1:]).reshape(batch_size, -1).to(self.device)
                
                logits, loss = self.model(batch_x, batch_y)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self):
        """Main training loop."""
        print("🚀 Starting distributed training...")
        
        # Setup dataset and model
        vocab_size, batch_size = self.setup_dataset()
        if vocab_size is None:
            return {"error": "Failed to setup dataset"}
        
        self.create_model(vocab_size)
        self.create_optimizer()
        
        # Training configuration
        max_epochs = self.config.get('max_epochs', 10)
        eval_interval = self.config.get('eval_interval', 1)
        save_interval = self.config.get('save_interval', 5)
        
        print(f"   Max epochs: {max_epochs}")
        print(f"   Eval interval: {eval_interval}")
        print(f"   Save interval: {save_interval}")
        
        # Training loop
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(1, max_epochs + 1):
            start_time = time.time()
            
            # Train epoch
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)
            
            # Validate
            if epoch % eval_interval == 0:
                val_loss = self.validate()
                val_losses.append(val_loss)
                
                epoch_time = time.time() - start_time
                
                if self.rank == 0:
                    print(f"📊 Epoch {epoch}/{max_epochs} completed in {epoch_time:.2f}s")
                    print(f"   Train Loss: {train_loss:.4f}")
                    print(f"   Val Loss: {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_checkpoint(epoch, val_loss, is_best=True)
            
            # Save checkpoint
            if epoch % save_interval == 0:
                self.save_checkpoint(epoch, train_loss)
            
            # Update learning rate
            self.scheduler.step()
        
        # Final save
        if self.rank == 0:
            self.save_checkpoint(max_epochs, train_losses[-1], is_final=True)
            print(f"🎉 Training completed!")
            print(f"   Best validation loss: {best_val_loss:.4f}")
        
        return {
            "success": True,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss
        }
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint."""
        if self.rank != 0:
            return
        
        # Create checkpoint directory
        output_dir = Path(self.config.get('output_dir', f'checkpoints/{self.dataset_name}'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model state
        if hasattr(self.model, 'module'):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config,
            'meta': self.meta,
            'vocab_size': self.meta['vocab_size'],
            'train_losses': getattr(self, 'train_losses', []),
            'val_losses': getattr(self, 'val_losses', [])
        }
        
        # Save checkpoint
        if is_best:
            checkpoint_path = output_dir / 'best_checkpoint.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Best checkpoint saved: {checkpoint_path}")
        
        if is_final:
            checkpoint_path = output_dir / 'final_checkpoint.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Final checkpoint saved: {checkpoint_path}")
        else:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            if self.rank == 0:
                print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save model in original format for compatibility
        if is_final or is_best:
            self.save_original_format(output_dir, is_best)
    
    def save_original_format(self, output_dir: Path, is_best: bool = False):
        """Save model in original SloGPT format."""
        if hasattr(self.model, 'module'):
            model = self.model.module
        else:
            model = self.model
        
        # Extract model weights
        weights = {}
        
        # Token embeddings
        weights['token_embedding_table'] = model.token_embedding_table.weight.detach().cpu()
        
        # Position embeddings
        weights['position_embedding_table'] = model.position_embedding_table.weight.detach().cpu()
        
        # Layer weights
        for i, block in enumerate(model.blocks):
            weights[f'layer_{i}_ln_1'] = block.ln_1.weight.detach().cpu()
            weights[f'layer_{i}_attn_c_attn'] = block.attn.c_attn.weight.detach().cpu()
            weights[f'layer_{i}_attn_c_proj'] = block.attn.c_proj.weight.detach().cpu()
            weights[f'layer_{i}_ln_2'] = block.ln_2.weight.detach().cpu()
            weights[f'layer_{i}_c_fc'] = block.mlp.c_fc.weight.detach().cpu()
            weights[f'layer_{i}_c_proj'] = block.mlp.c_proj.weight.detach().cpu()
        
        # Final layer norm
        weights['final_layer_norm'] = model.ln_f.weight.detach().cpu()
        
        # Output projection
        weights['output_projection'] = model.lm_head.weight.detach().cpu()
        
        # Save in different formats
        suffix = '_best' if is_best else ''
        
        # Save as PyTorch checkpoint
        pt_path = output_dir / f'model{suffix}.pt'
        torch.save(weights, pt_path)
        
        # Save metadata
        meta_path = output_dir / f'meta{suffix}.pkl'
        with open(meta_path, 'wb') as f:
            pickle.dump(self.meta, f)
        
        print(f"💾 Original format saved: {pt_path}")
        print(f"💾 Metadata saved: {meta_path}")


def launch_distributed_slogpt_training(config: Dict) -> Dict:
    """Launch distributed SloGPT training."""
    print("🚀 Launching Distributed SloGPT Training")
    print("=" * 50)
    
    # Validate configuration
    required_keys = ['dataset', 'use_distributed']
    for key in required_keys:
        if key not in config:
            return {"error": f"Missing required config key: {key}"}
    
    # Check if dataset exists
    dataset_dir = Path(f"datasets/{config['dataset']}")
    if not dataset_dir.exists():
        print(f"❌ Dataset not found: {config['dataset']}")
        return {"error": f"Dataset not found: {config['dataset']}"}
    
    # Setup distributed training if needed
    if config.get('use_distributed', False):
        result = launch_distributed_training(config)
        if 'error' in result:
            return result
    
    # Create trainer
    trainer = DistributedSloGPTTrainer(config)
    
    # Start training
    result = trainer.train()
    
    # Cleanup
    if trainer.distributed_trainer:
        trainer.distributed_trainer.cleanup_distributed()
    
    return result


def main():
    """Command line interface for distributed SloGPT training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed SloGPT Training")
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--config', help='Configuration file')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--multi-gpu', action='store_true', help='Use multi-GPU training')
    parser.add_argument('--gpus', type=int, help='Number of GPUs to use')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--embed', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--heads', type=int, default=6, help='Number of attention heads')
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'dataset': args.dataset,
        'output_dir': args.output or f'checkpoints/{args.dataset}',
        'use_distributed': args.distributed or args.multi_gpu,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'n_embed': args.embed,
        'n_layer': args.layers,
        'n_head': args.heads,
        'max_epochs': args.epochs,
        'eval_interval': 1,
        'save_interval': 5
    }
    
    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # Setup multi-GPU if requested
    if args.multi_gpu and not args.distributed:
        if torch.cuda.is_available():
            config['use_distributed'] = True
            config['num_gpus'] = args.gpus or torch.cuda.device_count()
            print(f"🚀 Enabling multi-GPU training with {config['num_gpus']} GPUs")
    
    # Launch training
    result = launch_distributed_slogpt_training(config)
    
    if result.get('success'):
        print(f"\n🎉 Training completed successfully!")
        print(f"   Best validation loss: {result.get('best_val_loss', 'N/A')}")
    else:
        print(f"\n❌ Training failed: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())