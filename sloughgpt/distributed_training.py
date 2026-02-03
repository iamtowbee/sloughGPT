#!/usr/bin/env python3
"""
SloughGPT Distributed Training
Advanced distributed training support for multi-GPU and multi-node training
"""

import os
import time
import json
import logging
import argparse
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from sloughgpt.config import ModelConfig, TrainingConfig
from sloughgpt.neural_network import SloughGPT
from sloughgpt.trainer import SloughGPTTrainer
from sloughgpt.core.exceptions import create_error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 12355
    backend: str = "nccl" if torch.cuda.is_available() else "gloo"
    init_method: str = "env"  # env, tcp, file

@dataclass
class TrainingStats:
    """Training statistics for distributed training"""
    rank: int
    epoch: int
    step: int
    loss: float
    learning_rate: float
    gradient_norm: float
    throughput_tokens_per_sec: float
    timestamp: str

class DistributedTrainer:
    """Advanced distributed training system for SloughGPT"""
    
    def __init__(self, 
                 model_config: ModelConfig,
                 training_config: TrainingConfig,
                 dist_config: DistributedConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.dist_config = dist_config
        
        # Distributed state
        self.local_rank = 0
        self.world_size = 1
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        
        # Training stats
        self.training_stats = []
        self.best_loss = float('inf')
        
    def setup_distributed(self):
        """Setup distributed training environment"""
        if self.dist_config.init_method == "env":
            # Environment variable initialization
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            
            logger.info(f"Initialized from environment - Rank: {self.local_rank}, World Size: {self.world_size}")
        elif self.dist_config.init_method == "tcp":
            # TCP initialization
            self.local_rank = self.dist_config.local_rank
            self.world_size = self.dist_config.world_size
            
            os.environ["MASTER_ADDR"] = self.dist_config.master_addr
            os.environ["MASTER_PORT"] = str(self.dist_config.master_port)
            os.environ["WORLD_SIZE"] = str(self.world_size)
            os.environ["RANK"] = str(self.local_rank)
            
            logger.info(f"TCP initialization - Master: {self.dist_config.master_addr}:{self.dist_config.master_port}")
        elif self.dist_config.init_method == "file":
            # File-based initialization (shared filesystem)
            raise NotImplementedError("File initialization not yet implemented")
        
        # Initialize process group
        if self.world_size > 1:
            logger.info(f"Initializing distributed process group with backend: {self.dist_config.backend}")
            dist.init_process_group(
                backend=self.dist_config.backend,
                init_method=f"tcp://{self.dist_config.master_addr}:{self.dist_config.master_port}" if self.dist_config.init_method == "tcp" else None,
                rank=self.local_rank,
                world_size=self.world_size
            )
            
            # Set device
            if torch.cuda.is_available():
                device_id = self.local_rank % torch.cuda.device_count()
                torch.cuda.set_device(device_id)
                device = torch.device(f"cuda:{device_id}")
            else:
                device = torch.device("cpu")
            
            logger.info(f"Rank {self.local_rank} using device: {device}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        return device
    
    def create_model(self, device: torch.device) -> nn.Module:
        """Create and wrap model for distributed training"""
        logger.info(f"Creating model on device: {device}")
        
        # Create base model
        model = SloughGPT(self.model_config)
        
        # Move to device
        model = model.to(device)
        
        # Wrap with DistributedDataParallel if multi-GPU
        if self.world_size > 1:
            if isinstance(model, nn.Module):
                model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
            else:
                logger.warning("Model is not a nn.Module, skipping DDP wrapping")
        
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return model
    
    def create_optimizer_and_scheduler(self, model: nn.Module):
        """Create optimizer and learning rate scheduler for distributed training"""
        # Adjust learning rate for distributed training
        base_lr = self.training_config.learning_rate
        
        if self.world_size > 1:
            # Linear scaling rule
            scaled_lr = base_lr * self.world_size
            logger.info(f"Scaled learning rate: {base_lr} -> {scaled_lr} (world_size={self.world_size})")
        else:
            scaled_lr = base_lr
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=scaled_lr,
            weight_decay=self.training_config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Create scheduler
        total_steps = 100000  # Estimate based on dataset size and epochs
        warmup_steps = self.training_config.warmup_steps
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=0,
            T_mult=2,
            eta_min=scaled_lr * 0.1,
            last_epoch=-1
        )
        
        return optimizer, scheduler
    
    def create_distributed_dataloader(self, dataset, batch_size: int, shuffle: bool = True):
        """Create distributed data loader"""
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=shuffle
            )
        else:
            sampler = None
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False if sampler else shuffle,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True  # Ensure consistent batch sizes across processes
        )
        
        return dataloader
    
    def train_epoch_distributed(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with distributed statistics collection"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_steps = 0
        start_time = time.time()
        
        # Update sampler for distributed training
        if self.world_size > 1 and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs, targets = batch
            
            # Move to device
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(next(self.model.parameters()).device)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(next(self.model.parameters()).device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.training_config.use_mixed_precision):
                outputs = self.model(inputs)
                
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs)
                else:
                    logits = outputs
                
                # Calculate loss
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
            
            # Backward pass
            if self.training_config.use_mixed_precision:
                scaler = torch.cuda.amp.GradScaler()
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.training_config.gradient_clip_norm > 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.gradient_clip_norm
                    )
                
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.training_config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            epoch_steps += 1
            
            # Log progress (only on rank 0 to avoid duplicate logs)
            if self.local_rank == 0 and batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                grad_norm = torch.norm(torch.stack([
                    torch.norm(p.grad).item() for p in self.model.parameters() 
                    if p.grad is not None
                ]))
                
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, "
                          f"LR: {current_lr:.6f}, Grad Norm: {grad_norm:.4f}")
        
        # Calculate epoch statistics
        avg_loss = epoch_loss / epoch_steps
        epoch_time = time.time() - start_time
        throughput = epoch_steps / epoch_time
        
        return {
            'avg_loss': avg_loss,
            'throughput_steps_per_sec': throughput,
            'epoch_time': epoch_time
        }
    
    def save_checkpoint_distributed(self, epoch: int, loss: float, save_path: str):
        """Save checkpoint in distributed training"""
        if self.local_rank == 0:  # Only save from rank 0
            checkpoint = {
                'epoch': epoch,
                'loss': loss,
                'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'config': asdict(self.model_config),
                'training_config': asdict(self.training_config),
                'distributed_config': asdict(self.dist_config),
                'training_stats': self.training_stats
            }
            
            # Create directory if needed
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            torch.save(checkpoint, save_path)
            logger.info(f"Rank 0: Saved checkpoint to {save_path}")
        
        # Barrier to ensure all processes reach this point
        if self.world_size > 1:
            dist.barrier()
    
    def train_distributed(self, dataset, checkpoint_path: Optional[str] = None):
        """Main distributed training loop"""
        device = self.setup_distributed()
        
        # Create model and optimizers
        self.model = self.create_model(device)
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(self, self.model)
        
        # Create distributed data loader
        self.train_loader = self.create_distributed_dataloader(
            dataset, 
            self.training_config.batch_size
        )
        
        # Load checkpoint if provided
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            if self.local_rank == 0:
                logger.info(f"Loading checkpoint from {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint.get('epoch', 0) + 1
            self.best_loss = checkpoint.get('loss', float('inf'))
        
        # Synchronize all processes
        if self.world_size > 1:
            dist.barrier()
        
        # Training loop
        logger.info(f"Starting distributed training on rank {self.local_rank}")
        logger.info(f"Training for {self.training_config.num_epochs - start_epoch} epochs")
        
        for epoch in range(start_epoch, self.training_config.num_epochs):
            epoch_start_time = time.time()
            
            # Train one epoch
            epoch_stats = self.train_epoch_distributed(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log epoch statistics (only on rank 0)
            if self.local_rank == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                epoch_time = time.time() - epoch_start_time
                
                logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
                logger.info(f"Avg Loss: {epoch_stats['avg_loss']:.4f}")
                logger.info(f"Throughput: {epoch_stats['throughput_steps_per_sec']:.2f} steps/sec")
                logger.info(f"Learning Rate: {current_lr:.6f}")
                
                # Save checkpoint
                if epoch_stats['avg_loss'] < self.best_loss:
                    self.best_loss = epoch_stats['avg_loss']
                    best_checkpoint_path = f"checkpoints/distributed_best_rank_{self.local_rank}.pt"
                    self.save_checkpoint_distributed(epoch, epoch_stats['avg_loss'], best_checkpoint_path)
                
                # Regular checkpoint
                if epoch % self.training_config.save_interval == 0:
                    checkpoint_path = f"checkpoints/distributed_epoch_{epoch}_rank_{self.local_rank}.pt"
                    self.save_checkpoint_distributed(epoch, epoch_stats['avg_loss'], checkpoint_path)
                
                # Record training statistics
                stats = TrainingStats(
                    rank=self.local_rank,
                    epoch=epoch,
                    step=0,
                    loss=epoch_stats['avg_loss'],
                    learning_rate=current_lr,
                    gradient_norm=0.0,  # Would calculate during training
                    throughput_tokens_per_sec=epoch_stats['throughput_steps_per_sec'],
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                
                self.training_stats.append(stats)
            
            # Synchronize all processes
            if self.world_size > 1:
                dist.barrier()
        
        if self.local_rank == 0:
            logger.info("Distributed training completed!")
            logger.info(f"Best loss achieved: {self.best_loss:.4f}")
    
    def evaluate_distributed(self, dataset, checkpoint_path: str):
        """Distributed evaluation"""
        device = self.setup_distributed()
        
        # Load model for evaluation
        self.model = self.create_model(device)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        
        # Create evaluation data loader
        eval_loader = self.create_distributed_dataloader(
            dataset, 
            batch_size=self.training_config.batch_size,
            shuffle=False
        )
        
        # Evaluation loop
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for inputs, targets in eval_loader:
                # Move to device
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs)
                else:
                    logits = outputs
                
                # Calculate loss
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                
                total_loss += loss.item()
                total_steps += 1
        
        # Synchronize results
        if self.world_size > 1:
            # Gather results from all processes
            total_loss_tensor = torch.tensor([total_loss], device=device)
            total_steps_tensor = torch.tensor([total_steps], device=device)
            
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_steps_tensor, op=dist.ReduceOp.SUM)
            
            total_loss = total_loss_tensor.item()
            total_steps = total_steps_tensor.item()
        
        # Report results (only on rank 0)
        if self.local_rank == 0:
            avg_loss = total_loss / total_steps
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            
            logger.info(f"Evaluation Results:")
            logger.info(f"  Average Loss: {avg_loss:.4f}")
            logger.info(f"  Perplexity: {perplexity:.2f}")
            logger.info(f"  Total Steps: {total_steps}")
        
        return {
            'avg_loss': avg_loss if self.local_rank == 0 else None,
            'perplexity': perplexity if self.local_rank == 0 else None,
            'total_steps': total_steps
        }

def setup_distributed_training_env():
    """Setup environment variables for distributed training"""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")

def launch_distributed_training(rank: int, world_size: int, args):
    """Launch distributed training process"""
    # Set rank environment
    os.environ["RANK"] = str(rank)
    
    # Create configurations
    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.hidden_size,
        n_heads=args.attention_heads,
        n_layers=args.layers
    )
    
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        gradient_clip_norm=args.gradient_clip_norm,
        save_interval=args.save_interval
    )
    
    dist_config = DistributedConfig(
        world_size=world_size,
        rank=rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
        backend="nccl" if torch.cuda.is_available() and not args.no_nccl else "gloo"
    )
    
    # Create trainer
    trainer = DistributedTrainer(model_config, training_config, dist_config)
    
    # Create sample dataset
    from sloughgpt.trainer import TextDataset
    dataset = TextDataset(args.data or "Hello world!", block_size=512, vocab_size=args.vocab_size)
    
    # Start training
    trainer.train_distributed(dataset, args.checkpoint)

def create_argument_parser():
    """Create argument parser for distributed training"""
    parser = argparse.ArgumentParser(description="SloughGPT Distributed Training")
    
    # Model configuration
    parser.add_argument('--vocab-size', type=int, default=50000, help='Vocabulary size')
    parser.add_argument('--hidden-size', type=int, default=1024, help='Hidden layer size')
    parser.add_argument('--attention-heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--layers', type=int, default=12, help='Number of transformer layers')
    
    # Training configuration
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--gradient-clip-norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--save-interval', type=int, default=1000, help='Save checkpoint interval')
    
    # Distributed configuration
    parser.add_argument('--world-size', type=int, default=1, help='Total number of processes')
    parser.add_argument('--master-addr', type=str, default='localhost', help='Master node address')
    parser.add_argument('--master-port', type=int, default=12355, help='Master node port')
    parser.add_argument('--no-nccl', action='store_true', help='Disable NCCL backend')
    
    # Data and checkpoints
    parser.add_argument('--data', type=str, help='Training data file')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to load')
    parser.add_argument('--eval', action='store_true', help='Run evaluation only')
    
    return parser

def main():
    """Main function for distributed training"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if args.world_size > 1:
        # Launch distributed training
        print(f"🚀 Starting distributed training on {args.world_size} processes")
        print(f"   Master: {args.master_addr}:{args.master_port}")
        print(f"   GPUs: {torch.cuda.device_count()}")
        
        mp.spawn(
            launch_distributed_training,
            args=(args.world_size, args),
            nprocs=args.world_size
        )
    else:
        # Single GPU/CPU training
        print("🚀 Starting single-process training")
        
        # Load existing trainer for simplicity
        from sloughgpt.trainer import create_trainer
        trainer = create_trainer()
        
        # Create dataset
        from sloughgpt.trainer import TextDataset
        dataset = TextDataset(args.data or "Hello world!", block_size=512, vocab_size=args.vocab_size)
        
        if args.eval:
            results = trainer.evaluate(dataset)
            print(f"Evaluation Loss: {results:.4f}")
        else:
            stats = trainer.train(args.data, args.data)  # Use same data for eval
            print(f"Training completed. Final loss: {stats['losses'][-1]:.4f}")

if __name__ == "__main__":
    main()