#!/usr/bin/env python3
"""
Simplified Distributed Training Support for SloGPT
Multi-GPU training capabilities for large-scale model training.
"""

import os
import json
import time
import subprocess
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from typing import Dict, List, Any, Optional


class SimpleDistributedTrainer:
    """Simplified distributed trainer for multi-GPU training."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.backend = config.get('backend', 'nccl')
        self.device = None
        
        self.setup_distributed()
    
    def setup_distributed(self):
        """Initialize distributed training."""
        if not self.config.get('use_distributed', False):
            print("🚀 Running in single-process mode")
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            return
        
        # Check if distributed training is available
        if not dist.is_available():
            print("⚠️ Distributed training not available")
            print("   Falling back to single GPU/CPU training")
            self.config['use_distributed'] = False
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            return
        
        # Initialize process group
        try:
            init_method = self.config.get('init_method', 'env://')
            world_size = self.config.get('world_size', 1)
            rank = self.config.get('rank', 0)
            
            dist.init_process_group(
                backend=self.backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank
            )
            
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get('LOCAL_RANK', self.rank))
            
            # Setup device
            if torch.cuda.is_available():
                self.device = f"cuda:{self.local_rank}"
                torch.cuda.set_device(self.local_rank)
            else:
                self.device = f"cpu"
            
            if self.rank == 0:
                print(f"🔗 Distributed setup complete")
                print(f"   World size: {self.world_size}")
                print(f"   Global rank: {self.rank}")
                print(f"   Local rank: {self.local_rank}")
                print(f"   Backend: {self.backend}")
                print(f"   Device: {self.device}")
            
        except Exception as e:
            print(f"❌ Distributed setup failed: {e}")
            print("   Falling back to single GPU/CPU training")
            self.config['use_distributed'] = False
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    def wrap_model(self, model):
        """Wrap model for distributed training."""
        if not self.config.get('use_distributed', False):
            return model
        
        if not self.config.get('wrap_model', True):
            return model
        
        if self.config.get('use_ddp', True) and dist.is_initialized():
            # Move model to correct device first
            model = model.to(self.device)
            
            # Wrap with DistributedDataParallel
            model = DDP(model, device_ids=[self.local_rank] if torch.cuda.is_available() else None)
            
            if self.rank == 0:
                print(f"🔗 Model wrapped with DDP")
                print(f"   Device IDs: {[self.local_rank] if torch.cuda.is_available() else None}")
        
        return model
    
    def save_checkpoint_distributed(self, model, optimizer, epoch: int, loss: float, additional_info: Dict = None):
        """Save checkpoint in distributed training."""
        # Only save on rank 0
        if self.rank != 0:
            return
        
        checkpoint_dir = Path(self.config.get('output_dir', 'out-distributed'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model state dict (handle DDP wrapped models)
        if hasattr(model, 'module'):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
            'world_size': self.world_size,
            'rank': self.rank,
            'local_rank': self.local_rank,
            'device': self.device
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Also save latest checkpoint
        latest_path = checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
        
        if self.rank == 0:
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            print(f"💾 Latest checkpoint: {latest_path}")
    
    def cleanup_distributed(self):
        """Clean up distributed training resources."""
        if self.config.get('use_distributed', False):
            if dist.is_initialized():
                dist.destroy_process_group()
                print("🧹 Distributed training cleanup complete")


def launch_multi_gpu_training(config: Dict) -> Dict:
    """Launch multi-GPU distributed training."""
    print("🚀 Launching multi-GPU distributed training")
    
    # Check available GPUs
    if not torch.cuda.is_available():
        return {"error": "CUDA not available for multi-GPU training"}
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        return {"error": f"Need at least 2 GPUs for multi-GPU training (found {num_gpus})"}
    
    # Create launch command
    dataset = config.get('dataset', 'default')
    gpus = config.get('num_gpus', num_gpus)
    
    cmd = [
        "python3", "-m", "torch.distributed.launch",
        f"--nproc_per_node={gpus}",
        "enhanced_distributed_training.py",
        "--dataset", dataset,
        "--distributed",
        "--epochs", str(config.get('max_epochs', 10)),
        "--batch-size", str(config.get('batch_size', 32)),
        "--embed", str(config.get('n_embed', 384)),
        "--layers", str(config.get('n_layer', 6)),
        "--heads", str(config.get('n_head', 6))
    ]
    
    print(f"📊 Multi-GPU Configuration:")
    print(f"   GPUs: {num_gpus}")
    print(f"   Processes: {gpus}")
    print(f"   Command: {' '.join(cmd)}")
    
    return {"status": "ready", "command": cmd, "config": config}


def check_distributed_availability():
    """Check if distributed training is available."""
    availability = {
        "cuda_available": torch.cuda.is_available(),
        "distributed_available": dist.is_available(),
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "nccl_available": False,
        "gloo_available": False
    }
    
    # Check backends
    if dist.is_available():
        try:
            dist.init_process_group(backend='nccl', init_method='tcp://localhost:12345', world_size=1, rank=0)
            availability["nccl_available"] = True
            dist.destroy_process_group()
        except:
            pass
        
        try:
            dist.init_process_group(backend='gloo', init_method='tcp://localhost:12345', world_size=1, rank=0)
            availability["gloo_available"] = True
            dist.destroy_process_group()
        except:
            pass
    
    return availability


def main():
    """Command line interface for distributed training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SloGPT Distributed Training")
    subparsers = parser.add_subparsers(dest='mode', help='Training modes')
    
    # Check availability
    check_parser = subparsers.add_parser('check', help='Check distributed training availability')
    
    # Multi-GPU training
    multi_parser = subparsers.add_parser('multi-gpu', help='Multi-GPU training')
    multi_parser.add_argument('--dataset', required=True, help='Dataset name')
    multi_parser.add_argument('--gpus', type=int, help='Number of GPUs to use')
    multi_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    multi_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    multi_parser.add_argument('--embed', type=int, default=384, help='Embedding dimension')
    multi_parser.add_argument('--layers', type=int, default=6, help='Number of layers')
    multi_parser.add_argument('--heads', type=int, default=6, help='Number of attention heads')
    
    args = parser.parse_args()
    
    if args.mode == 'check':
        print("🔍 Checking Distributed Training Availability")
        print("=" * 50)
        
        availability = check_distributed_availability()
        
        print(f"🎯 CUDA Available: {'✅' if availability['cuda_available'] else '❌'}")
        print(f"🔗 Distributed Available: {'✅' if availability['distributed_available'] else '❌'}")
        print(f"🖥 Number of GPUs: {availability['num_gpus']}")
        print(f"🚀 NCCL Backend: {'✅' if availability['nccl_available'] else '❌'}")
        print(f"🌐 GLOO Backend: {'✅' if availability['gloo_available'] else '❌'}")
        
        if availability['num_gpus'] >= 2:
            print(f"\n🎉 Multi-GPU training is available!")
            print(f"   Use: python3 {__file__} multi-gpu --dataset <name>")
        else:
            print(f"\n⚠️ Multi-GPU training requires at least 2 GPUs")
            print(f"   Found: {availability['num_gpus']} GPUs")
    
    elif args.mode == 'multi-gpu':
        config = {
            'dataset': args.dataset,
            'use_distributed': True,
            'use_ddp': True,
            'num_gpus': args.gpus or torch.cuda.device_count(),
            'max_epochs': args.epochs,
            'batch_size': args.batch_size,
            'n_embed': args.embed,
            'n_layer': args.layers,
            'n_head': args.heads
        }
        
        result = launch_multi_gpu_training(config)
        
        if result.get('status') == 'ready':
            print(f"\n🚀 Ready to launch multi-GPU training!")
            print(f"   Command: {' '.join(result['command'])}")
            print(f"\n💡 To start training, run the command above")
        else:
            print(f"\n❌ Failed to prepare multi-GPU training: {result.get('error')}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()