"""
Distributed Training Support - Ported from recovered distributed_training.py
"""

import os
from typing import Dict, Any, Optional
import torch
import torch.distributed as dist


class DistributedConfig:
    """Configuration for distributed training."""
    
    def __init__(
        self,
        use_distributed: bool = False,
        backend: str = "nccl",
        world_size: int = 1,
        rank: int = 0,
        init_method: str = "env://",
    ):
        self.use_distributed = use_distributed
        self.backend = backend
        self.world_size = world_size
        self.rank = rank
        self.init_method = init_method


class DistributedTrainer:
    """Manages distributed training across multiple GPUs or nodes."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.device = None
        self._initialized = False
        
        if config.use_distributed:
            self._init_distributed()
    
    def _init_distributed(self) -> None:
        """Initialize distributed training."""
        if not dist.is_available():
            print("⚠️ Distributed training not available")
            self.config.use_distributed = False
            return
        
        try:
            world_size = self.config.world_size
            rank = self.config.rank
            
            if 'RANK' not in os.environ:
                os.environ['RANK'] = str(rank)
            if 'WORLD_SIZE' not in os.environ:
                os.environ['WORLD_SIZE'] = str(world_size)
            if 'LOCAL_RANK' not in os.environ:
                os.environ['LOCAL_RANK'] = str(rank % max(1, torch.cuda.device_count()))
            
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=world_size,
                rank=rank
            )
            
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get('LOCAL_RANK', self.rank))
            self._initialized = True
            
            print(f"✓ Distributed initialized: rank={self.rank}, world_size={self.world_size}")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize distributed: {e}")
            self.config.use_distributed = False
    
    def get_device(self) -> torch.device:
        """Get the device for this process."""
        if self.config.use_distributed and self._initialized:
            return torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0
    
    def sync(self) -> None:
        """Synchronize all processes."""
        if self.config.use_distributed and self._initialized:
            dist.barrier()
    
    def cleanup(self) -> None:
        """Clean up distributed training."""
        if self.config.use_distributed and self._initialized:
            dist.destroy_process_group()
            self._initialized = False


def is_distributed_available() -> bool:
    """Check if distributed training is available."""
    return dist.is_available()


def get_world_size() -> int:
    """Get the world size."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get the current rank."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


__all__ = ["DistributedConfig", "DistributedTrainer", "is_distributed_available", "get_world_size", "get_rank"]
