"""Distributed training system for SloughGPT."""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import asyncio
import time
import json
import os
from datetime import datetime
from pathlib import Path
import logging

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class TrainingConfig:
    model_name: str
    model_path: str
    dataset_path: str
    output_dir: str
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    max_length: int = 512
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    dataloader_num_workers: int = 2
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


@dataclass
class DistributedConfig:
    backend: str = "nccl"
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"


@dataclass
class TrainingJob:
    job_id: str
    config: TrainingConfig
    distributed_config: DistributedConfig
    status: str = "pending"
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: Dict[str, Any] = None
    checkpoints: List[str] = None
    final_model_path: Optional[str] = None


class DistributedTrainer:
    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = {}
        self.active_processes: Dict[str, Any] = {}
        self.job_counter = 0
        
    def create_training_job(self, config: TrainingConfig, 
                          distributed_config: Optional[DistributedConfig] = None) -> str:
        """Create a new training job."""
        self.job_counter += 1
        job_id = f"train_{self.job_counter}_{int(time.time())}"
        
        if distributed_config is None:
            distributed_config = DistributedConfig()
        
        job = TrainingJob(
            job_id=job_id,
            config=config,
            distributed_config=distributed_config,
            created_at=datetime.now(),
            metrics={},
            checkpoints=[]
        )
        
        self.jobs[job_id] = job
        return job_id
    
    async def start_training_job(self, job_id: str) -> bool:
        """Start a training job."""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            logging.error("PyTorch or Transformers not available")
            job.status = "failed"
            return False
        
        try:
            job.status = "starting"
            job.started_at = datetime.now()
            
            # Set up environment variables for distributed training
            env = os.environ.copy()
            env.update({
                "WORLD_SIZE": str(job.distributed_config.world_size),
                "RANK": str(job.distributed_config.rank),
                "LOCAL_RANK": str(job.distributed_config.local_rank),
                "MASTER_ADDR": job.distributed_config.master_addr,
                "MASTER_PORT": job.distributed_config.master_port
            })
            
            # Start training process
            if job.distributed_config.world_size == 1:
                # Single GPU/CPU training
                process = await asyncio.create_subprocess_exec(
                    "python3", "-m", "torch.distributed.launch",
                    f"--nproc_per_node={job.distributed_config.world_size}",
                    f"--master_port={job.distributed_config.master_port}",
                    __file__, "--train", job_id,
                    env=env
                )
            else:
                # Multi-GPU training
                process = await asyncio.create_subprocess_exec(
                    "python3", "-m", "torch.distributed.launch",
                    f"--nproc_per_node={job.distributed_config.world_size}",
                    f"--master_addr={job.distributed_config.master_addr}",
                    f"--master_port={job.distributed_config.master_port}",
                    __file__, "--train", job_id,
                    env=env
                )
            
            self.active_processes[job_id] = process
            job.status = "running"
            
            # Monitor the process
            asyncio.create_task(self._monitor_training_job(job_id))
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to start training job {job_id}: {e}")
            job.status = "failed"
            return False
    
    async def _monitor_training_job(self, job_id: str):
        """Monitor a training job and update status."""
        if job_id not in self.active_processes:
            return
        
        process = self.active_processes[job_id]
        job = self.jobs[job_id]
        
        # Wait for process to complete
        return_code = await process.wait()
        
        if return_code == 0:
            job.status = "completed"
            job.completed_at = datetime.now()
        else:
            job.status = "failed"
            job.completed_at = datetime.now()
        
        # Clean up
        if job_id in self.active_processes:
            del self.active_processes[job_id]
    
    async def stop_training_job(self, job_id: str) -> bool:
        """Stop a running training job."""
        if job_id not in self.active_processes:
            return False
        
        process = self.active_processes[job_id]
        job = self.jobs[job_id]
        
        try:
            process.terminate()
            await process.wait()
            job.status = "stopped"
            job.completed_at = datetime.now()
            
            if job_id in self.active_processes:
                del self.active_processes[job_id]
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to stop training job {job_id}: {e}")
            return False
    
    def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get the status of a training job."""
        return self.jobs.get(job_id)
    
    def list_jobs(self, status_filter: Optional[str] = None) -> List[TrainingJob]:
        """List all training jobs, optionally filtered by status."""
        jobs = list(self.jobs.values())
        
        if status_filter:
            jobs = [job for job in jobs if job.status == status_filter]
        
        return jobs
    
    def get_job_metrics(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get training metrics for a job."""
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        
        # Load metrics from output directory if available
        metrics_path = Path(job.config.output_dir) / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    job.metrics.update(metrics)
            except Exception as e:
                logging.warning(f"Failed to load metrics for job {job_id}: {e}")
        
        return job.metrics


class SimpleTrainer:
    """Simple trainer for single GPU/CPU training."""
    
    @staticmethod
    def setup_distributed(rank: int, world_size: int):
        """Set up distributed training."""
        if not HAS_TORCH:
            return False
            
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=rank,
            world_size=world_size
        )
        return True
    
    @staticmethod
    def cleanup_distributed():
        """Clean up distributed training."""
        if HAS_TORCH and dist.is_initialized():
            dist.destroy_process_group()
    
    @staticmethod
    def train_model(config: TrainingConfig, rank: int = 0, world_size: int = 1) -> Dict[str, Any]:
        """Train a model with the given configuration."""
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            return {"error": "PyTorch or Transformers not available"}
        
        try:
            # Set up distributed training
            if world_size > 1:
                SimpleTrainer.setup_distributed(rank, world_size)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype=torch.float16 if config.fp16 else torch.float32
            )
            
            # Wrap model for distributed training
            if world_size > 1:
                model = DDP(model)
            
            # Load dataset (simplified)
            # In practice, you would load your actual dataset here
            train_dataset = SimpleTrainer._create_dummy_dataset(tokenizer, config.max_length, 1000)
            eval_dataset = SimpleTrainer._create_dummy_dataset(tokenizer, config.max_length, 100)
            
            # Set up data loader
            train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
            eval_sampler = DistributedSampler(eval_dataset) if world_size > 1 else None
            
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                sampler=train_sampler,
                num_workers=config.dataloader_num_workers
            )
            
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=config.batch_size,
                sampler=eval_sampler,
                num_workers=config.dataloader_num_workers
            )
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=config.output_dir,
                num_train_epochs=config.num_epochs,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                learning_rate=config.learning_rate,
                warmup_steps=config.warmup_steps,
                logging_steps=config.logging_steps,
                save_steps=config.save_steps,
                eval_steps=config.eval_steps,
                evaluation_strategy="steps",
                save_total_limit=config.save_total_limit,
                load_best_model_at_end=config.load_best_model_at_end,
                metric_for_best_model=config.metric_for_best_model,
                greater_is_better=config.greater_is_better,
                fp16=config.fp16,
                dataloader_num_workers=config.dataloader_num_workers,
                local_rank=rank,
                deepspeed=None,  # Could add DeepSpeed integration
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
            )
            
            # Train model
            train_result = trainer.train()
            eval_result = trainer.evaluate()
            
            # Save model
            trainer.save_model()
            
            # Clean up distributed training
            if world_size > 1:
                SimpleTrainer.cleanup_distributed()
            
            return {
                "train_loss": train_result.training_loss,
                "eval_loss": eval_result["eval_loss"],
                "train_steps": train_result.global_step,
                "model_saved": True,
                "output_dir": config.output_dir
            }
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def _create_dummy_dataset(tokenizer, max_length: int, num_samples: int):
        """Create a dummy dataset for testing."""
        import torch
        from torch.utils.data import Dataset
        
        class DummyDataset(Dataset):
            def __init__(self, tokenizer, max_length, num_samples):
                self.tokenizer = tokenizer
                self.max_length = max_length
                self.num_samples = num_samples
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                # Create dummy text
                text = f"This is a sample training text number {idx}. " * 10
                
                # Tokenize
                encodings = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encodings['input_ids'].flatten(),
                    'attention_mask': encodings['attention_mask'].flatten(),
                    'labels': encodings['input_ids'].flatten()
                }
        
        return DummyDataset(tokenizer, max_length, num_samples)


class TrainingManager:
    """High-level training manager."""
    
    def __init__(self):
        self.trainer = DistributedTrainer()
        self.training_history: List[Dict[str, Any]] = []
    
    async def create_and_start_training(self, model_name: str, dataset_path: str, 
                                     output_dir: str, **kwargs) -> str:
        """Create and start a training job."""
        config = TrainingConfig(
            model_name=model_name,
            model_path=model_name,
            dataset_path=dataset_path,
            output_dir=output_dir,
            **kwargs
        )
        
        job_id = self.trainer.create_training_job(config)
        success = await self.trainer.start_training_job(job_id)
        
        if success:
            self.training_history.append({
                "job_id": job_id,
                "model_name": model_name,
                "dataset_path": dataset_path,
                "output_dir": output_dir,
                "started_at": datetime.now().isoformat(),
                "status": "started"
            })
        
        return job_id
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        jobs = self.trainer.list_jobs()
        
        status_counts = {}
        for job in jobs:
            status_counts[job.status] = status_counts.get(job.status, 0) + 1
        
        return {
            "total_jobs": len(jobs),
            "status_counts": status_counts,
            "active_jobs": len(self.trainer.active_processes),
            "recent_jobs": [
                {
                    "job_id": job.job_id,
                    "model_name": job.config.model_name,
                    "status": job.status,
                    "created_at": job.created_at.isoformat() if job.created_at else None
                }
                for job in jobs[-10:]  # Last 10 jobs
            ]
        }


# Global training manager
training_manager = TrainingManager()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2 and sys.argv[1] == "--train":
        # Running as subprocess for distributed training
        job_id = sys.argv[2]
        
        # Load job configuration
        # This is simplified - in practice, you'd load from a file or database
        config = TrainingConfig(
            model_name="microsoft/DialoGPT-small",
            model_path="microsoft/DialoGPT-small",
            dataset_path="./data",
            output_dir="./output",
            num_epochs=1,
            batch_size=2,
            learning_rate=5e-5
        )
        
        # Get rank and world size from environment
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Train model
        result = SimpleTrainer.train_model(config, rank, world_size)
        
        print(f"Training completed: {result}")