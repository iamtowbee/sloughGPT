"""
SloughGPT Distributed Training Framework
Large-scale model training with multi-GPU and multi-node support
"""

import asyncio
import json
import time
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from sloughgpt.core.logging_system import get_logger, timer
from sloughgpt.core.performance import get_performance_optimizer

class TrainingStatus(Enum):
    """Training job status"""
    INITIALIZING = "initializing"
    PREPARING_DATA = "preparing_data"
    DISTRIBUTING = "distributing"
    TRAINING = "training"
    VALIDATING = "validating"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class DistributedStrategy(Enum):
    """Distributed training strategies"""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID = "hybrid"

@dataclass
class TrainingNode:
    """Represents a training node (GPU or machine)"""
    node_id: str
    host: str
    port: int
    gpu_ids: List[int] = field(default_factory=list)
    cpu_count: int = 4
    memory_gb: float = 16.0
    rank: int = 0
    world_size: int = 1
    is_master: bool = False
    status: str = "idle"
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary"""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "gpu_ids": self.gpu_ids,
            "cpu_count": self.cpu_count,
            "memory_gb": self.memory_gb,
            "rank": self.rank,
            "world_size": self.world_size,
            "is_master": self.is_master,
            "status": self.status,
            "metrics": self.metrics
        }

@dataclass
class TrainingJob:
    """Represents a distributed training job"""
    job_id: str
    model_config: Dict[str, Any]
    dataset_config: Dict[str, Any]
    distributed_strategy: DistributedStrategy
    nodes: List[TrainingNode] = field(default_factory=list)
    status: TrainingStatus = TrainingStatus.INITIALIZING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    current_epoch: int = 0
    total_epochs: int = 0
    best_loss: float = float('inf')
    learning_rate: float = 0.001
    metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Calculate training duration"""
        if not self.started_at:
            return 0.0
        end_time = self.completed_at or time.time()
        return end_time - self.started_at
    
    @property
    def is_active(self) -> bool:
        """Check if job is actively training"""
        return self.status in [TrainingStatus.TRAINING, TrainingStatus.VALIDATING]
    
    @property
    def progress(self) -> float:
        """Calculate training progress"""
        if self.total_epochs == 0:
            return 0.0
        return self.current_epoch / self.total_epochs

class DistributedTrainer(ABC):
    """Abstract base class for distributed trainers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"distributed_trainer_{self.__class__.__name__.lower()}")
        self.optimizer = get_performance_optimizer()
    
    @abstractmethod
    async def initialize_training(self, job: TrainingJob) -> bool:
        """Initialize distributed training environment"""
        pass
    
    @abstractmethod
    async def distribute_data(self, job: TrainingJob) -> bool:
        """Distribute training data across nodes"""
        pass
    
    @abstractmethod
    async def start_training(self, job: TrainingJob) -> bool:
        """Start distributed training"""
        pass
    
    @abstractmethod
    async def monitor_training(self, job: TrainingJob) -> Dict[str, Any]:
        """Monitor training progress and metrics"""
        pass
    
    @abstractmethod
    async def save_checkpoint(self, job: TrainingJob, epoch: int) -> str:
        """Save training checkpoint"""
        pass
    
    @abstractmethod
    async def cleanup(self, job: TrainingJob) -> bool:
        """Cleanup training resources"""
        pass

class DataParallelTrainer(DistributedTrainer):
    """Data parallel training implementation"""
    
    async def initialize_training(self, job: TrainingJob) -> bool:
        """Initialize data parallel training"""
        self.logger.info(f"Initializing data parallel training for job {job.job_id}")
        
        try:
            # Initialize distributed process group
            await self._setup_distributed_process_group(job)
            
            # Setup model on each GPU
            await self._setup_model_on_nodes(job)
            
            # Initialize optimizers
            await self._setup_optimizers(job)
            
            self.logger.info("Data parallel training initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data parallel training: {e}")
            return False
    
    async def _setup_distributed_process_group(self, job: TrainingJob) -> None:
        """Setup distributed process group for communication"""
        self.logger.info("Setting up distributed process group")
        
        # In a real implementation, this would:
        # 1. Initialize NCCL for GPU communication
        # 2. Setup process groups for all-to-all communication
        # 3. Configure communication backends
        
        await asyncio.sleep(0.1)  # Simulate setup time
        
        # Mock process group setup
        for node in job.nodes:
            node.status = "process_group_ready"
            node.metrics["process_group_id"] = f"pg_{job.job_id}"
    
    async def _setup_model_on_nodes(self, job: TrainingJob) -> None:
        """Setup model on each training node"""
        self.logger.info("Setting up model on training nodes")
        
        # Mock model setup
        for node in job.nodes:
            await asyncio.sleep(0.05)  # Simulate model loading time
            
            node.status = "model_ready"
            node.metrics["model_parameters"] = 175_000_000  # 175M parameters
            node.metrics["memory_usage"] = 8.5  # GB
            
            self.logger.debug(f"Node {node.node_id} model setup complete")
    
    async def _setup_optimizers(self, job: TrainingJob) -> None:
        """Setup optimizers for training"""
        self.logger.info("Setting up optimizers")
        
        # Mock optimizer setup
        for node in job.nodes:
            await asyncio.sleep(0.02)
            
            node.metrics["optimizer"] = "AdamW"
            node.metrics["learning_rate"] = job.learning_rate
            node.metrics["gradient_clip"] = 1.0
    
    async def distribute_data(self, job: TrainingJob) -> bool:
        """Distribute data across nodes"""
        self.logger.info(f"Distributing data across {len(job.nodes)} nodes")
        
        job.status = TrainingStatus.PREPARING_DATA
        
        try:
            # Calculate data splits
            total_samples = job.dataset_config.get("total_samples", 1000000)
            samples_per_node = total_samples // len(job.nodes)
            
            # Distribute data shards
            for i, node in enumerate(job.nodes):
                await asyncio.sleep(0.03)  # Simulate data distribution
                
                node.metrics["data_shard_id"] = i
                node.metrics["sample_count"] = samples_per_node
                node.metrics["data_path"] = f"/data/shard_{i}_{job.job_id}"
                
                self.logger.debug(f"Data shard {i} assigned to node {node.node_id}")
            
            self.logger.info(f"Data distributed successfully: {samples_per_node} samples per node")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to distribute data: {e}")
            return False
    
    async def start_training(self, job: TrainingJob) -> bool:
        """Start distributed training"""
        self.logger.info("Starting distributed training")
        
        job.status = TrainingStatus.TRAINING
        job.started_at = time.time()
        
        try:
            # Start training on all nodes
            training_tasks = []
            for node in job.nodes:
                task = self._train_node(node, job)
                training_tasks.append(task)
            
            # Wait for all nodes to complete training
            results = await asyncio.gather(*training_tasks, return_exceptions=True)
            
            # Check if all nodes completed successfully
            successful_nodes = sum(1 for result in results if result is True)
            
            if successful_nodes == len(job.nodes):
                job.status = TrainingStatus.COMPLETED
                job.completed_at = time.time()
                self.logger.info("Distributed training completed successfully")
                return True
            else:
                job.status = TrainingStatus.FAILED
                self.logger.error(f"Training failed on {len(job.nodes) - successful_nodes} nodes")
                return False
                
        except Exception as e:
            job.status = TrainingStatus.FAILED
            self.logger.error(f"Training failed: {e}")
            return False
    
    async def _train_node(self, node: TrainingNode, job: TrainingJob) -> bool:
        """Train on a single node"""
        self.logger.info(f"Starting training on node {node.node_id}")
        
        node.status = "training"
        
        try:
            # Simulate training epochs
            for epoch in range(job.current_epoch, job.total_epochs):
                job.current_epoch = epoch + 1
                
                # Simulate epoch training time
                epoch_time = 30.0 + (len(job.nodes) * 5.0)  # More nodes = slower sync
                await asyncio.sleep(epoch_time / 100.0)  # Scale down for demo
                
                # Mock training metrics
                train_loss = 2.5 * (0.95 ** epoch) + 0.1 * (len(job.nodes) - 1)
                val_loss = train_loss + 0.2
                
                # Update node metrics
                node.metrics["current_epoch"] = epoch + 1
                node.metrics["train_loss"] = train_loss
                node.metrics["val_loss"] = val_loss
                node.metrics["throughput"] = 1000.0 / len(job.nodes)  # Samples/second
                
                # Update job metrics
                if train_loss < job.best_loss:
                    job.best_loss = train_loss
                    # Save checkpoint
                    checkpoint_path = await self.save_checkpoint(job, epoch)
                    job.checkpoints.append(checkpoint_path)
                
                # Log progress
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch + 1}/{job.total_epochs}: "
                                   f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Check if we should stop early
                if val_loss < 0.1:
                    self.logger.info("Early stopping triggered")
                    break
            
            node.status = "completed"
            return True
            
        except Exception as e:
            node.status = "failed"
            node.metrics["error"] = str(e)
            self.logger.error(f"Training on node {node.node_id} failed: {e}")
            return False
    
    async def monitor_training(self, job: TrainingJob) -> Dict[str, Any]:
        """Monitor training progress"""
        if not job.is_active:
            return {"status": job.status.value}
        
        # Aggregate metrics from all nodes
        total_samples = sum(node.metrics.get("sample_count", 0) for node in job.nodes)
        avg_train_loss = sum(node.metrics.get("train_loss", 0) for node in job.nodes) / len(job.nodes)
        avg_val_loss = sum(node.metrics.get("val_loss", 0) for node in job.nodes) / len(job.nodes)
        total_throughput = sum(node.metrics.get("throughput", 0) for node in job.nodes)
        
        # Calculate resource utilization
        total_memory_usage = sum(node.metrics.get("memory_usage", 0) for node in job.nodes)
        avg_memory_usage = total_memory_usage / len(job.nodes) if job.nodes else 0
        
        monitoring_data = {
            "job_id": job.job_id,
            "status": job.status.value,
            "progress": job.progress,
            "current_epoch": job.current_epoch,
            "total_epochs": job.total_epochs,
            "duration": job.duration,
            "metrics": {
                "total_samples": total_samples,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
                "best_loss": job.best_loss,
                "total_throughput": total_throughput,
                "avg_throughput": total_throughput / len(job.nodes),
                "memory_utilization": avg_memory_usage,
                "active_nodes": sum(1 for node in job.nodes if node.status == "training"),
                "failed_nodes": sum(1 for node in job.nodes if node.status == "failed")
            },
            "node_details": [node.to_dict() for node in job.nodes]
        }
        
        return monitoring_data
    
    async def save_checkpoint(self, job: TrainingJob, epoch: int) -> str:
        """Save training checkpoint"""
        checkpoint_path = f"/checkpoints/{job.job_id}_epoch_{epoch}.pt"
        
        # Mock checkpoint saving
        await asyncio.sleep(0.5)  # Simulate save time
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    async def cleanup(self, job: TrainingJob) -> bool:
        """Cleanup training resources"""
        self.logger.info("Cleaning up training resources")
        
        try:
            # Stop training on all nodes
            for node in job.nodes:
                node.status = "cleaning"
                await asyncio.sleep(0.1)  # Simulate cleanup time
                node.status = "idle"
                
                # Clear node metrics
                node.metrics.clear()
            
            self.logger.info("Training resources cleaned up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return False

class ModelParallelTrainer(DistributedTrainer):
    """Model parallel training implementation"""
    
    async def initialize_training(self, job: TrainingJob) -> bool:
        """Initialize model parallel training"""
        self.logger.info(f"Initializing model parallel training for job {job.job_id}")
        
        # Mock model parallel setup
        await asyncio.sleep(1.0)  # Simulate complex setup
        
        # Split model across nodes
        total_layers = job.model_config.get("num_layers", 24)
        layers_per_node = total_layers // len(job.nodes)
        
        for i, node in enumerate(job.nodes):
            start_layer = i * layers_per_node
            end_layer = (i + 1) * layers_per_node if i < len(job.nodes) - 1 else total_layers
            
            node.metrics["model_layers"] = list(range(start_layer, end_layer))
            node.status = "model_partitioned"
        
        self.logger.info("Model parallel training initialized successfully")
        return True
    
    async def distribute_data(self, job: TrainingJob) -> bool:
        """For model parallel, data is typically replicated"""
        self.logger.info("Setting up data replication for model parallel training")
        
        # Replicate full dataset on each node
        for node in job.nodes:
            node.metrics["data_strategy"] = "replicated"
            node.metrics["sample_count"] = job.dataset_config.get("total_samples", 1000000)
            await asyncio.sleep(0.05)
        
        return True
    
    async def start_training(self, job: TrainingJob) -> bool:
        """Start model parallel training"""
        # Implementation similar to data parallel but with model-specific considerations
        return await super().start_training(job)
    
    async def monitor_training(self, job: TrainingJob) -> Dict[str, Any]:
        """Monitor model parallel training"""
        monitoring_data = await super().monitor_training(job)
        
        # Add model parallel specific metrics
        communication_overhead = len(job.nodes) * 0.1  # Simulate communication cost
        monitoring_data["metrics"]["communication_overhead"] = communication_overhead
        
        return monitoring_data
    
    async def save_checkpoint(self, job: TrainingJob, epoch: int) -> str:
        """Save model parallel checkpoint"""
        checkpoint_path = f"/checkpoints/model_parallel_{job.job_id}_epoch_{epoch}.pt"
        await asyncio.sleep(0.8)  # Model parallel checkpoints are larger
        return checkpoint_path
    
    async def cleanup(self, job: TrainingJob) -> bool:
        """Cleanup model parallel resources"""
        return await super().cleanup(job)

class DistributedTrainingManager:
    """Manages distributed training jobs and resources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("distributed_training_manager")
        self.optimizer = get_performance_optimizer()
        
        # Job tracking
        self.active_jobs: Dict[str, TrainingJob] = {}
        self.completed_jobs: Dict[str, TrainingJob] = {}
        
        # Node pool
        self.available_nodes: Dict[str, TrainingNode] = {}
        self.occupied_nodes: Dict[str, str] = {}  # node_id -> job_id
        
        # Initialize trainers
        self.trainers = {
            DistributedStrategy.DATA_PARALLEL: DataParallelTrainer(config),
            DistributedStrategy.MODEL_PARALLEL: ModelParallelTrainer(config),
            # Additional strategies can be added
        }
        
        # Performance tracking
        self._performance_stats = {
            "total_jobs": 0,
            "successful_jobs": 0,
            "failed_jobs": 0,
            "total_training_time": 0.0,
            "avg_throughput": 0.0
        }
    
    async def register_node(self, node: TrainingNode) -> bool:
        """Register a training node"""
        self.logger.info(f"Registering training node: {node.node_id}")
        
        if node.node_id in self.available_nodes:
            self.logger.warning(f"Node {node.node_id} already registered")
            return False
        
        self.available_nodes[node.node_id] = node
        self.logger.info(f"Node {node.node_id} registered successfully")
        return True
    
    async def submit_job(self, job_config: Dict[str, Any]) -> str:
        """Submit a distributed training job"""
        self.logger.info("Submitting distributed training job")
        
        # Generate job ID
        job_id = f"job_{int(time.time() * 1000)}"
        
        # Create training job
        job = TrainingJob(
            job_id=job_id,
            model_config=job_config.get("model", {}),
            dataset_config=job_config.get("dataset", {}),
            distributed_strategy=DistributedStrategy(job_config.get("strategy", "data_parallel")),
            total_epochs=job_config.get("epochs", 100)
        )
        
        try:
            # Allocate nodes
            allocated_nodes = await self._allocate_nodes(job)
            if not allocated_nodes:
                raise Exception("No available nodes for training")
            
            job.nodes = allocated_nodes
            
            # Add to active jobs
            self.active_jobs[job_id] = job
            
            # Update performance stats
            self._performance_stats["total_jobs"] += 1
            
            self.logger.info(f"Job {job_id} submitted successfully with {len(allocated_nodes)} nodes")
            return job_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit job: {e}")
            raise
    
    async def _allocate_nodes(self, job: TrainingJob) -> List[TrainingNode]:
        """Allocate nodes for training job"""
        requested_nodes = job.model_config.get("required_nodes", 2)
        available_count = len(self.available_nodes) - len(self.occupied_nodes)
        
        if available_count < requested_nodes:
            self.logger.warning(f"Insufficient nodes: requested={requested_nodes}, available={available_count}")
            return []
        
        # Allocate best available nodes
        allocated_nodes = []
        for node_id, node in self.available_nodes.items():
            if node_id not in self.occupied_nodes:
                allocated_nodes.append(node)
                self.occupied_nodes[node_id] = job.job_id
                
                if len(allocated_nodes) >= requested_nodes:
                    break
        
        # Set master node (first node)
        if allocated_nodes:
            allocated_nodes[0].is_master = True
            allocated_nodes[0].rank = 0
            
            # Set ranks for other nodes
            for i, node in enumerate(allocated_nodes[1:], 1):
                node.rank = i
        
        return allocated_nodes
    
    async def start_job(self, job_id: str) -> bool:
        """Start a training job"""
        self.logger.info(f"Starting training job: {job_id}")
        
        if job_id not in self.active_jobs:
            self.logger.error(f"Job {job_id} not found")
            return False
        
        job = self.active_jobs[job_id]
        trainer = self.trainers.get(job.distributed_strategy)
        
        if not trainer:
            self.logger.error(f"No trainer for strategy: {job.distributed_strategy}")
            return False
        
        try:
            # Initialize training
            if not await trainer.initialize_training(job):
                raise Exception("Training initialization failed")
            
            # Distribute data
            if not await trainer.distribute_data(job):
                raise Exception("Data distribution failed")
            
            # Start training
            training_task = asyncio.create_task(trainer.start_training(job))
            job.metrics["training_task"] = training_task
            
            self.logger.info(f"Job {job_id} started successfully")
            return True
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            self.logger.error(f"Failed to start job {job_id}: {e}")
            return False
    
    async def monitor_job(self, job_id: str) -> Dict[str, Any]:
        """Monitor training job progress"""
        if job_id not in self.active_jobs:
            return {"error": "Job not found"}
        
        job = self.active_jobs[job_id]
        trainer = self.trainers.get(job.distributed_strategy)
        
        if trainer:
            monitoring_data = await trainer.monitor_training(job)
            monitoring_data["job_config"] = {
                "model": job.model_config,
                "dataset": job.dataset_config,
                "strategy": job.distributed_strategy.value
            }
            return monitoring_data
        
        return {"error": "No trainer available"}
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job"""
        self.logger.info(f"Cancelling training job: {job_id}")
        
        if job_id not in self.active_jobs:
            return False
        
        job = self.active_jobs[job_id]
        job.status = TrainingStatus.CANCELLED
        
        # Cancel training task
        if "training_task" in job.metrics:
            job.metrics["training_task"].cancel()
        
        # Cleanup resources
        trainer = self.trainers.get(job.distributed_strategy)
        if trainer:
            await trainer.cleanup(job)
        
        # Release nodes
        for node in job.nodes:
            if node.node_id in self.occupied_nodes:
                del self.occupied_nodes[node.node_id]
            node.status = "idle"
        
        # Move to completed jobs
        self.completed_jobs[job_id] = job
        del self.active_jobs[job_id]
        
        self.logger.info(f"Job {job_id} cancelled successfully")
        return True
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get comprehensive job status"""
        monitoring_data = await self.monitor_job(job_id)
        
        if "error" in monitoring_data:
            return monitoring_data
        
        job = self.active_jobs.get(job_id, self.completed_jobs.get(job_id))
        if not job:
            return {"error": "Job not found"}
        
        return {
            "job": job.to_dict() if hasattr(job, 'to_dict') else str(job),
            "monitoring": monitoring_data,
            "performance_stats": self.get_performance_stats()
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get distributed training performance statistics"""
        return self._performance_stats.copy()
    
    async def list_jobs(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all jobs with optional status filter"""
        all_jobs = {**self.active_jobs, **self.completed_jobs}
        
        if status_filter:
            all_jobs = {job_id: job for job_id, job in all_jobs.items() 
                        if job.status.value == status_filter}
        
        job_list = []
        for job_id, job in all_jobs.items():
            job_data = {
                "job_id": job_id,
                "status": job.status.value,
                "progress": job.progress,
                "current_epoch": job.current_epoch,
                "total_epochs": job.total_epochs,
                "duration": job.duration,
                "node_count": len(job.nodes),
                "strategy": job.distributed_strategy.value
            }
            job_list.append(job_data)
        
        return sorted(job_list, key=lambda x: x.get("duration", 0), reverse=True)
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status"""
        total_nodes = len(self.available_nodes)
        active_nodes = len(self.occupied_nodes)
        idle_nodes = total_nodes - active_nodes
        
        # Calculate cluster resources
        total_gpus = sum(len(node.gpu_ids) for node in self.available_nodes.values())
        total_memory = sum(node.memory_gb for node in self.available_nodes.values())
        total_cpus = sum(node.cpu_count for node in self.available_nodes.values())
        
        cluster_status = {
            "total_nodes": total_nodes,
            "active_nodes": active_nodes,
            "idle_nodes": idle_nodes,
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "total_gpus": total_gpus,
            "total_memory_gb": total_memory,
            "total_cpus": total_cpus,
            "utilization": {
                "node_utilization": (active_nodes / total_nodes * 100) if total_nodes > 0 else 0,
                "gpu_utilization": (sum(1 for node in self.available_nodes.values() 
                                    if node.node_id in self.occupied_nodes) * len(node.gpu_ids)) / total_gpus * 100) if total_gpus > 0 else 0
            },
            "nodes": [node.to_dict() for node in self.available_nodes.values()]
        }
        
        return cluster_status

# Global distributed training manager instance
_global_distributed_manager: Optional[DistributedTrainingManager] = None

def get_distributed_training_manager(config: Optional[Dict[str, Any]] = None) -> DistributedTrainingManager:
    """Get or create global distributed training manager"""
    global _global_distributed_manager
    if _global_distributed_manager is None:
        _global_distributed_manager = DistributedTrainingManager(config or {})
    return _global_distributed_manager

# Decorators for easy use
def distributed_training(strategy: DistributedStrategy = DistributedStrategy.DATA_PARALLEL):
    """Decorator for distributed training"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            manager = get_distributed_training_manager()
            
            job_config = kwargs.get("job_config", {})
            job_config["strategy"] = strategy.value
            
            job_id = await manager.submit_job(job_config)
            
            # Start the job
            if await manager.start_job(job_id):
                # Monitor until completion
                while True:
                    status = await manager.get_job_status(job_id)
                    
                    if status["job"]["status"] in ["completed", "failed", "cancelled"]:
                        break
                    
                    await asyncio.sleep(10)  # Check every 10 seconds
                
                return status
            else:
                raise Exception(f"Failed to start distributed training job: {job_id}")
        
        return wrapper
    return decorator