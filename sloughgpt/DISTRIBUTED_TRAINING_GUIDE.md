# SloughGPT Distributed Training Guide

## 🚀 Distributed Training Overview

SloughGPT supports advanced distributed training across multiple GPUs and multiple nodes, enabling scalable training of large models.

## 🏗️ Architecture

### Multi-GPU Training (Single Node)
```
┌─────────────────────────────────────────┐
│           Master Process (Rank 0)     │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│  │ GPU 0    │ │ GPU 1    │ │ GPU 2    │ │
│  │ SloughGPT │ │ SloughGPT │ │ SloughGPT │ │
│  └─────────┘ └─────────┘ └─────────┘ │
└─────────────────────────────────────────┘
```

### Multi-Node Training (Distributed)
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Node 1    │    │   Node 2    │    │   Node 3    │
│ ┌───────────┐│    │┌───────────┐│    │┌───────────┐│
│ │GPU 0 GPU 1││    ││GPU 2 GPU 3││    ││GPU 4 GPU 5││
│ │SloughGPT  ││    ││SloughGPT  ││    ││SloughGPT  ││
│ └───────────┘│    │└───────────┘│    │└───────────┘│
└─────────────┘    └─────────────┘    └─────────────┘
```

## 🔧 Quick Start

### Single Node Multi-GPU Training
```bash
# Basic 4-GPU training
python -m sloughgpt.distributed_training \
    --world-size 4 \
    --hidden-size 1024 \
    --attention-heads 16 \
    --layers 12 \
    --batch-size 32 \
    --epochs 20

# With larger model
python -m sloughgpt.distributed_training \
    --world-size 8 \
    --hidden-size 2048 \
    --attention-heads 32 \
    --layers 24 \
    --batch-size 16 \
    --epochs 50
```

### Multi-Node Training
```bash
# On master node (usually rank 0)
python -m sloughgpt.distributed_training \
    --world-size 8 \
    --rank 0 \
    --master-addr 192.168.1.100 \
    --master-port 29500 \
    --hidden-size 1024 \
    --batch-size 64

# On worker nodes
python -m sloughgpt.distributed_training \
    --world-size 8 \
    --rank 1 \
    --master-addr 192.168.1.100 \
    --master-port 29500 \
    --hidden-size 1024 \
    --batch-size 64

python -m sloughgpt.distributed_training \
    --world-size 8 \
    --rank 2 \
    --master-addr 192.168.1.100 \
    --master-port 29500 \
    --hidden-size 1024 \
    --batch-size 64

# ... continue for ranks 3-7
```

## ⚙️ Configuration Options

### Model Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--vocab-size` | 50000 | Vocabulary size |
| `--hidden-size` | 1024 | Hidden layer dimension |
| `--attention-heads` | 16 | Number of attention heads |
| `--layers` | 12 | Number of transformer layers |
| `--max-seq-length` | 2048 | Maximum sequence length |

### Training Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--learning-rate` | 1e-4 | Base learning rate |
| `--batch-size` | 32 | Batch size per GPU |
| `--epochs` | 20 | Number of training epochs |
| `--gradient-clip-norm` | 1.0 | Gradient clipping norm |
| `--save-interval` | 1000 | Checkpoint save interval |

### Distributed Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--world-size` | 1 | Total number of processes |
| `--rank` | 0 | Rank of current process |
| `--master-addr` | localhost | Master node address |
| `--master-port` | 29500 | Master node port |
| `--no-nccl` | False | Disable NCCL backend |

## 📊 Performance Optimization

### Memory Optimization
```bash
# Enable gradient checkpointing
python -m sloughgpt.distributed_training \
    --world-size 8 \
    --use-gradient-checkpointing \
    --batch-size 16 \
    --hidden-size 1024

# Use mixed precision
python -m sloughgpt.distributed_training \
    --world-size 8 \
    --mixed-precision \
    --batch-size 64 \
    --hidden-size 1024
```

### Communication Backend
- **NCCL**: Fastest for NVIDIA GPUs (default)
- **Gloo**: CPU and mixed GPU/CPU environments
- **MPI**: High-performance for large clusters

### Synchronization
```python
# Custom synchronization points
import torch.distributed as dist

def custom_synchronization():
    # Barrier - wait for all processes
    dist.barrier()
    
    # All-reduce - combine values from all processes
    local_loss = compute_local_loss()
    global_loss = torch.tensor([local_loss])
    dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
    
    # All-gather - collect values from all processes
    local_metrics = get_local_metrics()
    all_metrics = [torch.zeros_like(local_metrics) for _ in range(dist.get_world_size())]
    dist.all_gather(local_metrics, all_metrics)
```

## 🎯 Training Strategies

### Data Parallelism
```bash
# Basic data parallelism
python -m sloughgpt.distributed_training \
    --world-size 4 \
    --batch-size 32 \
    --hidden-size 1024

# Effective batch size = 4 * 32 = 128
```

### Pipeline Parallelism
```python
# Pipeline parallelism for very large models
from torch.distributed.pipeline.sync import Pipe

# Split model across GPUs
pipe = Pipe([
    model_part1,  # GPU 0
    model_part2,  # GPU 1
    model_part3,  # GPU 2
    model_part4,  # GPU 3
])

# Train with pipeline parallelism
output = pipe(input_batch)
```

### Tensor Parallelism
```python
# Tensor parallelism for extreme scale
import torch.distributed.tensor_parallel as tp

# Initialize tensor parallelism
tp_size = 4  # Number of GPUs for tensor parallelism
tp_rank = dist.get_rank() % tp_size

# Create tensor parallel model
model = SloughGPTWithTP(config, tp_size, tp_rank)
```

## 📈 Monitoring & Logging

### Distributed Logging
```python
# Only log from rank 0 to avoid duplicate logs
if dist.get_rank() == 0:
    logger.info("Training started")
    logger.info(f"World size: {dist.get_world_size()}")
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
```

### Performance Metrics
```python
# Track distributed training metrics
class DistributedMetrics:
    def __init__(self):
        self.all_ranks_loss = []
        self.rank_throughput = []
    
    def log_metrics(self, loss, throughput):
        if dist.get_rank() == 0:
            # Collect from all ranks
            all_losses = [loss] * dist.get_world_size()
            self.all_ranks_loss.extend(all_losses)
            self.rank_throughput.append(throughput)
            
            avg_loss = sum(all_losses) / len(all_losses)
            logger.info(f"Avg loss across all ranks: {avg_loss:.4f}")
```

## 🛠️ Troubleshooting

### Common Issues

#### Process Hang
```bash
# Check if all processes started
ps aux | grep distributed_training

# Check network connectivity
telnet master-node 29500

# Check firewall settings
sudo ufw status
```

#### Memory Issues
```bash
# Reduce batch size
--batch-size 16

# Enable gradient checkpointing
--use-gradient-checkpointing

# Use mixed precision
--mixed-precision
```

#### Communication Issues
```bash
# Use TCP instead of NCCL
--no-nccl

# Check network latency
ping master-node

# Use different backend
--backend gloo
```

### Performance Debugging
```python
# Profile distributed training
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./distributed_profiler"),
    record_shapes=True,
    with_stack=True
) as prof:
    for batch in dataloader:
        output = model(batch)
```

## 🚀 Production Deployment

### Slurm Cluster
```bash
# Submit to Slurm cluster
#!/bin/bash
#SBATCH --job-name=sloughgpt-distributed
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Export environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Launch distributed training
srun --ntasks-per-node=8 --nodes=4 \
python -m sloughgpt.distributed_training \
    --world-size 32 \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT
```

### Kubernetes
```yaml
# distributed-training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: sloughgpt-distributed
spec:
  parallelism: 4  # Number of nodes
  template:
    spec:
      containers:
      - name: sloughgpt
        image: sloughgpt:latest
        command: ["python", "-m", "sloughgpt.distributed_training"]
        env:
        - name: MASTER_ADDR
          value: "sloughgpt-master"
        - name: WORLD_SIZE
          value: "4"
        resources:
          requests:
            nvidia.com/gpu: 2
```

## 📚 Best Practices

### Model Scaling
- **Batch Size**: Use largest batch size that fits in memory
- **Learning Rate**: Scale learning rate with world size
- **Gradient Accumulation**: For effective larger batch sizes
- **Mixed Precision**: Use FP16 for faster training and memory savings

### Network Optimization
- **InfiniBand**: Use high-speed interconnect for multi-node
- **Network Topology**: Consider network topology for data placement
- **Compression**: Enable network compression for communication

### Fault Tolerance
```python
# Checkpoint with fault tolerance
def save_checkpoint_with_fault_tolerance():
    try:
        save_checkpoint()
    except Exception as e:
        logger.error(f"Checkpoint save failed: {e}")
        # Continue training
        return
    
    # Synchronize with all ranks
    if dist.get_world_size() > 1:
        dist.barrier()
```

### Mixed Precision Training
```python
# Enable automatic mixed precision
scaler = torch.cuda.amp.GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast():
        output = model(batch)
        loss = criterion(output, targets)
    
    scaler.scale(loss).backward()
    
    if args.gradient_clip_norm > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
    
    scaler.step(optimizer)
    scaler.update()
```

## 🔍 Example Workflows

### Research Training (8 GPUs)
```bash
# Setup training environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Launch distributed training
python -m sloughgpt.distributed_training \
    --world-size 8 \
    --hidden-size 2048 \
    --attention-heads 32 \
    --layers 24 \
    --batch-size 32 \
    --learning-rate 3e-4 \
    --epochs 100 \
    --mixed-precision \
    --save-interval 500
```

### Production Training (16 GPUs)
```bash
# Setup on 2 nodes with 8 GPUs each
# Node 1 (ranks 0-7)
python -m sloughgpt.distributed_training \
    --world-size 16 \
    --rank 0 \
    --master-addr node1-ip \
    --master-port 29500 \
    --hidden-size 4096 \
    --attention-heads 64 \
    --layers 32 \
    --batch-size 16 \
    --learning-rate 2e-4 \
    --epochs 50

# Node 2 (ranks 8-15)
python -m sloughgpt.distributed_training \
    --world-size 16 \
    --rank 8 \
    --master-addr node1-ip \
    --master-port 29500 \
    --hidden-size 4096 \
    --attention-heads 64 \
    --layers 32 \
    --batch-size 16 \
    --learning-rate 2e-4 \
    --epochs 50
```

---

## 🎉 Advanced Distributed Training Ready!

SloughGPT now provides enterprise-grade distributed training capabilities supporting:

✅ **Multi-GPU Training** across multiple devices  
✅ **Multi-Node Training** across multiple machines  
✅ **Advanced Optimization** with mixed precision and gradient checkpointing  
✅ **Fault Tolerance** with checkpoint recovery  
✅ **Production Deployment** for Slurm, Kubernetes, and cloud platforms  
✅ **Monitoring & Debugging** tools for distributed environments  
✅ **Flexible Configuration** for different training strategies  

**Start scaling your SloughGPT training across multiple GPUs today!** 🚀