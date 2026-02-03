# 🚀 SloughGPT CLI Documentation

## Overview

The SloughGPT CLI provides a comprehensive command-line interface for managing models, training, inference, and deployment.

## Installation

```bash
# Add to PATH (recommended)
export PATH="$PATH:/path/to/sloughgpt"

# Or use directly
python3 -m sloughgpt.cli <command>
```

## Commands

### 🔍 `list` - List Available Models
```bash
sloughgpt list
```

**Shows all available models with:**
- Model name and description
- Parameter count
- Configuration details
- Tags and metadata

### 📊 `info` - Model Information
```bash
sloughgpt info sloughgpt-medium
```

**Get detailed information about a specific model:**
- Architecture details
- Performance characteristics
- Version and license
- File size and memory requirements

### 🎯 `generate` - Text Generation
```bash
sloughgpt generate --model sloughgpt-medium \
                --text "Hello, world!" \
                --max-length 100 \
                --temperature 1.0 \
                --top-k 50
```

**Generate text using any available model:**
- Choose from available models
- Customize generation parameters
- Real-time output with performance metrics

### 🏃 `benchmark` - Performance Testing
```bash
sloughgpt benchmark --model sloughgpt-medium
sloughgpt benchmark  # Quick benchmark
```

**Compare model performance:**
- Inference speed (tokens/sec)
- Memory usage optimization
- Performance across different configurations
- Optimization impact analysis

### 🏋️‍♂️ `train` - Model Training
```bash
sloughgpt train --hidden-size 512 \
                --attention-heads 8 \
                --layers 6 \
                --epochs 10 \
                --batch-size 32 \
                --learning-rate 1e-4
```

**Train custom models with flexible configuration:**
- Customizable architecture parameters
- Training hyperparameters
- Checkpoint management
- Fine-tuning support

### 🌐 `serve` - API Server
```bash
sloughgpt serve --host 0.0.0.0 --port 8000
sloughgpt serve --dev  # Development mode
```

**Start the production API server:**
- Host and port configuration
- Development mode with auto-reload
- Performance monitoring

### 💾 `status` - System Status
```bash
sloughgpt status
```

**Get comprehensive system information:**
- Hardware and GPU details
- Model zoo statistics
- PyTorch version
- Available resources

## Configuration Files

### Model Configuration
```python
from sloughgpt.config import ModelConfig

config = ModelConfig(
    vocab_size=50000,
    d_model=1024,
    n_heads=16,
    n_layers=12,
    dropout=0.1
)
```

### Training Configuration
```python
from sloughgpt.trainer import TrainingConfig

config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=10,
    save_interval=1000,
    use_mixed_precision=True
)
```

## Integration Examples

### Python API
```python
from sloughgpt.model_zoo import get_model_zoo
from sloughgpt.neural_network import SloughGPT

# Load model
zoo = get_model_zoo()
model = zoo.load_model("sloughgpt-medium", optimize=True)

# Generate text
input_ids = torch.randint(0, model.config.vocab_size, (1, 10))
with torch.no_grad():
    output = model(input_ids)
    generated = model.generate(input_ids, max_length=50)
```

### Shell Scripting
```bash
#!/bin/bash

# Load multiple models
for model in $(sloughgpt list --format json | jq -r '.[].name'); do
    echo "Loading $model..."
    sloughgpt info $model
done

# Batch generation
for text in "Hello world" "How are you" "What is AI"; do
    sloughgpt generate --text "$text" --model sloughgpt-small
done
```

### Web Service Integration
```python
from sloughgpt.api_server import create_app
import uvicorn

app = create_app()

# Start with custom configuration
uvicorn.run(
    "web_server:app",
    host="0.0.0.0",
    port=8000,
    workers=4
)
```

## Advanced Usage

### Custom Model Architecture
```python
from sloughgpt.config import ModelConfig
from sloughgpt.neural_network import SloughGPT

# Create custom configuration
custom_config = ModelConfig(
    d_model=2048,    # Larger model
    n_heads=32,       # More attention heads
    n_layers=24,       # Deeper network
    max_seq_length=2048, # Longer sequences
    dropout=0.1        # Regularization
)

model = SloughGPT(custom_config)
```

### Performance Optimization
```python
from sloughgpt.optimizations import create_optimized_model

# Create optimized model
optimized = create_optimized_model(
    config,
    enable_quantization=True,    # 4x memory reduction
    enable_compilation=True,     # 10-20% speedup
    enable_mixed_precision=True   # 2x training speedup
)

model = optimized.get_model_for_inference()
```

### Training Pipeline Integration
```python
from sloughgpt.trainer import SloughGPTTrainer

# Create trainer with custom config
trainer = SloughGPTTrainer(model, training_config)

# Train with custom data
trainer.train("path/to/training_data.txt")

# Fine-tune on domain-specific data
trainer.fine_tune(
    data="domain_specific_data.txt",
    learning_rate=1e-5,
    num_epochs=5
)
```

## Model Zoo Management

### Adding Custom Models
```python
from sloughgpt.model_zoo import get_model_zoo

zoo = get_model_zoo()

# Add a trained model
zoo.add_model(
    name="my-custom-model",
    description="Custom trained model for domain X",
    config=my_model_config,
    model_file="path/to/model.pt",
    tags=["custom", "domain-x"],
    training_data="path/to/training_data.txt"
)
```

### Model Search and Filtering
```python
# Search by name
models = zoo.search_models(query="small")

# Filter by parameters
models = zoo.search_models(min_parameters=100_000_000)

# Filter by tags
models = zoo.search_models(tags=["quantized", "production"])
```

### Benchmarking Integration
```python
from sloughgpt.benchmark_suite import BenchmarkSuite

# Create benchmark suite
suite = BenchmarkSuite()

# Run comprehensive benchmarks
results = suite.run_full_benchmark([
    "small", "medium", "large"
])

# Generate performance report
suite.generate_summary(results)
```

## Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Check model availability
sloughgpt list

# Verify model files
ls -la ~/.sloughgpt/models/

# Check system compatibility
sloughgpt status
```

#### Performance Issues
```bash
# Run quick benchmark
sloughgpt benchmark

# Check system resources
sloughgpt status

# Optimize for available hardware
python -c "import torch; print(torch.cuda.is_available())"
```

#### Memory Issues
```bash
# Use quantized models
sloughgpt generate --model sloughgpt-small

# Reduce batch size
sloughgpt train --batch-size 8

# Enable gradient checkpointing
sloughgpt train --use-gradient-checkpointing
```

## Development

### Contributing to CLI
```bash
# Clone repository
git clone https://github.com/sloughgpt/sloughgpt.git

# Install development dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/cli/

# Run linting
python -m ruff check sloughgpt/cli/
```

### CLI Development
```python
# Create new command
from sloughgpt.cli import create_parser

def cmd_custom_command(args):
    """Custom command implementation"""
    pass

# Add to parser
parser = create_parser()
subparser = parser.add_subparsers()
custom_parser = subparsers.add_parser('custom', help='Custom command')
custom_parser.set_defaults(func=cmd_custom_command)
```

## Best Practices

### Performance
- Use optimized models for CPU deployment
- Enable quantization for memory-constrained environments
- Use GPU acceleration when available
- Monitor resource usage during training

### Model Management
- Keep models organized with consistent naming
- Document model training data and configurations
- Use version control for model updates
- Regular cleanup of unused models

### Training
- Start with smaller models for prototyping
- Use mixed precision for faster training
- Implement proper validation splits
- Monitor training metrics and adjust hyperparameters

### Deployment
- Use Docker for consistent environments
- Implement health checks and monitoring
- Use load balancing for production deployments
- Keep models and dependencies updated

## Reference

### Environment Variables
```bash
export SLUGHGPT_MODEL_PATH=/path/to/models
export SLUGHGPT_DATA_PATH=/path/to/data
export SLUGHGPT_LOG_LEVEL=INFO
export CUDA_VISIBLE_DEVICES=0
```

### Configuration Files
```bash
# Default config location: ~/.sloughgpt/config.json
# Model zoo: ~/.sloughgpt/models.json
# Checkpoints: ~/.sloughgpt/checkpoints/
# Logs: ~/.sloughgpt/logs/
```

### Exit Codes
- `0`: Success
- `1`: Command line error
- `2`: Model loading error
- `3`: Training error
- `4`: Generation error
- `5`: Configuration error

---

*For detailed help, use: `sloughgpt <command> --help`*