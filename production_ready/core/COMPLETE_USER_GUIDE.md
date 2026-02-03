# SloGPT Dataset Standardization System
## Complete User Guide & Documentation

### 🚀 Quick Start (5 Minutes to Training)

#### **1. Create Your First Dataset**
```bash
# Method 1: Direct text
python3 create_dataset_fixed.py mydata "Your training text goes here"

# Method 2: From file
python3 create_dataset_fixed.py mydata --file your_text_file.txt

# Method 3: From folder (all files, recursive)
python3 create_dataset_fixed.py mydata --folder ./your_data_folder
```

#### **2. Train Your Model**
```bash
# Auto-detect and train
python3 train_simple.py

# Train on specific dataset
python3 train_simple.py mydata

# Fine-tune from GPT-2
python3 train_simple.py --from gpt2

# Multi-dataset training
python3 train_simple.py --mixed '{"web": 0.7, "code": 0.3}'
```

### 📊 Dataset Management

#### **List Available Datasets**
```bash
python3 train_simple.py --list
```

#### **Advanced Dataset Creation**
```bash
# From multiple file types
python3 universal_prepare.py --name mixed_data --source ./data_folder --recursive

# Large files (streaming)
python3 universal_prepare.py --name huge_data --source large_file.txt --streaming

# Batch from configuration
python3 universal_prepare.py --config datasets.yaml
```

#### **Dataset Registry & Mixing**
```bash
# Register dataset manually
python3 dataset_manager.py register --name custom --path ./my_dataset

# Discover all datasets
python3 dataset_manager.py discover

# Create mixed dataset
python3 dataset_manager.py mix --ratios web:0.7,code:0.3 --output mixed

# Generate training config
python3 dataset_manager.py generate-config --ratios web:0.7,code:0.3
```

### 🎯 Advanced Usage

#### **Configuration Files**
Create `datasets.yaml` for batch operations:
```yaml
datasets:
  - name: "web_content"
    sources: ["data/articles/", "data/blogs/"]
    recursive: true
    weight: 0.5
  - name: "code_data" 
    sources: ["repositories/python/"]
    recursive: true
    weight: 0.5

training:
  batch_size: 32
  learning_rate: 3e-4
  max_iters: 10000
```

#### **Mixed Dataset Strategies**
```bash
# Method 1: On-the-fly mixing
python3 train.py dataset=multi datasets='{"web": 0.7, "code": 0.3}'

# Method 2: Pre-mixed dataset
python3 dataset_manager.py mix --ratios web:0.7,code:0.3 --output webcode_mix
python3 train_simple.py webcode_mix

# Method 3: Configuration file
python3 dataset_manager.py mix --config datasets.yaml
python3 train_simple.py --mixed_from_config datasets.yaml
```

### 🔧 File Type Support

#### **Supported Formats**
- **Text**: `.txt`, `.md`, `.rst`, etc.
- **Code**: `.py`, `.js`, `.java`, `.cpp`, `.go`, etc.
- **Structured**: `.json`, `.csv`, `.xml`, `.yaml`
- **Mixed**: Any combination of above

#### **Special Handling**
- **Unicode**: Full international character support
- **Large Files**: Streaming mode for GB+ datasets
- **Nested Folders**: Recursive processing with `--recursive`
- **Binary Files**: Text extraction from PDFs, docs (if text extractable)

### 📈 Performance Optimization

#### **Memory Management**
```bash
# For large datasets (>1GB)
python3 universal_prepare.py --name bigdata --source huge_file.txt --streaming

# For mixed content
python3 train_simple.py --optimize_memory

# Automatic optimization
python3 train_simple.py --auto_optimize
```

#### **Device Selection**
```bash
# Auto-detect best device
python3 train_simple.py --auto_device

# Force specific device
python3 train_simple.py --device cuda  # NVIDIA
python3 train_simple.py --device mps   # Apple Silicon
python3 train_simple.py --device cpu   # CPU fallback
```

### 🛠️ Troubleshooting

#### **Common Issues**

**Issue**: "Dataset not found"
```bash
# Solution: List available datasets
python3 train_simple.py --list
```

**Issue**: "Permission denied"
```bash
# Solution: Check file permissions
ls -la your_data_folder/
chmod -R 644 your_data_folder/
```

**Issue**: "Memory error during preparation"
```bash
# Solution: Use streaming mode
python3 universal_prepare.py --name large_data --source huge_file.txt --streaming
```

**Issue**: "Unicode/encoding errors"
```bash
# Solution: Specify encoding
python3 create_dataset_fixed.py mydata --file your_file.txt --encoding utf-8
```

#### **Debug Mode**
```bash
# Enable verbose output
python3 train_simple.py mydata --verbose

# Dry run (no actual training)
python3 train_simple.py mydata --dry_run
```

### 🎨 Advanced Features

#### **Dataset Validation**
```bash
# Validate dataset integrity
python3 dataset_manager.py validate --name mydata

# Check dataset statistics
python3 dataset_manager.py stats --name mydata

# Compare datasets
python3 dataset_manager.py compare --dataset1 mydata --dataset2 other
```

#### **Versioning**
```bash
# Create dataset version
python3 dataset_manager.py version --name mydata --version v1.0

# List versions
python3 dataset_manager.py list_versions --name mydata

# Restore version
python3 dataset_manager.py restore --name mydata --version v1.0
```

#### **Backup & Sync**
```bash
# Backup all datasets
python3 dataset_manager.py backup --path ./backups/

# Sync to remote
python3 dataset_manager.py sync --remote s3://my-bucket/datasets/

# Export to other formats
python3 dataset_manager.py export --name mydata --format jsonl
```

### 🔄 Automation & CI/CD

#### **Batch Processing**
```bash
# Process multiple datasets
for dataset in web_data code_data docs; do
    python3 create_dataset_fixed.py $dataset --folder ./data/$dataset
done

# Train multiple models
python3 batch_train.py --config batch_config.yaml
```

#### **Workflow Integration**
```bash
# GitHub Actions example
name: Train Model
on: [push]
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Train Model
      run: |
        python3 create_dataset_fixed.py prod_data --file training_data.txt
        python3 train_simple.py prod_data
```

### 📚 API Reference

#### **create_dataset_fixed.py**
```bash
python3 create_dataset_fixed.py [name] [text] [options]
Options:
  --file PATH      Input file path
  --folder PATH    Input folder path (recursive)
  --encoding TEXT  File encoding (default: utf-8)
```

#### **train_simple.py**
```bash
python3 train_simple.py [dataset] [options]
Options:
  --from MODEL    Starting model (gpt2, resume)
  --mixed JSON    Mixed datasets configuration
  --device TYPE   Force device (cuda, mps, cpu)
  --auto_device    Auto-detect best device
  --verbose       Verbose output
  --dry_run       Dry run without training
```

#### **dataset_manager.py**
```bash
Commands:
  register --name NAME --path PATH       Register dataset
  list                                  List datasets
  discover                              Find datasets automatically
  mix --ratios JSON --output NAME      Create mixed dataset
  generate-config --ratios JSON          Generate training config
  validate --name NAME                   Validate dataset
  stats --name NAME                      Show dataset statistics
```

### 🎯 Best Practices

#### **Dataset Organization**
```
project/
├── data/
│   ├── raw/           # Original files
│   ├── processed/     # After universal_prepare.py
│   └── datasets/      # Final train.bin/val.bin
├── configs/
│   ├── training/      # Training configurations
│   └── datasets/      # Dataset definitions
└── models/
    ├── checkpoints/   # Model checkpoints
    └── final/        # Trained models
```

#### **Training Workflow**
1. **Data Collection**: Gather your source files
2. **Dataset Creation**: Use `create_dataset_fixed.py`
3. **Validation**: Check with `dataset_manager.py validate`
4. **Training**: Train with `train_simple.py`
5. **Evaluation**: Test model performance
6. **Iteration**: Refine and repeat

#### **Performance Tips**
- Use **streaming mode** for files > 1GB
- Enable **auto device detection** for best performance
- Use **mixed datasets** for varied training data
- Monitor **memory usage** with `--verbose`
- **Version** your datasets for reproducibility

### 🆘️ Support & Contributing

#### **Getting Help**
```bash
# Help commands
python3 create_dataset_fixed.py --help
python3 train_simple.py --help
python3 dataset_manager.py --help

# Debug information
python3 train_simple.py --version
python3 dataset_manager.py --system_info
```

#### **Common Issues & Solutions**
- **Memory errors**: Use `--streaming` flag
- **Slow training**: Use `--auto_device` and check GPU utilization
- **Unicode issues**: Specify `--encoding utf-8`
- **Permission errors**: Check file/folder permissions

#### **Feature Requests**
- File issues at: [GitHub Issues](link-to-repo)
- Feature requests: [Discussions](link-to-repo)
- Documentation: [Wiki](link-to-repo/wiki)

---

🎉 **You're now ready to train models on any dataset without complexity!**

The system handles all edge cases, supports multiple file formats, provides automatic optimization, and integrates seamlessly with existing SloGPT infrastructure. No more terminal gymnastics - just simple commands that work!