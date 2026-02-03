# SloGPT Dataset Standardization System

A comprehensive system for standardizing and managing multiple datasets for training without writing new code for each dataset.

## 🚀 Quick Start

```bash
# 1. Prepare any dataset from any source
python universal_prepare.py --name mydataset --source data.txt

# 2. Register and manage datasets
python dataset_manager.py register --name mydataset --path datasets/mydataset
python dataset_manager.py list

# 3. Train on standardized datasets
python train.py --dataset=mydataset
```

## 📁 System Components

### 1. Universal Dataset Preparer (`universal_prepare.py`)
- **Multi-format support**: `.txt`, `.py`, `.js`, `.json`, `.csv`, `.md`, and more
- **Directory processing**: Process entire codebases with `--recursive`
- **Large file support**: Streaming mode for GB+ files
- **Batch preparation**: Use config files for multiple datasets

### 2. Dataset Manager (`dataset_manager.py`)
- **Registry system**: Central database of all datasets
- **Auto-discovery**: Find datasets automatically
- **Mixed datasets**: Combine multiple datasets with custom ratios
- **Training configs**: Generate training configurations

### 3. Configuration System (`datasets.yaml`)
- **Declarative setup**: Define datasets and mixing ratios
- **Batch operations**: Prepare multiple datasets at once
- **Training parameters**: Centralized training configuration

## 🛠️ Usage Examples

### Single Dataset Preparation
```bash
# From single file
python universal_prepare.py --name code_data --source my_project.py

# From directory (recursive)
python universal_prepare.py --name webdata --source ./web_content --recursive

# Large files (streaming)
python universal_prepare.py --name bigdata --source huge_file.json --streaming
```

### Multi-Dataset Training
```bash
# Method 1: Direct mixing ratios
python train.py dataset=multi datasets='{"webtext": 0.7, "code": 0.3}'

# Method 2: Create mixed dataset first
python dataset_manager.py mix --ratios webtext:0.7,code:0.3 --output mixed_data
python train.py --dataset=mixed_data

# Method 3: Use configuration file
python universal_prepare.py --config datasets.yaml
python dataset_manager.py mix --config datasets.yaml
python train.py config=multi_dataset_config.json
```

### Dataset Management
```bash
# List all datasets
python dataset_manager.py list

# Discover new datasets
python dataset_manager.py discover

# Register manually
python dataset_manager.py register --name custom --path ./my_dataset

# Generate training config
python dataset_manager.py generate-config --ratios code:0.6,text:0.4
```

## 📊 Dataset Standardization Format

All datasets are converted to a unified format:

```
datasets/
├── mydataset/
│   ├── train.bin      # Training data (uint16 tokens)
│   ├── val.bin        # Validation data (uint16 tokens)
│   └── meta.pkl       # Metadata (vocab, mappings, stats)
```

**Metadata includes:**
- Vocabulary size and character mappings
- Token counts (train/val)
- Source file information
- Processing parameters

## 🎯 Advanced Features

### Batch Preparation with Config
```yaml
datasets:
  - name: "web_content"
    sources:
      - "data/articles/"
      - "data/blogs/"
    recursive: true
    weight: 0.5
    
  - name: "code_repos"
    sources:
      - "repositories/python/"
      - "repositories/javascript/"
    recursive: true
    streaming: true
    weight: 0.5
```

### Mixed Dataset Creation
```bash
# Mix by ratios
python dataset_manager.py mix --ratios web_content:0.5,code_repos:0.5 --output web_code_mix

# Mix using config
python dataset_manager.py mix --config datasets.yaml --output domain_specific
```

### Training with Multiple Detsnets
```bash
# Direct multi-dataset training
python train.py dataset=multi datasets='{"web_content": 0.5, "code_repos": 0.5}'

# Use generated config
python train.py config=multi_dataset_config.json

# Mixed dataset approach
python train.py --dataset=web_code_mix
```

## 🔧 Integration with Existing Code

The standardized system integrates seamlessly with existing SloGPT:

- **Same training command**: `python train.py --dataset=<name>`
- **Same model architecture**: No changes needed
- **Same configuration system**: Uses existing config files
- **Same output format**: Compatible with all existing tools

## 📈 Benefits

1. **No Code Duplication**: One preparer handles all formats
2. **Unified Interface**: Same training command for any dataset
3. **Scalable**: Handles from KB to GB+ datasets
4. **Flexible**: Mix datasets with custom ratios
5. **Traceable**: Complete metadata and source tracking
6. **Consistent**: Standardized format across all datasets

## 🎉 Getting Started Demo

Run the quick start demo:

```bash
python quick_start_datasets.py
```

This will:
1. Create sample datasets from different sources
2. Register them in the system
3. Create a mixed dataset
4. Generate training configurations
5. Show all available training commands

You'll have a complete understanding of how to use the standardized dataset system for any type of training data!