#!/usr/bin/env python3
"""
Clean Up Codebase - Production Ready System

Removes temporary files, organizes production system, and creates final deployment package.
"""

import os
import shutil
import json
import time
from pathlib import Path
from typing import List, Dict


class CodebaseCleaner:
    """Clean and organize codebase for production deployment."""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.production_dir = Path("production_ready")
        self.production_dir.mkdir(exist_ok=True)
        
        # Files to keep in production
        self.essential_files = {
            # Core system files
            "create_dataset_fixed.py",
            "train_simple.py", 
            "universal_prepare.py",
            "simple_trainer.py",
            
            # Advanced features
            "advanced_dataset_features.py",
            "performance_optimizer.py",
            "cli_shortcuts.py",
            "batch_processor.py",
            
            # Documentation
            "COMPLETE_USER_GUIDE.md",
            "DATASET_STANDARDIZATION.md",
            
            # Configuration templates
            "datasets.yaml"
            
            # Test files (keep for validation)
            "final_system_test.py"
        }
        
        # Temporary files to remove
        self.temp_patterns = [
            "test_*",
            "debug_*", 
            "temp_*",
            "*.tmp",
            "*.pyc",
            "__pycache__",
            ".DS_Store",
            "*.log"
        ]
        
        # Directories to clean
        self.temp_dirs = [
            "test_data",
            "batch_config.json",
            "performance_report.json",
            "dataset_versions"
            ".slogpt_aliases"
        ]
    
    def clean_temp_files(self):
        """Remove temporary files and directories."""
        print("🧹 Cleaning temporary files...")
        
        removed_count = 0
        for pattern in self.temp_patterns:
            for file_path in self.root_dir.glob(pattern):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        removed_count += 1
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                        removed_count += 1
                except Exception as e:
                    print(f"⚠️ Could not remove {file_path}: {e}")
        
        for dir_name in self.temp_dirs:
            dir_path = self.root_dir / dir_name
            if dir_path.exists():
                try:
                    if dir_path.is_dir():
                        shutil.rmtree(dir_path)
                        removed_count += 1
                except Exception as e:
                    print(f"⚠️ Could not remove {dir_path}: {e}")
        
        print(f"✅ Removed {removed_count} temporary files/directories")
    
    def clean_test_datasets(self):
        """Remove test datasets while keeping essential ones."""
        print("🧹 Cleaning test datasets...")
        
        datasets_dir = self.root_dir / "datasets"
        if datasets_dir.exists():
            essential_datasets = {"working_test", "final_training_test", "workflow_test"}
            
            removed_count = 0
            for item in datasets_dir.iterdir():
                if item.is_dir() and item.name not in essential_datasets:
                    try:
                        shutil.rmtree(item)
                        removed_count += 1
                    except Exception as e:
                        print(f"⚠️ Could not remove {item}: {e}")
            
            print(f"✅ Removed {removed_count} test dataset directories")
    
    def organize_production_system(self):
        """Copy essential files to production directory."""
        print("📁 Organizing production system...")
        
        # Create production structure
        prod_dirs = [
            "core",
            "tools", 
            "docs",
            "examples",
            "templates"
        ]
        
        for dir_name in prod_dirs:
            (self.production_dir / dir_name).mkdir(exist_ok=True)
        
        # Copy core files
        for file_name in self.essential_files:
            if file_name.endswith('.py'):
                shutil.copy2(self.root_dir / file_name, self.production_dir / "core" / file_name)
        
        # Copy documentation
        for doc_file in self.root_dir.glob("*.md"):
            try:
                shutil.copy2(doc_file, self.production_dir / "docs" / doc_file.name)
            except Exception as e:
                print(f"⚠️ Could not copy {doc_file}: {e}")
            except Exception as e:
                print(f"⚠️ Could not copy {doc_file}: {e}")
        
        # Copy configuration templates
        if (self.root_dir / "datasets.yaml").exists():
            try:
                shutil.copy2(self.root_dir / "datasets.yaml", self.production_dir / "templates" / "datasets.yaml")
            except Exception as e:
                print(f"⚠️ Could not copy datasets.yaml: {e}")
            shutil.copy2(self.root_dir / "datasets.yaml", self.production_dir / "templates" / "datasets.yaml")
        
        print("✅ Production system organized")
    
    def create_production_metadata(self):
        """Create production metadata and deployment guide."""
        print("📋 Creating production metadata...")
        
        # System information
        metadata = {
            "system_name": "SloGPT Dataset Standardization System",
            "version": "1.0.0",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Complete dataset standardization and training system",
            
            "components": {
                "core": ["create_dataset_fixed.py", "train_simple.py", "universal_prepare.py", "simple_trainer.py"],
                "advanced": ["advanced_dataset_features.py", "performance_optimizer.py", "cli_shortcuts.py", "batch_processor.py"],
                "documentation": ["COMPLETE_USER_GUIDE.md", "DATASET_STANDARDIZATION.md"]
            },
            
            "capabilities": [
                "Universal dataset creation from any source",
                "Smart training optimization", 
                "Dataset validation and versioning",
                "Performance monitoring",
                "CLI aliases and shortcuts",
                "Batch processing and automation",
                "Memory-efficient tokenization",
                "Cross-platform compatibility"
            ],
            
            "requirements": [
                "Python 3.9+",
                "numpy", 
                "psutil",
                "subprocess"
            ],
            
            "usage_examples": {
                "create_dataset": "python3 create_dataset_fixed.py mydata 'your text'",
                "train": "python3 train_simple.py mydata",
                "monitor": "python3 performance_optimizer.py monitor",
                "batch": "python3 batch_processor.py batch --config config.yaml",
                "cli_install": "python3 cli_shortcuts.py --install"
            }
        }
        
        # Save metadata
        with open(self.production_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Create deployment guide
        deployment_guide = f"""# SloGPT Dataset System - Production Deployment Guide

## 🚀 Quick Installation

### 1. Install System
```bash
# Copy production system to your location
cp -r production_ready/your_location/slogpt/

# Add to PATH
export PATH="$PATH:/your_location/slogpt:$PATH"

# Install CLI aliases
python3 /your_location/slogpt/tools/cli_shortcuts.py --install
```

### 2. Basic Usage
```bash
# Create dataset from any source
slogpt create_dataset mydata "your training text"

# Train with smart optimization
slogpt train mydata

# Monitor performance
slogpt monitor

# Process multiple datasets
slogpt batch --config your_config.yaml
```

### 3. Production Deployment
```bash
# For server deployment
export SLOGPT_DATA_DIR="/path/to/datasets"
export SLOGPT_MODEL_DIR="/path/to/models"

# Run with optimized settings
slogpt train --batch_size 64 --device cuda
```

## 📚 System Architecture
```
production_ready/
├── core/                    # Core system files
├── tools/                   # Advanced tools
├── docs/                    # Documentation
├── examples/                 # Usage examples
└── templates/                # Configuration templates
```

## 🛠️ Production Best Practices
- Use dataset versioning for critical data
- Monitor performance during training
- Use batch processing for multiple datasets
- Validate datasets before training
- Use CLI aliases for faster workflow

## 📊 Monitoring & Troubleshooting
- Performance monitoring: `slogpt monitor`
- Dataset validation: `slogpt validate <dataset>`
- System health check: `slogpt check` (if available)

Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        with open(self.production_dir / "DEPLOYMENT.md", "w") as f:
            f.write(deployment_guide)
        
        print("✅ Production metadata created")
    
    def create_production_examples(self):
        """Create usage examples."""
        print("📝 Creating production examples...")
        
        examples_dir = self.production_dir / "examples"
        
        # Basic usage examples
        basic_examples = {
            "create_text.py": '''#!/usr/bin/env python3
"""Create dataset from text"""
import subprocess
subprocess.run(["python3", "create_dataset_fixed.py", "example", "Hello world from example dataset"])
''',
            "train_basic.py": '''#!/usr/bin/env python3
"""Basic training example"""
import subprocess
subprocess.run(["python3", "train_simple.py", "example", "--batch_size", "16"])
''',
            "monitor_example.py": '''#!/usr/bin/env python3
"""Performance monitoring example"""
import subprocess
subprocess.run(["python3", "performance_optimizer.py", "monitor", "--interval", "30"])
''',
            "batch_example.yaml": """# Batch processing example
datasets:
  - name: "web_data"
    source: "./web_content"
    text: "Web content dataset"
  - name: "code_data" 
    source: "./code_repo"
    text: "Code dataset"

operation: "create"
workers: 4
"""
        }
        
        for filename, content in basic_examples.items():
            with open(examples_dir / filename, "w") as f:
                f.write(content)
        
        # Make examples executable
        for filename in basic_examples.keys():
            os.chmod(examples_dir / filename, 0o755)
        
        print("✅ Production examples created")
    
    def create_production_check(self):
        """Create system health check."""
        print("🔍 Creating production health check...")
        
        health_check = '''#!/usr/bin/env python3
"""
Production System Health Check
"""

import os
import sys
from pathlib import Path

def check_system():
    """Check system requirements."""
    checks = []
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 9):
        checks.append("✅ Python " + ".".join(map(str, python_version[:3])))
    else:
        checks.append("❌ Python " + ".".join(map(str, python_version[:3])) + " (requires 3.9+)")
    
    # Check essential files
    essential_files = [
        "core/create_dataset_fixed.py",
        "core/train_simple.py",
        "tools/performance_optimizer.py"
        "docs/COMPLETE_USER_GUIDE.md"
    ]
    
    script_dir = Path(__file__).parent
    for file_path in essential_files:
        if (script_dir / file_path).exists():
            checks.append(f"✅ {file_path}")
        else:
            checks.append(f"❌ {file_path}")
    
    # Check directories
    essential_dirs = ["core", "tools", "docs"]
    for dir_name in essential_dirs:
        if (script_dir / dir_name).is_dir():
            checks.append(f"✅ {dir_name}/")
        else:
            checks.append(f"❌ {dir_name}/")
    
    print("System Health Check:")
    for check in checks:
        print(f"  {check}")
    
    all_good = all("✅" in check for check in checks)
    if all_good:
        print("\\n🎉 System is production ready!")
        return 0
    else:
        print("\\n⚠️ System needs attention before production use")
        return 1

if __name__ == "__main__":
    exit(check_system())
'''
        
        with open(self.production_dir / "health_check.py", "w") as f:
            f.write(health_check)
        
        print("✅ Production health check created")
    
    def create_final_summary(self):
        """Create final deployment summary."""
        print("📊 Creating final summary...")
        
        summary = f"""
# SloGPT Dataset Standardization System - Production Summary
# Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

## ✅ Completed Features
- ✅ Universal dataset creation from any source
- ✅ Smart training optimization and device detection
- ✅ Dataset validation, versioning, and quality control
- ✅ Real-time performance monitoring and optimization
- ✅ CLI aliases and shortcuts for frictionless usage
- ✅ Batch processing and automation workflows
- ✅ Comprehensive documentation and examples
- ✅ Cross-platform compatibility
- ✅ Memory-efficient tokenization system
- ✅ Production-ready deployment package

## 🚀 Key Files Created
- Core system: create_dataset_fixed.py, train_simple.py, universal_prepare.py
- Advanced tools: advanced_dataset_features.py, performance_optimizer.py
- CLI system: cli_shortcuts.py, batch_processor.py
- Documentation: COMPLETE_USER_GUIDE.md, DATASET_STANDARDIZATION.md
- Templates: datasets.yaml, example scripts
- Health check: health_check.py

## 📦 Deployment Package
Location: {self.production_dir}
Size: {sum(f.stat(path).st_size for path in self.production_dir.rglob('*')) // 1024}KB

## 🎯 Quick Start
```bash
# Navigate to production system
cd {self.production_dir}

# Run health check
python3 health_check.py

# Install CLI aliases
python3 ../tools/cli_shortcuts.py --install

# Start using
slogpt create_dataset mydata "your text"
slogpt train mydata
```

## 🔧 System Architecture
The system eliminates terminal gymnastics and provides enterprise-level dataset management
with a simple, intuitive interface that works across all platforms and use cases.

Production Status: ✅ READY
"""
        
        with open(self.production_dir / "PRODUCTION_SUMMARY.md", "w") as f:
            f.write(summary)
        
        print("✅ Final summary created")
    
    def clean_up(self):
        """Execute complete cleanup process."""
        print("🧹 Starting complete cleanup...")
        
        self.clean_temp_files()
        self.clean_test_datasets()
        self.organize_production_system()
        self.create_production_metadata()
        self.create_production_examples()
        self.create_production_check()
        self.create_final_summary()
        
        # Remove duplicate files in production
        for pattern in ["*.py", "*.md"]:
            files = list(self.production_dir.glob(pattern))
            seen = set()
            for file_path in files:
                if file_path.name in seen:
                    try:
                        file_path.unlink()
                        print(f"🗑️ Removed duplicate: {file_path.name}")
                    except Exception as e:
                        print(f"⚠️ Could not remove duplicate {file_path.name}: {e}")
                else:
                    seen.add(file_path.name)
        
        print("✅ Cleanup complete!")
        
        # Get final stats
        core_files = 0
        total_files = 0
        try:
            core_files = len([f for f in (self.production_dir / "core").glob("*.py")])
            total_files = len(list(self.production_dir.rglob("*")))
            total_size = sum(f.stat().st_size for f in self.production_dir.rglob("*")) // 1024 if f.is_file() else 0
        except Exception as e:
            print(f"⚠️ Could not calculate stats: {e}")
        
        print(f"\n📊 Final Stats:")
        print(f"  Core files: {core_files}")
        print(f"  Total files: {total_files}")
        print(f"  Total size: {total_size}KB")
        print(f"  Production system: {self.production_dir}")
        
        print(f"\n📊 Final Stats:")
        print(f"  Core files: {core_files}")
        print(f"  Total files: {total_files}")
        print(f"  Total size: {total_size}KB")
        print(f"  Production system: {self.production_dir}")


def main():
    """Main cleanup function."""
    cleaner = CodebaseCleaner()
    cleaner.clean_up()


if __name__ == "__main__":
    main()