#!/usr/bin/env python3
"""
Ultra Simple Dataset Creator - Fixed Version

Just train your model. All complexity is handled internally.
No terminal gymnastics required.

Usage:
    python create_dataset.py mydata "Your training text here"
    python create_dataset.py mydata --file path/to/your/file.txt
    python create_dataset.py mydata --folder ./data_folder
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path


def create_dataset(name: str, text: str = None, file_path: str = None, folder_path: str = None):
    """Create dataset with minimal thinking required."""
    
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    dataset_dir = datasets_dir / name
    
    # Determine input source
    if text:
        # Direct text input
        input_file = dataset_dir / "input.txt"
        dataset_dir.mkdir(exist_ok=True)
        
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"✅ Created dataset '{name}' with direct text input")
        
    elif file_path:
        # File input
        if not Path(file_path).exists():
            print(f"❌ File not found: {file_path}")
            sys.exit(1)
        
        dataset_dir.mkdir(exist_ok=True)
        input_file = dataset_dir / "input.txt"
        
        # Copy file content
        with open(file_path, 'r', encoding='utf-8') as src:
            with open(input_file, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        print(f"✅ Created dataset '{name}' from file: {file_path}")
        
    elif folder_path:
        # Folder input (use universal preparer)
        if not Path(folder_path).exists():
            print(f"❌ Folder not found: {folder_path}")
            sys.exit(1)
            
        print(f"📁 Creating dataset '{name}' from folder: {folder_path}")
        cmd = f"python3 universal_prepare.py --name {name} --source {folder_path} --recursive"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Created dataset '{name}' from folder")
            return
        else:
            print(f"❌ Failed to create dataset: {result.stderr}")
            return
    
    else:
        # No input provided - create empty template
        dataset_dir.mkdir(exist_ok=True)
        input_file = dataset_dir / "input.txt"
        
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write("Your training text goes here.\n")
        print(f"✅ Created empty dataset template '{name}'")
        print(f"📝 Edit {input_file} to add your training text")
        return
    
    # Prepare dataset (unless already handled by universal preparer)
    if not folder_path:
        print(f"⏳ Preparing dataset for training...")
        
        # Copy prepare.py to dataset directory
        prepare_src = Path(__file__).parent / "datasets" / "mydata" / "prepare.py"
        prepare_dst = dataset_dir / "prepare.py"
        
        if prepare_src.exists():
            shutil.copy(prepare_src, prepare_dst)
            
            cmd = f"cd {dataset_dir} && python3 prepare.py"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"🎉 Dataset '{name}' ready for training!")
                print(f"🚀 Train with: python3 train_simple.py {name}")
            else:
                print(f"❌ Preparation failed: {result.stderr}")
        else:
            print(f"❌ prepare.py not found at {prepare_src}")


def main():
    parser = argparse.ArgumentParser(
        description="Create dataset without thinking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python create_dataset.py mydata "Your training text here"
    python create_dataset.py mydata --file mytext.txt
    python create_dataset.py code_data --folder ./src
    python create_dataset.py mydata  # Creates empty template
        """
    )
    
    parser.add_argument("name", help="Dataset name")
    parser.add_argument("text", nargs="?", help="Training text (optional)")
    parser.add_argument("--file", help="Input file path")
    parser.add_argument("--folder", help="Input folder path (recursive)")
    
    args = parser.parse_args()
    
    create_dataset(
        args.name,
        args.text,
        args.file,
        args.folder
    )


if __name__ == "__main__":
    main()