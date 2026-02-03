#!/usr/bin/env python3
"""
Universal Dataset Preparer for SloGPT

Standardizes multiple dataset formats into unified training format.
Handles:
- Text files (.txt, .md, .py, .js, .json, .csv, etc.)
- Multiple input files per dataset
- Directory processing
- Large file streaming
- Mixed format datasets

Usage:
    python universal_prepare.py --name mydataset --source data.txt
    python universal_prepare.py --name mydataset --source dir/ --recursive
    python universal_prepare.py --config datasets.yaml
"""

import argparse
import os
import pickle
import sys
import yaml
import json
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Optional, Union, Generator
import numpy as np


class UniversalDatasetPreparer:
    """Universal dataset preparation tool for SloGPT training."""
    
    def __init__(self, dataset_name: str, output_dir: Optional[str] = None):
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir or f"datasets/{dataset_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def read_file(self, file_path: Path, encoding: str = "utf-8") -> str:
        """Read file content with error handling."""
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding=encoding) as f:
                    data = json.load(f)
                    # Extract text from JSON
                    if isinstance(data, list):
                        return '\n'.join(str(item) for item in data)
                    elif isinstance(data, dict):
                        return json.dumps(data, indent=2)
                    else:
                        return str(data)
            elif file_path.suffix.lower() == '.csv':
                # Simple CSV text extraction
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            else:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    return f.read()
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return ""
    
    def get_files(self, source: Union[str, Path], recursive: bool = False) -> List[Path]:
        """Get list of files to process."""
        source = Path(source)
        
        if source.is_file():
            return [source]
        elif source.is_dir():
            if recursive:
                return [f for f in source.rglob("*") if f.is_file()]
            else:
                return [f for f in source.iterdir() if f.is_file()]
        else:
            raise FileNotFoundError(f"Source not found: {source}")
    
    def stream_text_files(self, files: List[Path], chunk_size: int = 1_000_000) -> Generator[str, None, None]:
        """Stream text from multiple files."""
        for file_path in files:
            print(f"Processing: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        yield chunk
            except Exception as e:
                print(f"Warning: Error streaming {file_path}: {e}")
                continue
    
    def prepare_from_files(self, files: List[Path], streaming: bool = False) -> None:
        """Prepare dataset from list of files."""
        if not files:
            raise ValueError("No files to process")
        
        # Calculate total size for mode selection
        total_size = sum(f.stat().st_size for f in files if f.exists())
        print(f"Total data size: {total_size / 1024 / 1024:.2f} MB")
        print(f"Files to process: {len(files)}")
        
        if streaming or total_size > 500_000_000:  # > 500MB
            self._prepare_streaming(files)
        else:
            self._prepare_standard(files)
    
    def _prepare_standard(self, files: List[Path]) -> None:
        """Standard in-memory preparation."""
        print("Using standard preparation mode...")
        
        # Read all text
        all_text = []
        for file_path in files:
            content = self.read_file(file_path)
            if content.strip():
                all_text.append(content)
        
        data = '\n\n'.join(all_text)
        print(f"Loaded {len(data):,} characters")
        
        # Build vocabulary
        chars = sorted(list(set(data)))
        vocab_size = len(chars)
        print(f"Vocabulary size: {vocab_size} unique characters")
        
        # Create mappings
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        
        # Tokenization functions
        def encode(s): return [stoi[c] for c in s]
        def decode(tokens): return "".join([itos[t] for t in tokens])
        
        # Train/val split
        n = len(data)
        train_data = data[:int(n * 0.9)]
        val_data = data[int(n * 0.9):]
        
        # Encode and save
        train_ids = np.array(encode(train_data), dtype=np.uint16)
        val_ids = np.array(encode(val_data), dtype=np.uint16)
        
        print(f"Train: {len(train_ids):,} tokens")
        print(f"Val: {len(val_ids):,} tokens")
        
        # Save files
        train_ids.tofile(self.output_dir / "train.bin")
        val_ids.tofile(self.output_dir / "val.bin")
        
        # Save metadata
        meta = {
            "vocab_size": vocab_size,
            "itos": itos,
            "stoi": stoi,
            "source_files": [str(f) for f in files],
            "total_characters": len(data),
            "train_tokens": len(train_ids),
            "val_tokens": len(val_ids)
        }
        
        with open(self.output_dir / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        
        print(f"\nDataset '{self.dataset_name}' prepared successfully!")
        print(f"Output directory: {self.output_dir}")
        print(f"Now run: python train.py --dataset={self.dataset_name}")
    
    def _prepare_streaming(self, files: List[Path]) -> None:
        """Streaming preparation for large datasets."""
        print("Using streaming preparation mode...")
        
        # First pass: build vocabulary
        print("Pass 1: Building vocabulary...")
        char_counts = Counter()
        total_chars = 0
        
        for chunk in self.stream_text_files(files):
            char_counts.update(chunk)
            total_chars += len(chunk)
            print(f"  Scanned {total_chars:,} characters...")
        
        # Build vocabulary
        chars = sorted(char_counts.keys(), key=lambda c: -char_counts[c])
        vocab_size = len(chars)
        print(f"Vocabulary size: {vocab_size}")
        
        # Create mappings
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        
        # Second pass: tokenize and split
        print("\nPass 2: Tokenizing and writing binary...")
        train_size = int(total_chars * 0.9)
        
        train_path = self.output_dir / "train.bin"
        val_path = self.output_dir / "val.bin"
        
        chars_written = 0
        train_tokens = 0
        val_tokens = 0
        
        with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:
            for chunk in self.stream_text_files(files):
                tokens = np.array([stoi.get(c, 0) for c in chunk], dtype=np.uint16)
                
                # Split between train and val
                remaining_train = max(0, train_size - chars_written)
                
                if remaining_train > 0:
                    train_chunk = tokens[:remaining_train]
                    f_train.write(train_chunk.tobytes())
                    train_tokens += len(train_chunk)
                
                if remaining_train < len(tokens):
                    val_chunk = tokens[remaining_train:]
                    f_val.write(val_chunk.tobytes())
                    val_tokens += len(val_chunk)
                
                chars_written += len(chunk)
                if chars_written % 10_000_000 == 0:
                    print(f"  Processed {chars_written:,} / {total_chars:,} characters...")
        
        # Save metadata
        meta = {
            "vocab_size": vocab_size,
            "stoi": stoi,
            "itos": itos,
            "source_files": [str(f) for f in files],
            "total_characters": total_chars,
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "streaming": True
        }
        
        with open(self.output_dir / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        
        print(f"\nDataset '{self.dataset_name}' prepared successfully!")
        print(f"  Train tokens: {train_tokens:,}")
        print(f"  Val tokens: {val_tokens:,}")
        print(f"  Output directory: {self.output_dir}")
        print(f"Now run: python train.py --dataset={self.dataset_name}")
    
    @classmethod
    def from_config(cls, config_path: str) -> List['UniversalDatasetPreparer']:
        """Create multiple preparers from config file."""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        preparers = []
        for dataset_config in config['datasets']:
            preparer = cls(
                dataset_config['name'],
                dataset_config.get('output_dir')
            )
            
            files = []
            for source in dataset_config['sources']:
                if isinstance(source, str):
                    source_path = Path(source)
                    if source_path.exists():
                        if source_path.is_file():
                            files.append(source_path)
                        elif source_path.is_dir():
                            recursive = dataset_config.get('recursive', False)
                            files.extend(preparer.get_files(source_path, recursive))
                elif isinstance(source, dict):
                    # Advanced source config
                    source_path = Path(source['path'])
                    if source_path.exists():
                        if source_path.is_file():
                            files.append(source_path)
                        elif source_path.is_dir():
                            files.extend(preparer.get_files(source_path, source.get('recursive', False)))
            
            if files:
                preparer.prepare_from_files(
                    files, 
                    streaming=dataset_config.get('streaming', False)
                )
                preparers.append(preparer)
        
        return preparers


def main():
    parser = argparse.ArgumentParser(description="Universal dataset preparer for SloGPT")
    parser.add_argument("--name", required=True, help="Dataset name")
    parser.add_argument("--source", help="Source file or directory")
    parser.add_argument("--recursive", action="store_true", help="Process directories recursively")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode for large files")
    parser.add_argument("--output", help="Output directory (default: datasets/<name>)")
    parser.add_argument("--config", help="Config file for batch preparation")
    parser.add_argument("--encoding", default="utf-8", help="File encoding")
    
    args = parser.parse_args()
    
    if args.config:
        # Batch preparation from config
        preparers = UniversalDatasetPreparer.from_config(args.config)
        print(f"Prepared {len(preparers)} datasets from config")
    else:
        if not args.source:
            print("Error: --source or --config is required")
            sys.exit(1)
        
        # Single dataset preparation
        preparer = UniversalDatasetPreparer(args.name, args.output)
        
        try:
            files = preparer.get_files(args.source, args.recursive)
            preparer.prepare_from_files(files, args.streaming)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()