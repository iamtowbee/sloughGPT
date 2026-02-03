#!/usr/bin/env python3
"""Simple corpus to dataset converter for SloughGPT training."""

import json
import sys
import os
from pathlib import Path

def convert_corpus_to_dataset(corpus_file: Path, dataset_name: str):
    """Convert corpus JSONL to training dataset format."""
    
    # Read corpus and extract content
    content_lines = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if 'content' in item:
                    # Check if content is JSON string or raw text
                    content = item['content']
                    try:
                        # Try to parse as JSON
                        nested = json.loads(content)
                        if isinstance(nested, dict):
                            if 'instruction' in nested and 'output' in nested:
                                # Instruction format
                                content_lines.append(f"Question: {nested['instruction']}\nAnswer: {nested['output']}")
                            else:
                                # Just use the content as-is
                                content_lines.append(json.dumps(nested, ensure_ascii=False))
                        else:
                            content_lines.append(str(nested))
                    except json.JSONDecodeError:
                        # Raw text content
                        content_lines.append(content)
                else:
                    # Use the file path and content
                    if 'path' in item and 'content' in item:
                        # Add file context
                        file_path = item['path']
                        content_lines.append(f"// File: {file_path}\n{item['content']}")
                    else:
                        # Use the whole item as content
                        content_lines.append(json.dumps(item, ensure_ascii=False))
            except json.JSONDecodeError:
                continue
    
    # Create dataset directory
    dataset_dir = Path(f"data/{dataset_name}")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Write input.txt for the dataset
    input_file = dataset_dir / "input.txt"
    with open(input_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content_lines))
    
    print(f"✅ Created dataset: {dataset_dir}")
    print(f"📝 Lines of content: {len(content_lines)}")
    print(f"📁 Input file: {input_file}")
    
    return input_file

def main():
    if len(sys.argv) != 5 or sys.argv[1] != '--input' or sys.argv[3] != '--dataset':
        print("Usage: python simple_corpus_converter.py --input <corpus_file> --dataset <dataset_name>")
        sys.exit(1)
    
    corpus_file = Path(sys.argv[2])
    dataset_name = sys.argv[4]
    
    if not corpus_file.exists():
        print(f"❌ Corpus file not found: {corpus_file}")
        sys.exit(1)
    
    convert_corpus_to_dataset(corpus_file, dataset_name)

if __name__ == "__main__":
    main()