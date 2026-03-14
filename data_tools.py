#!/usr/bin/env python3
"""Data Tools - Dataset validation and preprocessing utilities."""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional


def validate_jsonl(path: str) -> Dict[str, Any]:
    """Validate a JSONL file."""
    issues = []
    line_count = 0
    valid_count = 0
    
    try:
        with open(path, 'r') as f:
            for i, line in enumerate(f, 1):
                line_count += 1
                line = line.strip()
                if not line:
                    issues.append(f"Line {i}: Empty line")
                    continue
                try:
                    data = json.loads(line)
                    valid_count += 1
                except json.JSONDecodeError as e:
                    issues.append(f"Line {i}: Invalid JSON - {e}")
    except FileNotFoundError:
        return {"valid": False, "error": f"File not found: {path}"}
    
    return {
        "valid": len(issues) == 0,
        "path": path,
        "total_lines": line_count,
        "valid_lines": valid_count,
        "issues": issues[:10]  # Limit to first 10
    }


def validate_dataset(path: str) -> Dict[str, Any]:
    """Validate a dataset directory."""
    path_obj = Path(path)
    
    if not path_obj.exists():
        return {"valid": False, "error": f"Path not found: {path}"}
    
    result = {"path": path, "files": [], "total_size": 0}
    
    if path_obj.is_file():
        if path.endswith('.jsonl'):
            result["files"].append(path)
            result["total_size"] = path_obj.stat().st_size
            result["validation"] = validate_jsonl(path)
    else:
        for f in path_obj.rglob('*'):
            if f.is_file():
                result["files"].append(str(f))
                result["total_size"] += f.stat().st_size
    
    result["file_count"] = len(result["files"])
    return result


def convert_jsonl_to_json(input_path: str, output_path: str) -> bool:
    """Convert JSONL to JSON array."""
    data = []
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return True


def convert_json_to_jsonl(input_path: str, output_path: str) -> bool:
    """Convert JSON array to JSONL."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    return True


def stats_jsonl(path: str) -> Dict[str, Any]:
    """Get statistics for a JSONL file or plain text."""
    total_lines = 0
    total_chars = 0
    keys = set()
    is_jsonl = False
    
    with open(path, 'r') as f:
        first_line = f.readline()
        if first_line.strip():
            try:
                json.loads(first_line.strip())
                is_jsonl = True
            except:
                is_jsonl = False
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                total_lines += 1
                total_chars += len(line)
                if is_jsonl:
                    try:
                        data = json.loads(line)
                        keys.update(data.keys())
                    except:
                        pass
    
    result = {
        "path": path,
        "total_lines": total_lines,
        "total_chars": total_chars,
        "avg_line_length": total_chars // total_lines if total_lines > 0 else 0,
    }
    
    if is_jsonl:
        result["keys"] = sorted(keys)
    
    return result


def filter_jsonl(input_path: str, output_path: str, key: str, min_length: int = 0) -> bool:
    """Filter JSONL by key minimum length."""
    count = 0
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            line = line.strip()
            if line:
                data = json.loads(line)
                if key in data and len(data[key]) >= min_length:
                    fout.write(line + '\n')
                    count += 1
    return count > 0


def main():
    parser = argparse.ArgumentParser(description="Data Tools for SloughGPT")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("path", help="Dataset path")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get dataset statistics")
    stats_parser.add_argument("path", help="JSONL file path")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert between formats")
    convert_parser.add_argument("input", help="Input file")
    convert_parser.add_argument("output", help="Output file")
    
    # Filter command
    filter_parser = subparsers.add_parser("filter", help="Filter JSONL by key length")
    filter_parser.add_argument("input", help="Input file")
    filter_parser.add_argument("output", help="Output file")
    filter_parser.add_argument("--key", default="text", help="Key to filter on")
    filter_parser.add_argument("--min-length", type=int, default=10, help="Minimum length")
    
    args = parser.parse_args()
    
    if args.command == "validate":
        result = validate_dataset(args.path)
        print(json.dumps(result, indent=2))
    elif args.command == "stats":
        result = stats_jsonl(args.path)
        print(json.dumps(result, indent=2))
    elif args.command == "convert":
        if args.input.endswith('.jsonl'):
            convert_jsonl_to_json(args.input, args.output)
        elif args.input.endswith('.json'):
            convert_json_to_jsonl(args.input, args.output)
        print(f"Converted {args.input} -> {args.output}")
    elif args.command == "filter":
        count = filter_jsonl(args.input, args.output, args.key, args.min_length)
        print(f"Filtered {count} lines to {args.output}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
