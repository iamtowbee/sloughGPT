#!/usr/bin/env python3
"""Fake lmtrain CLI for testing"""

import sys
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")
    
    # web command
    web = subparsers.add_parser("web")
    web.add_argument("--query")
    web.add_argument("--output")
    web.add_argument("--max-examples", type=int, default=1)
    web.add_argument("--language")
    web.add_argument("--format", default="instruction")
    
    # repo command
    repo = subparsers.add_parser("repo")
    repo.add_argument("--query")
    repo.add_argument("--output")
    repo.add_argument("--max-repos", type=int, default=1)
    repo.add_argument("--max-files", type=int, default=1)
    repo.add_argument("--language")
    repo.add_argument("--format", default="instruction")
    
    args = parser.parse_args()
    
    # Write one fake example
    example = {
        "instruction": args.query or "sample query",
        "output": "sample output"
    }
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(example) + "\n")


if __name__ == "__main__":
    main()
