#!/usr/bin/env python3
"""dset – improved dataset fetcher

Better UX, error handling, and extensibility.
"""

import argparse
import json
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from .repo_obtainer import (
    run_lmtrain_web_search,
    run_lmtrain_github_search,
    fetch_to_corpus,
)
from .stats import DatasetStats


def validate_args(args: argparse.Namespace) -> bool:
    """Validate input arguments."""
    if args.source == "web" and args.max_examples <= 0:
        print("❌ --max-examples must be positive for web search")
        return False
    if args.source == "github" and (args.max_repos <= 0 or args.max_files <= 0):
        print("❌ --max-repos and --max-files must be positive for GitHub search")
        return False
    if not args.query or not args.query.strip():
        print("❌ --query cannot be empty")
        return False
    return True


def display_progress(source: str, query: str) -> None:
    """Show progress information."""
    print(f"🔍 Fetching from {source}: {query}")
    if source == "web":
        print("📡 Searching web for examples...")
    else:
        print("🐙 Searching GitHub repositories...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="dset – improved dataset fetcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -c "from dset.cli import main; main()" \\
    --source web --query "react hooks" --dataset react
  python3 -c "from dset.cli import main; main()" \\
    --source github --query "machine learning" --language python --dataset ml
        """
    )
    
    # Core arguments
    parser.add_argument(
        "--source",
        choices=["web", "github"],
        default="web",
        help="Source: web search or GitHub repositories (default: web)",
    )
    parser.add_argument(
        "--query", 
        required=True, 
        help="Search query for finding examples"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (output: runs/<name>.jsonl)",
    )
    
    # Web search options
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Maximum examples for web search (default: 100)",
    )
    
    # GitHub search options
    parser.add_argument(
        "--max-repos",
        type=int,
        default=10,
        help="Maximum repositories for GitHub search (default: 10)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=20,
        help="Maximum files per repo for GitHub search (default: 20)",
    )
    
    # Common options
    parser.add_argument(
        "--language", 
        help="Filter results by programming language"
    )
    parser.add_argument(
        "--format",
        default="instruction",
        choices=["instruction", "completion", "chat"],
        help="Output format for lmtrain (default: instruction)",
    )
    parser.add_argument(
        "--output",
        help="Custom output path (default: runs/<dataset>.jsonl)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output",
    )
    
    args = parser.parse_args()

    if not validate_args(args):
        sys.exit(1)

    display_progress(args.source, args.query)

    # Setup stats tracking
    stats = DatasetStats()
    stats.start(args.source, args.query)

    # Setup temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix="dset_tmp_"))
    lmtrain_out = temp_dir / "lmtrain_output.jsonl"
    
    try:
        # Run lmtrain
        success = False
        if args.source == "web":
            if args.verbose:
                print(f"📊 Web search limit: {args.max_examples} examples")
            success = run_lmtrain_web_search(
                query=args.query,
                output_path=str(lmtrain_out),
                max_examples=args.max_examples,
                language=args.language,
                format_type=args.format,
            )
        else:  # github
            if args.verbose:
                print(f"📊 GitHub search limit: {args.max_repos} repos, {args.max_files} files each")
            success = run_lmtrain_github_search(
                query=args.query,
                output_path=str(lmtrain_out),
                max_repos=args.max_repos,
                max_files=args.max_files,
                language=args.language,
                format_type=args.format,
            )

        if not success:
            sys.stderr.write("\n❌ lmtrain failed\n")
            sys.exit(1)

        # Write final corpus
        out_path = Path(args.output) if args.output else Path("runs") / f"{args.dataset}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        converted = fetch_to_corpus(
            lmtrain_output=str(lmtrain_out),
            corpus_output=str(out_path),
        )
        
        # Complete stats and show summary
        stats.finish(converted, out_path)
        stats.print_summary()
        
        if args.verbose:
            print(f"📁 Output file: {out_path.resolve()}")
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()