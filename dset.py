#!/usr/bin/env python3
"""dset – single‑tool dataset fetcher

A tiny CLI that:
  1️⃣ Runs lmtrain (web or GitHub) to gather code examples.
  2️⃣ Converts the lmtrain JSONL into the repo_obtainer corpus format.
  3️⃣ Writes the corpus to `runs/<dataset>.jsonl`.

This keeps the core functionality of both original tools while exposing
a concise, well‑named entry point.

Usage examples::

    python -m dset \
        --source web \
        --query "react hooks examples" \
        --dataset react_hooks

    python -m dset \
        --source github \
        --query "machine learning" \
        --language python \
        --dataset ml_python
"""

import sys
import os
from pathlib import Path
import argparse
import tempfile
import shutil
import importlib.util

# ----------------------------------------------------------------------
# Load the unified repo_obtainer implementation directly from its file.
# ----------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
REPO_OBTAINER_FILE = CURRENT_DIR / "well_tool" / "unified_tool" / "repo_obtainer.py"
spec = importlib.util.spec_from_file_location("repo_obtainer_unified", REPO_OBTAINER_FILE)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load repo_obtainer from {REPO_OBTAINER_FILE}")
repo_obtainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repo_obtainer)

# Pull the three helpers we need directly.
run_lmtrain_web_search = repo_obtainer.run_lmtrain_web_search  # type: ignore
run_lmtrain_github_search = repo_obtainer.run_lmtrain_github_search  # type: ignore
convert_lmtrain_to_corpus = repo_obtainer.convert_lmtrain_to_corpus  # type: ignore

# ----------------------------------------------------------------------
# CLI implementation (unchanged logic)
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset fetcher – single tool (dset)")
    parser.add_argument(
        "--source",
        choices=["web", "github"],
        default="web",
        help="lmtrain source: web search or GitHub repository search",
    )
    parser.add_argument("--query", required=True, help="Search query string")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Base name for the output corpus (writes to runs/\u003cname\u003e.jsonl)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Maximum examples for web search (ignored for GitHub)",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=10,
        help="Maximum repositories for GitHub search (ignored for web)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=20,
        help="Maximum files per repo for GitHub search (ignored for web)",
    )
    parser.add_argument("--language", help="Filter results by language (optional)")
    parser.add_argument(
        "--format",
        default="instruction",
        help="Output format for lmtrain (instruction/completion/chat)",
    )
    parser.add_argument(
        "--output",
        help="Custom output path for final corpus JSONL (default runs/\u003cdataset\u003e.jsonl)",
    )
    args = parser.parse_args()

    # Temporary directory for lmtrain JSONL output
    temp_dir = Path(tempfile.mkdtemp(prefix="lmtrain_tmp_"))
    lmtrain_output = temp_dir / "lmtrain_output.jsonl"

    # Run the selected lmtrain command
    success = False
    if args.source == "web":
        success = run_lmtrain_web_search(
            query=args.query,
            output_path=str(lmtrain_output),
            max_examples=args.max_examples,
            language=args.language,
            format_type=args.format,
        )
    else:  # github
        success = run_lmtrain_github_search(
            query=args.query,
            output_path=str(lmtrain_output),
            max_repos=args.max_repos,
            max_files=args.max_files,
            language=args.language,
            format_type=args.format,
        )

    if not success:
        sys.stderr.write("\n❌ lmtrain step failed – aborting.\n")
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)

    # Convert lmtrain JSONL → repo_obtainer corpus format
    output_path = Path(args.output) if args.output else Path("runs") / f"{args.dataset}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted = convert_lmtrain_to_corpus(
        lmtrain_output=str(lmtrain_output),
        corpus_output=str(output_path),
    )

    # Cleanup temporary files
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"✅ Corpus written to {output_path} (converted {converted} files)")

if __name__ == "__main__":
    main()
    parser = argparse.ArgumentParser(description="Dataset fetcher – single tool (dset)")
    parser.add_argument(
        "--source",
        choices=["web", "github"],
        default="web",
        help="lmtrain source: web search or GitHub repository search",
    )
    parser.add_argument("--query", required=True, help="Search query string")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Base name for the output corpus (writes to runs/<name>.jsonl)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Maximum examples for web search (ignored for GitHub)",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=10,
        help="Maximum repositories for GitHub search (ignored for web)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=20,
        help="Maximum files per repo for GitHub search (ignored for web)",
    )
    parser.add_argument("--language", help="Filter results by language (optional)")
    parser.add_argument(
        "--format",
        default="instruction",
        help="Output format for lmtrain (instruction/completion/chat)",
    )
    parser.add_argument(
        "--output",
        help="Custom output path for final corpus JSONL (default runs/<dataset>.jsonl)",
    )
    args = parser.parse_args()

    # Temporary directory for lmtrain JSONL output
    temp_dir = Path(tempfile.mkdtemp(prefix="lmtrain_tmp_"))
    lmtrain_output = temp_dir / "lmtrain_output.jsonl"

    # Run the selected lmtrain command
    success = False
    if args.source == "web":
        success = run_lmtrain_web_search(
            query=args.query,
            output_path=str(lmtrain_output),
            max_examples=args.max_examples,
            language=args.language,
            format_type=args.format,
        )
    else:  # github
        success = run_lmtrain_github_search(
            query=args.query,
            output_path=str(lmtrain_output),
            max_repos=args.max_repos,
            max_files=args.max_files,
            language=args.language,
            format_type=args.format,
        )

    if not success:
        sys.stderr.write("\n❌ lmtrain step failed – aborting.\n")
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)

    # Convert lmtrain JSONL → repo_obtainer corpus format
    output_path = Path(args.output) if args.output else Path("runs") / f"{args.dataset}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted = convert_lmtrain_to_corpus(
        lmtrain_output=str(lmtrain_output),
        corpus_output=str(output_path),
    )

    # Cleanup temporary files
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"✅ Corpus written to {output_path} (converted {converted} files)")

if __name__ == "__main__":
    main()
    parser = argparse.ArgumentParser(description="Dataset fetcher – single tool (dset)")
    parser.add_argument(
        "--source",
        choices=["web", "github"],
        default="web",
        help="lmtrain source: web search or GitHub repository search",
    )
    parser.add_argument("--query", required=True, help="Search query string")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Base name for the output corpus (writes to runs/<name>.jsonl)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Maximum examples for web search (ignored for GitHub)",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=10,
        help="Maximum repositories for GitHub search (ignored for web)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=20,
        help="Maximum files per repo for GitHub search (ignored for web)",
    )
    parser.add_argument("--language", help="Filter results by language (optional)")
    parser.add_argument(
        "--format",
        default="instruction",
        help="Output format for lmtrain (instruction/completion/chat)",
    )
    parser.add_argument(
        "--output",
        help="Custom output path for final corpus JSONL (default runs/<dataset>.jsonl)",
    )
    args = parser.parse_args()

    # Temporary directory for lmtrain JSONL output
    temp_dir = Path(tempfile.mkdtemp(prefix="lmtrain_tmp_"))
    lmtrain_output = temp_dir / "lmtrain_output.jsonl"

    # Run the selected lmtrain command
    success = False
    if args.source == "web":
        success = run_lmtrain_web_search(
            query=args.query,
            output_path=str(lmtrain_output),
            max_examples=args.max_examples,
            language=args.language,
            format_type=args.format,
        )
    else:  # github
        success = run_lmtrain_github_search(
            query=args.query,
            output_path=str(lmtrain_output),
            max_repos=args.max_repos,
            max_files=args.max_files,
            language=args.language,
            format_type=args.format,
        )

    if not success:
        sys.stderr.write("\n❌ lmtrain step failed – aborting.\n")
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)

    # Convert lmtrain JSONL → repo_obtainer corpus format
    output_path = Path(args.output) if args.output else Path("runs") / f"{args.dataset}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted = convert_lmtrain_to_corpus(
        lmtrain_output=str(lmtrain_output),
        corpus_output=str(output_path),
    )

    # Cleanup temporary files
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"✅ Corpus written to {output_path} (converted {converted} files)")

if __name__ == "__main__":
    main()

    parser = argparse.ArgumentParser(description="Dataset fetcher – single tool (dset)")
    parser.add_argument(
        "--source",
        choices=["web", "github"],
        default="web",
        help="lmtrain source: web search or GitHub repository search",
    )
    parser.add_argument("--query", required=True, help="Search query string")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Base name for the output corpus (writes to runs/<name>.jsonl)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Maximum examples for web search (ignored for GitHub)",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=10,
        help="Maximum repositories for GitHub search (ignored for web)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=20,
        help="Maximum files per repo for GitHub search (ignored for web)",
    )
    parser.add_argument("--language", help="Filter results by language (optional)")
    parser.add_argument(
        "--format",
        default="instruction",
        help="Output format for lmtrain (instruction/completion/chat)",
    )
    parser.add_argument(
        "--output",
        help="Custom output path for final corpus JSONL (default runs/<dataset>.jsonl)",
    )
    args = parser.parse_args()

    # Temporary directory for lmtrain JSONL output
    temp_dir = Path(tempfile.mkdtemp(prefix="lmtrain_tmp_"))
    lmtrain_output = temp_dir / "lmtrain_output.jsonl"

    # --------------------------------------------------------------
    # Run the selected lmtrain command
    # --------------------------------------------------------------
    success = False
    if args.source == "web":
        success = run_lmtrain_web_search(
            query=args.query,
            output_path=str(lmtrain_output),
            max_examples=args.max_examples,
            language=args.language,
            format_type=args.format,
        )
    else:  # github
        success = run_lmtrain_github_search(
            query=args.query,
            output_path=str(lmtrain_output),
            max_repos=args.max_repos,
            max_files=args.max_files,
            language=args.language,
            format_type=args.format,
        )

    if not success:
        sys.stderr.write("\n❌ lmtrain step failed – aborting.\n")
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)

    # --------------------------------------------------------------
    # Convert lmtrain JSONL → repo_obtainer corpus format
    # --------------------------------------------------------------
    output_path = Path(args.output) if args.output else Path("runs") / f"{args.dataset}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted = convert_lmtrain_to_corpus(
        lmtrain_output=str(lmtrain_output),
        corpus_output=str(output_path),
    )

    # Clean up temporary files
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"✅ Corpus written to {output_path} (converted {converted} files)")


if __name__ == "__main__":
    main()
