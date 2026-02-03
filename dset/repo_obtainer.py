"""
repo_obtainer - clone or read a repo and build a file tree index or training corpus.

Usage:
  python repo_obtainer.py index --source https://github.com/user/repo.git --dest runs/vcs
  python repo_obtainer.py export --source /path/to/local/repo --format jsonl
  python repo_obtainer.py web --query "react hooks examples" --dataset react_hooks
  python repo_obtainer.py github --query "machine learning" --language python --dataset ml_repos
"""

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_IGNORES = {
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
}


def run_git_clone(source: str, dest: Path, branch: str = None, depth: int = None) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    repo_name = Path(source).stem.replace(".git", "")
    target = dest / repo_name
    if target.exists():
        return target
    cmd = ["git", "clone", source, str(target)]
    if branch:
        cmd.extend(["-b", branch])
    if depth:
        cmd.extend(["--depth", str(depth)])
    subprocess.check_call(cmd)
    return target


def file_hash(path: Path) -> str:
    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha1.update(chunk)
    return sha1.hexdigest()


def build_tree(
    root: Path,
    ignores: set,
    max_files: int = None,
    include_hidden: bool = False,
    hash_files: bool = False,
) -> dict:
    root = root.resolve()
    files = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d not in ignores and (include_hidden or not d.startswith("."))
        ]
        for name in filenames:
            if name in ignores:
                continue
            if not include_hidden and name.startswith("."):
                continue
            full = Path(dirpath) / name
            try:
                stat = full.stat()
            except OSError:
                continue
            entry = {
                "path": str(full.relative_to(root)),
                "size": stat.st_size,
                "mtime": int(stat.st_mtime),
            }
            if hash_files:
                entry["sha1"] = file_hash(full)
            files.append(entry)
            count += 1
            if max_files and count >= max_files:
                return {"root": str(root), "files": files, "truncated": True}
    return {"root": str(root), "files": files, "truncated": False}


def iter_files(root: Path, ignores: set[str], include_hidden: bool):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d not in ignores and (include_hidden or not d.startswith("."))
        ]
        for name in filenames:
            if name in ignores:
                continue
            if not include_hidden and name.startswith("."):
                continue
            yield Path(dirpath) / name


def export_corpus(
    root: Path,
    ignores: set,
    include_hidden: bool,
    max_files: int,
    max_bytes: int,
    output: Path,
) -> int:
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for full in iter_files(root, ignores, include_hidden):
            try:
                stat = full.stat()
            except OSError:
                continue
            if max_files and count >= max_files:
                break
            try:
                content = full.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if max_bytes and len(content) > max_bytes:
                content = content[:max_bytes]
            record = {
                "path": str(full.relative_to(root)),
                "size": stat.st_size,
                "mtime": int(stat.st_mtime),
                "content": content,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def detect_language(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".py":
        return "python"
    if ext in {".js", ".jsx"}:
        return "javascript"
    if ext in {".ts", ".tsx"}:
        return "typescript"
    if ext in {".md", ".rst"}:
        return "markdown"
    if ext in {".json", ".yml", ".yaml"}:
        return "config"
    if ext in {".sh", ".bash"}:
        return "shell"
    return "text"


def python_extract(text: str) -> tuple[list[str], list[str], str]:
    exports = []
    imports = []
    summary = ""
    try:
        tree = ast.parse(text)
        summary = ast.get_docstring(tree) or ""
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                exports.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                exports.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
    except SyntaxError:
        summary = ""
    return exports, imports, summary


def chunk_text(text: str, max_lines: int, max_chars: int) -> list[dict]:
    lines = text.splitlines()
    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        chunk_lines = lines[start:end]
        chunk_text = "\n".join(chunk_lines)
        if len(chunk_text) > max_chars:
            chunk_text = chunk_text[:max_chars]
        chunks.append(
            {
                "start_line": start + 1,
                "end_line": end,
                "text": chunk_text,
            }
        )
        start = end
    return chunks


from typing import Optional

def export_map(
    root: Path,
    ignores: set[str],
    include_hidden: bool,
    max_files: Optional[int] = None,
    max_bytes: int = 10_000_000,
    max_lines: int = 1000,
    max_chars: int = 5000,
    output: Path = Path("runs/output.jsonl"),
) -> int:
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for full in iter_files(root, ignores, include_hidden):
            if max_files and count >= max_files:
                break
            try:
                stat = full.stat()
            except OSError:
                continue
            try:
                content = full.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if len(content) > max_bytes:
                content = content[:max_bytes]
            language = detect_language(full)
            exports = []
            dependencies = []
            summary = ""
            if language == "python":
                exports, dependencies, summary = python_extract(content)
            chunks = chunk_text(content, max_lines=max_lines, max_chars=max_chars)
            record = {
                "path": str(full.relative_to(root)),
                "language": language,
                "size": stat.st_size,
                "mtime": int(stat.st_mtime),
                "summary": summary,
                "exports": exports,
                "dependencies": dependencies,
                "chunks": chunks,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def run_lmtrain_web_search(
    query: str,
    output_path: str,
    max_examples: int = 100,
    language: str = None,
    format_type: str = "instruction",
) -> bool:
    """Run lmtrain web search and save results."""
    try:
        # Try to run lmtrain via subprocess
        lmtrain_path = Path(__file__).parent.parent / "lmtrain"
        cmd = [
            sys.executable,
            str(lmtrain_path / "cli.py"),
            "web",
            "--query",
            query,
            "--output",
            output_path,
            "--max-examples",
            str(max_examples),
            "--format",
            format_type,
        ]

        if language:
            cmd.extend(["--language", language])

        # Change to lmtrain directory and run command
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(lmtrain_path))

        if result.returncode == 0:
            return True
        else:
            print(f"lmtrain failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"Error running lmtrain: {e}")
        return False


def run_lmtrain_github_search(
    query: str,
    output_path: str,
    max_repos: int = 10,
    max_files: int = 20,
    language: str = None,
    format_type: str = "instruction",
) -> bool:
    """Run lmtrain GitHub search and save results."""
    try:
        # Try to run lmtrain via subprocess
        lmtrain_path = Path(__file__).parent.parent / "lmtrain"
        cmd = [
            sys.executable,
            str(lmtrain_path / "cli.py"),
            "repo",
            "--query",
            query,
            "--output",
            output_path,
            "--max-repos",
            str(max_repos),
            "--max-files",
            str(max_files),
            "--format",
            format_type,
        ]

        if language:
            cmd.extend(["--language", language])

        # Change to lmtrain directory and run command
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(lmtrain_path))

        if result.returncode == 0:
            return True
        else:
            print(f"lmtrain failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"Error running lmtrain: {e}")
        return False


def fetch_to_corpus(lmtrain_output: str, corpus_output: str) -> int:
    """Write lmtrain JSONL directly to corpus format without conversion."""
    count = 0
    with open(corpus_output, "w", encoding="utf-8") as outfile:
        with open(lmtrain_output, "r", encoding="utf-8") as infile:
            for line_num, line in enumerate(infile):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    
                    # Use raw lmtrain item as-is, just add path/metadata
                    record = {
                        "path": f"example_{count}.json",
                        "size": len(line.encode("utf-8")),
                        "mtime": int(time.time()),
                        "content": json.dumps(item, ensure_ascii=False),
                        "source": item.get("source", "lmtrain"),
                    }
                    outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num + 1}: {e}")
                    continue
    
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Repo obtainer and index builder")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="Clone/read repo and build file index")
    index.add_argument("--source", required=True, help="Git URL or local path")
    index.add_argument("--dest", default="runs/vcs", help="Destination for cloned repos")
    index.add_argument("--branch", default=None, help="Branch name")
    index.add_argument("--depth", type=int, default=None, help="Shallow clone depth")
    index.add_argument("--output", default=None, help="Output JSON path")
    index.add_argument("--max-files", type=int, default=None, help="Max files to index")
    index.add_argument("--ignore", action="append", default=[], help="Extra ignore names")
    index.add_argument("--include-hidden", action="store_true", help="Include dotfiles")
    index.add_argument("--hash", action="store_true", help="Compute sha1 for files")

    export = sub.add_parser("export", help="Export repo to training corpus (jsonl)")
    export.add_argument("--source", required=True, help="Git URL or local path")
    export.add_argument("--dest", default="runs/vcs", help="Destination for cloned repos")
    export.add_argument("--branch", default=None, help="Branch name")
    export.add_argument("--depth", type=int, default=None, help="Shallow clone depth")
    export.add_argument("--output", default=None, help="Output JSONL path")
    export.add_argument("--max-files", type=int, default=None, help="Max files to export")
    export.add_argument("--max-bytes", type=int, default=200_000, help="Max bytes per file")
    export.add_argument("--ignore", action="append", default=[], help="Extra ignore names")
    export.add_argument("--include-hidden", action="store_true", help="Include dotfiles")
    export.add_argument(
        "--format", default="jsonl", choices=["jsonl", "map", "dataset"], help="Export format"
    )
    export.add_argument("--max-lines", type=int, default=200, help="Max lines per chunk (map)")
    export.add_argument("--max-chars", type=int, default=8000, help="Max chars per chunk (map)")

    # Web search command using lmtrain
    web = sub.add_parser("web", help="Search web for code examples using lmtrain")
    web.add_argument("--query", "-q", required=True, help="Search query for code examples")
    web.add_argument("--output", "-o", default=None, help="Output JSONL path")
    web.add_argument("--language", "-l", help="Programming language filter")
    web.add_argument(
        "--format",
        "-f",
        default="instruction",
        choices=["instruction", "completion", "chat"],
        help="Output format for training data",
    )
    web.add_argument(
        "--max-examples", "-m", type=int, default=100, help="Maximum number of examples"
    )
    web.add_argument("--dataset", "-d", default=None, help="Convert to dataset after search")
    web.add_argument("--include-hidden", action="store_true", help="Include dotfiles")

    # GitHub search command using lmtrain
    github = sub.add_parser("github", help="Search GitHub repositories using lmtrain")
    github.add_argument("--query", "-q", required=True, help="Search query for repositories")
    github.add_argument("--output", "-o", default=None, help="Output JSONL path")
    github.add_argument("--language", "-l", help="Programming language filter")
    github.add_argument(
        "--format",
        "-f",
        default="instruction",
        choices=["instruction", "completion", "chat"],
        help="Output format for training data",
    )
    github.add_argument(
        "--max-repos", "-r", type=int, default=10, help="Maximum number of repositories"
    )
    github.add_argument("--max-files", type=int, default=20, help="Maximum files per repository")
    github.add_argument("--dataset", "-d", default=None, help="Convert to dataset after search")
    github.add_argument("--include-hidden", action="store_true", help="Include dotfiles")

    args = parser.parse_args()

    source = args.source
    if source.startswith(("http://", "https://")) or source.endswith(".git"):
        repo_path = run_git_clone(source, Path(args.dest), args.branch, args.depth)
    else:
        repo_path = Path(source).expanduser().resolve()
        if not repo_path.exists():
            raise SystemExit(f"Source path not found: {repo_path}")

    ignores = set(DEFAULT_IGNORES) | set(args.ignore)
    if args.command == "index":
        tree = build_tree(repo_path, ignores, args.max_files, args.include_hidden, args.hash)
        output = (
            Path(args.output) if args.output else Path("runs/repo_index") / f"{repo_path.name}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": source,
            "repo_path": str(repo_path),
            "generated_at": int(time.time()),
            "index": tree,
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Indexed {len(tree['files'])} files (truncated={tree['truncated']})")
        print(f"Wrote index: {output}")
        return

    if args.command == "export":
        if args.format == "jsonl":
            output = (
                Path(args.output)
                if args.output
                else Path("runs/repo_corpus") / f"{repo_path.name}.jsonl"
            )
            count = export_corpus(
                repo_path,
                ignores,
                args.include_hidden,
                args.max_files,
                args.max_bytes,
                output,
            )
            print(f"Exported {count} files")
            print(f"Wrote corpus: {output}")
            return
        if args.format == "map":
            output = (
                Path(args.output)
                if args.output
                else Path("runs/repo_map") / f"{repo_path.name}.jsonl"
            )
            count = export_map(
                repo_path,
                ignores,
                args.include_hidden,
                args.max_files,
                args.max_bytes,
                args.max_lines,
                args.max_chars,
                output,
            )
            print(f"Exported {count} files")
            print(f"Wrote map: {output}")
            return
        if args.format == "dataset":
            output = (
                Path(args.output)
                if args.output
                else Path("runs/repo_corpus") / f"{repo_path.name}.jsonl"
            )
            count = export_corpus(
                repo_path,
                ignores,
                args.include_hidden,
                args.max_files,
                args.max_bytes,
                output,
            )
            print(f"Exported {count} files")
            print(f"Wrote corpus: {output}")
            dataset_name = repo_path.name.replace(".", "_")
            import sys as _sys

            from packages.core.src.scripts.corpus_to_dataset import main as corpus_main

            _sys.argv = [
                _sys.argv[0],
                "--input",
                str(output),
                "--dataset",
                dataset_name,
            ]
            corpus_main()
            return

    # Handle web search command
    if args.command == "web":
        output_path = args.output or "runs/lmtrain_web.jsonl"

        print(f"Searching web for: {args.query}")
        if not run_lmtrain_web_search(
            args.query, output_path, args.max_examples, args.language, args.format
        ):
            print("Web search failed!")
            return

        print(f"Web search completed: {output_path}")

        # Convert to corpus format
        corpus_path = output_path.replace(".jsonl", "_corpus.jsonl")
        count = fetch_to_corpus(output_path, corpus_path)
        print(f"Converted {count} examples to corpus format: {corpus_path}")

        # Convert to dataset if requested
        if args.dataset:
            dataset_name = args.dataset
            import sys as _sys

            from packages.core.src.scripts.corpus_to_dataset import main as corpus_main

            _sys.argv = [
                _sys.argv[0],
                "--input",
                corpus_path,
                "--dataset",
                dataset_name,
            ]
            corpus_main()

        # Clean up temporary lmtrain output
        Path(output_path).unlink(missing_ok=True)
        return

    # Handle github search command
    if args.command == "github":
        output_path = args.output or "runs/lmtrain_github.jsonl"

        print(f"Searching GitHub for: {args.query}")
        if not run_lmtrain_github_search(
            args.query, output_path, args.max_repos, args.max_files, args.language, args.format
        ):
            print("GitHub search failed!")
            return

        print(f"GitHub search completed: {output_path}")

        # Convert to corpus format
        corpus_path = output_path.replace(".jsonl", "_corpus.jsonl")
        count = fetch_to_corpus(output_path, corpus_path)
        print(f"Converted {count} examples to corpus format: {corpus_path}")

        # Convert to dataset if requested
        if args.dataset:
            dataset_name = args.dataset
            import sys as _sys

            from packages.core.src.scripts.corpus_to_dataset import main as corpus_main

            _sys.argv = [
                _sys.argv[0],
                "--input",
                corpus_path,
                "--dataset",
                dataset_name,
            ]
            corpus_main()

        # Clean up temporary lmtrain output
        Path(output_path).unlink(missing_ok=True)
        return


if __name__ == "__main__":
    main()
