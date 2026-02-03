#!/usr/bin/env python3
"""dset benchmarking & stats utility

Add statistics and performance tracking to dataset fetching.
"""

import json
import time
from pathlib import Path
from typing import Dict, List


class DatasetStats:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.example_count = 0
        self.source_type = None
        self.query = None
        self.output_size = 0

    def start(self, source: str, query: str) -> None:
        self.start_time = time.time()
        self.source_type = source
        self.query = query

    def finish(self, example_count: int, output_path: Path) -> None:
        self.end_time = time.time()
        self.example_count = example_count
        self.output_size = output_path.stat().st_size if output_path.exists() else 0

    def duration(self) -> float:
        return (self.end_time or 0) - (self.start_time or 0)

    def summary(self) -> Dict:
        return {
            "source": self.source_type,
            "query": self.query,
            "examples": self.example_count,
            "output_size_bytes": self.output_size,
            "duration_seconds": self.duration(),
            "examples_per_second": self.example_count / max(self.duration(), 0.001),
        }

    def print_summary(self) -> None:
        stats = self.summary()
        print(f"\n📊 Performance Summary:")
        print(f"   ⏱️  Duration: {stats['duration_seconds']:.2f}s")
        print(f"   📝 Examples: {stats['examples']}")
        print(f"   💾 Size: {stats['output_size_bytes'] / (1024*1024):.2f} MB")
        print(f"   ⚡ Rate: {stats['examples_per_second']:.1f} examples/s")


def analyze_dataset(file_path: Path) -> Dict:
    """Analyze existing dataset for insights."""
    if not file_path.exists():
        return {"error": "File not found"}
    
    examples = []
    total_size = 0
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    examples.append(item)
                    total_size += len(line.encode("utf-8"))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return {"error": str(e)}
    
    # Calculate statistics
    languages = {}
    for example in examples:
        content = example.get("content", "")
        if isinstance(content, str):
            # Simple language detection by file extensions
            if "python" in content.lower():
                languages["python"] = languages.get("python", 0) + 1
            elif "javascript" in content.lower() or "react" in content.lower():
                languages["javascript"] = languages.get("javascript", 0) + 1
            elif "typescript" in content.lower():
                languages["typescript"] = languages.get("typescript", 0) + 1
    
    return {
        "total_examples": len(examples),
        "total_size_mb": total_size / (1024 * 1024),
        "languages": languages,
        "avg_example_size": total_size / max(len(examples), 1),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 stats.py <dataset_file.jsonl>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    stats = analyze_dataset(file_path)
    if "error" in stats:
        print(f"❌ Error: {stats['error']}")
    else:
        print(f"📊 Dataset Analysis for {file_path.name}:")
        print(f"   📝 Total examples: {stats['total_examples']}")
        print(f"   💾 Total size: {stats['total_size_mb']:.2f} MB")
        print(f"   📏 Avg example size: {stats['avg_example_size']} bytes")
        if stats['languages']:
            print(f"   🌐 Languages detected: {stats['languages']}")