#!/usr/bin/env python3
"""
Advanced Dataset Features - Versioning, Validation, and Quality Control

Adds enterprise-level dataset management capabilities.
"""

import os
import json
import hashlib
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil
import subprocess


class DatasetValidator:
    """Validate dataset quality and integrity."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.validation_results = {}
    
    def validate_structure(self) -> Dict:
        """Validate dataset file structure."""
        required_files = ["input.txt", "prepare.py"]
        optional_files = ["train.bin", "val.bin", "meta.pkl"]
        
        results = {
            "required_files": {},
            "optional_files": {},
            "structure_valid": True
        }
        
        for file_name in required_files:
            file_path = self.dataset_path / file_name
            exists = file_path.exists()
            results["required_files"][file_name] = {
                "exists": exists,
                "path": str(file_path)
            }
            if not exists:
                results["structure_valid"] = False
        
        for file_name in optional_files:
            file_path = self.dataset_path / file_name
            results["optional_files"][file_name] = {
                "exists": file_path.exists(),
                "path": str(file_path),
                "size_mb": file_path.stat().st_size / (1024*1024) if file_path.exists() else 0
            }
        
        return results
    
    def validate_content(self) -> Dict:
        """Validate dataset content quality."""
        input_file = self.dataset_path / "input.txt"
        
        if not input_file.exists():
            return {"valid": False, "error": "input.txt not found"}
        
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        results = {
            "character_count": len(content),
            "word_count": len(content.split()),
            "line_count": len(content.splitlines()),
            "unique_characters": len(set(content)),
            "empty_lines": sum(1 for line in content.splitlines() if not line.strip()),
            "unicode_issues": self._check_unicode_issues(content),
            "duplicate_lines": self._check_duplicate_lines(content)
        }
        
        # Quality scores
        results["quality_scores"] = {
            "content_ratio": max(0, (len(content.strip()) / len(content))) if content else 0,
            "character_diversity": len(set(content)) / len(content) if content else 0,
            "line_completeness": 1 - (results["empty_lines"] / results["line_count"]) if results["line_count"] > 0 else 0
        }
        
        results["overall_quality"] = sum(results["quality_scores"].values()) / len(results["quality_scores"])
        
        return results
    
    def _check_unicode_issues(self, content: str) -> List[str]:
        """Check for unicode encoding issues."""
        issues = []
        
        try:
            content.encode('utf-8')
        except UnicodeEncodeError as e:
            issues.append(f"Unicode encoding error: {e}")
        
        # Check for problematic characters
        problematic_chars = ['\x00', '\ufffd']  # Null and replacement characters
        for char in problematic_chars:
            if char in content:
                issues.append(f"Contains problematic character: {repr(char)}")
        
        return issues
    
    def _check_duplicate_lines(self, content: str) -> Dict:
        """Check for duplicate lines."""
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        unique_lines = set(lines)
        
        return {
            "total_lines": len(lines),
            "unique_lines": len(unique_lines),
            "duplicate_count": len(lines) - len(unique_lines),
            "duplicate_ratio": (len(lines) - len(unique_lines)) / len(lines) if lines else 0
        }
    
    def validate_processed_data(self) -> Dict:
        """Validate processed training data (if exists)."""
        results = {"train_bin": {}, "val_bin": {}, "meta_pkl": {}}
        
        # Check train.bin
        train_file = self.dataset_path / "train.bin"
        if train_file.exists():
            try:
                import numpy as np
                train_data = np.fromfile(train_file, dtype=np.uint16)
                results["train_bin"] = {
                    "exists": True,
                    "token_count": len(train_data),
                    "file_size_mb": train_file.stat().st_size / (1024*1024),
                    "data_valid": True
                }
            except Exception as e:
                results["train_bin"] = {
                    "exists": True,
                    "error": str(e),
                    "data_valid": False
                }
        else:
            results["train_bin"]["exists"] = False
        
        # Check val.bin
        val_file = self.dataset_path / "val.bin"
        if val_file.exists():
            try:
                import numpy as np
                val_data = np.fromfile(val_file, dtype=np.uint16)
                results["val_bin"] = {
                    "exists": True,
                    "token_count": len(val_data),
                    "file_size_mb": val_file.stat().st_size / (1024*1024),
                    "data_valid": True
                }
            except Exception as e:
                results["val_bin"] = {
                    "exists": True,
                    "error": str(e),
                    "data_valid": False
                }
        else:
            results["val_bin"]["exists"] = False
        
        # Check meta.pkl
        meta_file = self.dataset_path / "meta.pkl"
        if meta_file.exists():
            try:
                with open(meta_file, 'rb') as f:
                    meta_data = pickle.load(f)
                
                results["meta_pkl"] = {
                    "exists": True,
                    "vocab_size": meta_data.get("vocab_size", 0),
                    "train_tokens": meta_data.get("train_tokens", 0),
                    "val_tokens": meta_data.get("val_tokens", 0),
                    "data_valid": True
                }
            except Exception as e:
                results["meta_pkl"] = {
                    "exists": True,
                    "error": str(e),
                    "data_valid": False
                }
        else:
            results["meta_pkl"]["exists"] = False
        
        return results
    
    def full_validation(self) -> Dict:
        """Run complete validation suite."""
        validation = {
            "dataset_name": self.dataset_path.name,
            "timestamp": datetime.now().isoformat(),
            "structure": self.validate_structure(),
            "content": self.validate_content(),
            "processed_data": self.validate_processed_data(),
            "overall_valid": True
        }
        
        # Determine overall validity
        if not validation["structure"]["structure_valid"]:
            validation["overall_valid"] = False
        
        if validation["content"].get("overall_quality", 0) < 0.5:
            validation["overall_valid"] = False
        
        if not validation["processed_data"]["train_bin"].get("data_valid", False):
            validation["overall_valid"] = False
        
        return validation


class DatasetVersioning:
    """Dataset versioning and management."""
    
    def __init__(self, datasets_dir: str = "datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.versions_dir = Path("dataset_versions")
        self.versions_dir.mkdir(exist_ok=True)
    
    def create_version(self, dataset_name: str, version_tag: str = None, message: str = "") -> str:
        """Create a new version of a dataset."""
        dataset_path = self.datasets_dir / dataset_name
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} not found")
        
        # Generate version tag if not provided
        if not version_tag:
            version_tag = f"v{int(time.time())}"
        
        version_dir = self.versions_dir / dataset_name / version_tag
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy dataset files
        for file_path in dataset_path.iterdir():
            if file_path.is_file():
                shutil.copy2(file_path, version_dir / file_path.name)
        
        # Create version metadata
        version_info = {
            "dataset_name": dataset_name,
            "version": version_tag,
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "files": [f.name for f in dataset_path.iterdir() if f.is_file()],
            "checksums": self._calculate_checksums(dataset_path)
        }
        
        with open(version_dir / "version.json", 'w') as f:
            json.dump(version_info, f, indent=2)
        
        print(f"📦 Created version {version_tag} for dataset {dataset_name}")
        return str(version_dir)
    
    def list_versions(self, dataset_name: str = None) -> List[Dict]:
        """List available versions."""
        versions = []
        
        if dataset_name:
            version_path = self.versions_dir / dataset_name
            if version_path.exists():
                for version_dir in version_path.iterdir():
                    if version_dir.is_dir():
                        version_info = self._load_version_info(version_dir)
                        versions.append(version_info)
        else:
            for dataset_dir in self.versions_dir.iterdir():
                if dataset_dir.is_dir():
                    for version_dir in dataset_dir.iterdir():
                        if version_dir.is_dir():
                            version_info = self._load_version_info(version_dir)
                            versions.append(version_info)
        
        return sorted(versions, key=lambda x: x.get("timestamp", ""))
    
    def restore_version(self, dataset_name: str, version_tag: str) -> bool:
        """Restore a dataset to a specific version."""
        version_dir = self.versions_dir / dataset_name / version_tag
        
        if not version_dir.exists():
            print(f"❌ Version {version_tag} not found for dataset {dataset_name}")
            return False
        
        dataset_path = self.datasets_dir / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Backup current version
        if any(dataset_path.iterdir()):
            backup_tag = f"backup_{int(time.time())}"
            self.create_version(dataset_name, backup_tag, "Automatic backup before restore")
        
        # Restore version files
        for file_path in version_dir.iterdir():
            if file_path.is_file() and file_path.name != "version.json":
                shutil.copy2(file_path, dataset_path / file_path.name)
        
        print(f"🔄 Restored dataset {dataset_name} to version {version_tag}")
        return True
    
    def compare_versions(self, dataset_name: str, version1: str, version2: str) -> Dict:
        """Compare two versions of a dataset."""
        v1_dir = self.versions_dir / dataset_name / version1
        v2_dir = self.versions_dir / dataset_name / version2
        
        if not v1_dir.exists() or not v2_dir.exists():
            return {"error": "One or both versions not found"}
        
        v1_info = self._load_version_info(v1_dir)
        v2_info = self._load_version_info(v2_dir)
        
        comparison = {
            "dataset_name": dataset_name,
            "version1": v1_info,
            "version2": v2_info,
            "differences": self._calculate_differences(v1_info, v2_info)
        }
        
        return comparison
    
    def _calculate_checksums(self, dataset_path: Path) -> Dict[str, str]:
        """Calculate checksums for all files in dataset."""
        checksums = {}
        
        for file_path in dataset_path.iterdir():
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    content = f.read()
                    checksums[file_path.name] = hashlib.md5(content).hexdigest()
        
        return checksums
    
    def _load_version_info(self, version_dir: Path) -> Dict:
        """Load version information from version directory."""
        version_file = version_dir / "version.json"
        
        if version_file.exists():
            with open(version_file, 'r') as f:
                return json.load(f)
        
        return {
            "version": version_dir.name,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_differences(self, v1_info: Dict, v2_info: Dict) -> Dict:
        """Calculate differences between two versions."""
        differences = {}
        
        # Compare checksums
        v1_checksums = v1_info.get("checksums", {})
        v2_checksums = v2_info.get("checksums", {})
        
        all_files = set(v1_checksums.keys()) | set(v2_checksums.keys())
        
        for file_name in all_files:
            v1_checksum = v1_checksums.get(file_name)
            v2_checksum = v2_checksums.get(file_name)
            
            if v1_checksum != v2_checksum:
                differences[file_name] = {
                    "status": "modified",
                    "v1_checksum": v1_checksum,
                    "v2_checksum": v2_checksum
                }
            elif v1_checksum and not v2_checksum:
                differences[file_name] = {"status": "removed"}
            elif not v1_checksum and v2_checksum:
                differences[file_name] = {"status": "added"}
        
        return differences


def main():
    """Main CLI for advanced dataset features."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced dataset features - versioning, validation")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validation commands
    validate_parser = subparsers.add_parser('validate', help='Validate dataset quality')
    validate_parser.add_argument('dataset', help='Dataset name to validate')
    validate_parser.add_argument('--output', help='Output validation report to file')
    
    # Versioning commands
    version_parser = subparsers.add_parser('version', help='Create dataset version')
    version_parser.add_argument('dataset', help='Dataset name')
    version_parser.add_argument('--tag', help='Version tag (e.g., v1.0.0)')
    version_parser.add_argument('--message', default='', help='Version description')
    
    # List versions
    list_versions_parser = subparsers.add_parser('list-versions', help='List dataset versions')
    list_versions_parser.add_argument('--dataset', help='Dataset name (optional)')
    
    # Restore version
    restore_parser = subparsers.add_parser('restore', help='Restore dataset version')
    restore_parser.add_argument('dataset', help='Dataset name')
    restore_parser.add_argument('version', help='Version tag to restore')
    
    # Compare versions
    compare_parser = subparsers.add_parser('compare', help='Compare dataset versions')
    compare_parser.add_argument('dataset', help='Dataset name')
    compare_parser.add_argument('v1', help='First version')
    compare_parser.add_argument('v2', help='Second version')
    
    args = parser.parse_args()
    
    if args.command == 'validate':
        validator = DatasetValidator(f"datasets/{args.dataset}")
        validation = validator.full_validation()
        
        print(f"🔍 Validation Results for {args.dataset}")
        print("=" * 50)
        print(f"Overall Valid: {'✅' if validation['overall_valid'] else '❌'}")
        print(f"Structure Valid: {'✅' if validation['structure']['structure_valid'] else '❌'}")
        print(f"Content Quality: {validation['content']['overall_quality']:.2%}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(validation, f, indent=2, default=str)
            print(f"\n📊 Report saved to: {args.output}")
    
    elif args.command == 'version':
        versioning = DatasetVersioning()
        version_path = versioning.create_version(args.dataset, args.tag, args.message)
        print(f"📦 Version created: {version_path}")
    
    elif args.command == 'list-versions':
        versioning = DatasetVersioning()
        versions = versioning.list_versions(args.dataset)
        
        print("📋 Available Versions:")
        for version in versions:
            dataset_name = version.get('dataset_name', 'Unknown')
            version_tag = version.get('version', 'Unknown')
            timestamp = version.get('timestamp', 'Unknown')
            message = version.get('message', '')
            
            print(f"  📦 {dataset_name}:{version_tag}")
            print(f"     📅 {timestamp}")
            if message:
                print(f"     📝 {message}")
    
    elif args.command == 'restore':
        versioning = DatasetVersioning()
        success = versioning.restore_version(args.dataset, args.version)
        
        if success:
            print("✅ Version restored successfully")
        else:
            print("❌ Failed to restore version")
    
    elif args.command == 'compare':
        versioning = DatasetVersioning()
        comparison = versioning.compare_versions(args.dataset, args.v1, args.v2)
        
        print(f"🔀 Comparison: {args.dataset} {args.v1} vs {args.v2}")
        print("=" * 50)
        
        differences = comparison.get('differences', {})
        if not differences:
            print("✅ No differences found")
        else:
            for file_name, diff in differences.items():
                status = diff['status']
                icon = {"added": "➕", "removed": "➖", "modified": "🔄"}[status]
                print(f"  {icon} {file_name}: {status}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()