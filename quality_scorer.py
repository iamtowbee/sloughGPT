#!/usr/bin/env python3
"""
Quality Scoring System for Datasets

Automated dataset quality evaluation with scoring metrics and recommendations.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any
import pickle
import math


class DatasetQualityScorer:
    """Evaluates and scores dataset quality across multiple dimensions."""
    
    def __init__(self):
        self.scores = {
            "content_quality": 0,
            "diversity": 0,
            "completeness": 0,
            "formatting": 0,
            "size": 0,
            "overall": 0
        }
        
        self.weights = {
            "content_quality": 0.3,
            "diversity": 0.2,
            "completeness": 0.2,
            "formatting": 0.15,
            "size": 0.15
        }
        
        self.recommendations = []
    
    def score_dataset(self, dataset_name: str) -> Dict:
        """Score a dataset across quality dimensions."""
        dataset_path = Path(f"datasets/{dataset_name}")
        
        if not dataset_path.exists():
            return {"error": f"Dataset {dataset_name} not found"}
        
        # Reset scores
        self.scores = {k: 0 for k in self.scores}
        self.recommendations = []
        
        # Check required files
        meta_file = dataset_path / "meta.pkl"
        train_file = dataset_path / "train.bin"
        val_file = dataset_path / "val.bin"
        input_file = dataset_path / "input.txt"
        
        if not all([meta_file.exists(), train_file.exists(), val_file.exists(), input_file.exists()]):
            self.recommendations.append("Dataset missing required files (train.bin, val.bin, meta.pkl, input.txt)")
            return self._generate_report(dataset_name)
        
        # Load metadata
        try:
            with open(meta_file, 'rb') as f:
                meta = pickle.load(f)
        except Exception as e:
            return {"error": f"Could not load metadata: {e}"}
        
        # Score each dimension
        self._score_content_quality(input_file)
        self._score_diversity(meta)
        self._score_completeness(meta)
        self._score_formatting(input_file)
        self._score_size(meta)
        
        # Calculate overall score
        overall = 0
        for dimension, weight in self.weights.items():
            overall += self.scores[dimension] * weight
        
        self.scores["overall"] = round(overall, 2)
        
        return self._generate_report(dataset_name)
    
    def _score_content_quality(self, input_file: Path):
        """Score content quality."""
        try:
            with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            self.scores["content_quality"] = 0
            self.recommendations.append("Cannot read input file")
            return
        
        if len(content.strip()) == 0:
            self.scores["content_quality"] = 0
            self.recommendations.append("Dataset is empty")
            return
        
        score = 50  # Start at 50 (average)
        
        # Length score
        if len(content) < 1000:
            score -= 20
            self.recommendations.append("Dataset too small (< 1000 characters)")
        elif len(content) > 1000000:
            score += 10
            self.recommendations.append("Large dataset (> 1M characters)")
        
        # Repeated content penalty
        lines = content.splitlines()
        unique_lines = set(line.strip() for line in lines if line.strip())
        
        if len(unique_lines) / len(lines) < 0.8:
            score -= 15
            self.recommendations.append(f"High duplicate content ({len(lines) - len(unique_lines)} duplicates)")
        
        # Text quality checks
        # Check for common patterns
        patterns = [
            (r'\s+', "Multiple consecutive spaces"),
            (r'\n{3,}', "Multiple consecutive empty lines"),
            (r'[A-Z]{5,}', "All caps sequences"),
            (r'[a-z]{5,}', "All lowercase sequences"),
            (r'.{3,}', "Multiple consecutive dots"),
            (r'[!]{3,}', "Multiple exclamation marks")
        ]
        
        for pattern, description in patterns:
            if re.search(pattern, content):
                score -= 5
                self.recommendations.append(f"Text quality issue: {description}")
        
        # Language diversity (simple check)
        if content.count(' ') / len(content) > 0.7:  # Not text-heavy
            score += 10
            self.recommendations.append("Good word/character ratio")
        elif content.count(' ') / len(content) < 0.1:  # Too space-heavy
            score -= 10
            self.recommendations.append("Low word diversity")
        
        # Encoding issues
        try:
            content.encode('utf-8')
        except UnicodeEncodeError:
            score -= 25
            self.recommendations.append("Text encoding issues detected")
        
        self.scores["content_quality"] = max(0, min(100, score))
    
    def _score_diversity(self, meta: Dict):
        """Score vocabulary and character diversity."""
        vocab_size = meta.get("vocab_size", 0)
        train_tokens = meta.get("train_tokens", 0)
        
        if vocab_size == 0:
            self.scores["diversity"] = 0
            return
        
        score = 50
        
        # Vocabulary size scoring
        if vocab_size < 50:
            score -= 30
            self.recommendations.append("Very small vocabulary (< 50)")
        elif vocab_size < 200:
            score -= 10
            self.recommendations.append("Small vocabulary (< 200)")
        elif vocab_size > 50000:
            score += 15
            self.recommendations.append("Large vocabulary (> 50K)")
        elif vocab_size > 100000:
            score -= 10
            self.recommendations.append("Very large vocabulary may cause memory issues")
        
        # Token efficiency
        if train_tokens > 0:
            unique_chars = vocab_size
            token_efficiency = train_tokens / unique_chars
            
            if token_efficiency < 2:
                score -= 10
                self.recommendations.append("Low token efficiency")
            elif token_efficiency > 10:
                score += 10
                self.recommendations.append("Good token efficiency")
        
        self.scores["diversity"] = max(0, min(100, score))
    
    def _score_completeness(self, meta: Dict):
        """Score dataset completeness."""
        total_tokens = meta.get("train_tokens", 0) + meta.get("val_tokens", 0)
        total_chars = meta.get("total_characters", 0)
        
        if total_tokens == 0:
            self.scores["completeness"] = 0
            self.recommendations.append("No tokens found")
            return
        
        score = 50
        
        # Train/validation split
        if total_tokens > 0:
            val_ratio = meta.get("val_tokens", 0) / total_tokens
            if val_ratio < 0.05:
                score -= 15
                self.recommendations.append(f"Very small validation set ({val_ratio:.1%})")
            elif val_ratio > 0.3:
                score -= 5
                self.recommendations.append(f"Large validation set ({val_ratio:.1%})")
            elif 0.1 <= val_ratio <= 0.2:
                score += 5
                self.recommendations.append(f"Good validation split ({val_ratio:.1%})")
        
        # Character coverage
        if total_chars > 0:
            chars_per_token = total_chars / total_tokens
            if chars_per_token < 1:
                score -= 10
                self.recommendations.append("Character coverage seems low")
            elif chars_per_token > 10:
                score += 5
                self.recommendations.append("Good character density")
        
        self.scores["completeness"] = max(0, min(100, score))
    
    def _score_formatting(self, input_file: Path):
        """Score text formatting."""
        try:
            with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            self.scores["formatting"] = 0
            self.recommendations.append("Cannot read file for formatting check")
            return
        
        if len(content) == 0:
            self.scores["formatting"] = 50
            return
        
        score = 50
        
        # Line ending consistency
        if '\r\n' in content:
            score -= 10
            self.recommendations.append("Mixed line endings (Windows/Unix)")
        
        # Trailing whitespace
        lines = content.splitlines()
        lines_with_trailing = [line for line in lines if line.endswith(' ') or line.endswith('\t')]
        
        if len(lines_with_trailing) / len(lines) > 0.1:
            score -= 10
            self.recommendations.append("Many lines with trailing whitespace")
        
        # Tab usage consistency
        if '\t' in content:
            tab_lines = [line for line in lines if '\t' in line]
            if 0 < len(tab_lines) / len(lines) < 0.5:  # Some but not too many
                score += 5
                self.recommendations.append("Consistent tab usage for structure")
            else:
                score -= 10
                self.recommendations.append("Inconsistent tab usage")
        
        # Special characters
        control_chars = sum(1 for c in content if ord(c) < 32)
        if control_chars / len(content) > 0.01:
            score -= 10
            self.recommendations.append("Contains control characters")
        
        self.scores["formatting"] = max(0, min(100, score))
    
    def _score_size(self, meta: Dict):
        """Score dataset size appropriateness."""
        train_tokens = meta.get("train_tokens", 0)
        
        if train_tokens == 0:
            self.scores["size"] = 0
            self.recommendations.append("No training tokens")
            return
        
        score = 50
        
        # Size categories
        if train_tokens < 1000:
            score -= 30
            self.recommendations.append("Very small dataset (< 1K tokens)")
        elif train_tokens < 10000:
            score -= 10
            self.recommendations.append("Small dataset (< 10K tokens)")
        elif train_tokens > 1000000:
            score += 10
            self.recommendations.append("Large dataset (> 1M tokens)")
        elif train_tokens > 10000000:
            score -= 10
            self.recommendations.append("Very large dataset may cause training issues")
        
        # Size efficiency (for character-level models)
        total_chars = meta.get("total_characters", 0)
        if train_tokens > 0 and total_chars > 0:
            chars_per_token = total_chars / train_tokens
            if 1.0 <= chars_per_token <= 3.0:  # Good range for character models
                score += 10
                self.recommendations.append("Optimal character/token ratio")
            elif chars_per_token > 8.0:
                score -= 10
                self.recommendations.append("Inefficient character/token ratio")
        
        self.scores["size"] = max(0, min(100, score))
    
    def _generate_report(self, dataset_name: str) -> Dict:
        """Generate quality report."""
        report = {
            "dataset": dataset_name,
            "timestamp": str(Path(f"datasets/{dataset_name}").stat().st_mtime),
            "scores": self.scores.copy(),
            "overall_score": self.scores["overall"],
            "grade": self._get_grade(self.scores["overall"]),
            "recommendations": self.recommendations,
            "dimension_scores": {
                "Content Quality": self.scores["content_quality"],
                "Diversity": self.scores["diversity"],
                "Completeness": self.scores["completeness"],
                "Formatting": self.scores["formatting"],
                "Size": self.scores["size"]
            }
        }
        
        # Add grade-specific recommendations
        grade = self._get_grade(self.scores["overall"])
        if grade in ["Poor", "Fair"]:
            report["recommendations"].append("Consider dataset preprocessing and cleaning before training")
        elif grade == "Good":
            report["recommendations"].append("Dataset is suitable for training with minor improvements")
        elif grade == "Excellent":
            report["recommendations"].append("High-quality dataset ready for production training")
        
        return report
    
    def _get_grade(self, score: float) -> str:
        """Convert score to grade."""
        if score >= 85:
            return "Excellent"
        elif score >= 70:
            return "Good"
        elif score >= 50:
            return "Fair"
        else:
            return "Poor"


def main():
    """Command line interface for dataset quality scoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset Quality Scoring System")
    parser.add_argument('dataset', help='Dataset name to score')
    parser.add_argument('--output', help='Output file for report (JSON)')
    parser.add_argument('--all', action='store_true', help='Score all datasets')
    parser.add_argument('--threshold', type=float, default=50.0, help='Quality threshold for datasets')
    
    args = parser.parse_args()
    
    if args.all:
        # Score all datasets
        datasets_dir = Path("datasets")
        all_reports = []
        
        for item in datasets_dir.iterdir():
            if item.is_dir():
                scorer = DatasetQualityScorer()
                report = scorer.score_dataset(item.name)
                if "error" not in report:
                    all_reports.append(report)
        
        # Filter by threshold
        filtered_reports = [r for r in all_reports if r["overall_score"] >= args.threshold]
        
        print(f"Dataset Quality Report (threshold: {args.threshold})")
        print("=" * 50)
        
        for report in sorted(filtered_reports, key=lambda x: x["overall_score"], reverse=True):
            print(f"\n📊 {report['dataset']}")
            print(f"   Score: {report['overall_score']}/100 ({report['grade']})")
            quality_scores = report['dimension_scores']
            print(f"   Quality: {quality_scores['Content Quality']:.0f} | "
                  f"{quality_scores['Diversity']:.0f} | "
                  f"{quality_scores['Completeness']:.0f} | "
                  f"{quality_scores['Formatting']:.0f} | "
                  f"{quality_scores['Size']:.0f}")
            
            if report["recommendations"]:
                print("   Issues:")
                for rec in report["recommendations"][:3]:  # Top 3 issues
                    print(f"     • {rec}")
        
        print(f"\n📈 {len(filtered_reports)} datasets passed quality threshold")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(filtered_reports, f, indent=2)
            print(f"\n💾 Report saved to: {args.output}")
    
    else:
        # Score single dataset
        scorer = DatasetQualityScorer()
        report = scorer.score_dataset(args.dataset)
        
        print(f"📊 Dataset Quality Report: {args.dataset}")
        print("=" * 50)
        print(f"Overall Score: {report['overall_score']}/100 ({report['grade']})")
        
        print("\n📈 Dimension Scores:")
        for dimension, score in report["dimension_scores"].items():
            status = "✅" if score >= 70 else "⚠️" if score >= 50 else "❌"
            print(f"  {status} {dimension}: {score}/100")
        
        if report["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in report["recommendations"]:
                print(f"  • {rec}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n💾 Report saved to: {args.output}")


if __name__ == '__main__':
    main()