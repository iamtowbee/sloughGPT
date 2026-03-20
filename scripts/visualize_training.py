#!/usr/bin/env python3
"""
Training Visualization Script
Visualize training metrics from experiment logs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from pathlib import Path


def plot_training_metrics(metrics_file: str, output_file: str = None):
    """Plot training metrics from JSON file."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False

    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Install with: pip install matplotlib")
        print_training_table(metrics_file)
        return

    with open(metrics_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'metrics' in data:
        metrics = data['metrics']
    elif isinstance(data, list):
        metrics = data
    else:
        metrics = [data]

    steps = [m.get('step', i) for i, m in enumerate(metrics)]
    losses = [m.get('loss', 0) for m in metrics]
    learning_rates = [m.get('learning_rate', 0) for m in metrics]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(steps, losses, 'b-', linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    if any(lr != 0 for lr in learning_rates):
        ax2.plot(steps, learning_rates, 'r-', linewidth=2)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to: {output_file}")
    else:
        plt.savefig('training_metrics.png')
        print("Plot saved to: training_metrics.png")


def print_training_table(metrics_file: str):
    """Print training metrics as ASCII table."""
    with open(metrics_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'metrics' in data:
        metrics = data['metrics']
    elif isinstance(data, list):
        metrics = data
    else:
        metrics = [data]

    print("\nTraining Metrics")
    print("=" * 50)
    print(f"{'Step':<10} {'Loss':<15} {'LR':<15}")
    print("-" * 50)

    for i, m in enumerate(metrics[:50]):
        step = m.get('step', i)
        loss = m.get('loss', 0)
        lr = m.get('learning_rate', 0)
        print(f"{step:<10} {loss:<15.6f} {lr:<15.6f}")

    if len(metrics) > 50:
        print(f"... ({len(metrics) - 50} more rows)")


def export_csv(metrics_file: str, output_file: str):
    """Export metrics to CSV."""
    import csv

    with open(metrics_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'metrics' in data:
        metrics = data['metrics']
    elif isinstance(data, list):
        metrics = data
    else:
        metrics = [data]

    if not metrics:
        print("No metrics to export")
        return

    keys = list(metrics[0].keys())

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)

    print(f"Exported {len(metrics)} rows to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument("metrics", nargs="?", default="experiments/metrics.json",
                       help="Metrics JSON file")
    parser.add_argument("--output", "-o", help="Output file for plot/CSV")
    parser.add_argument("--format", "-f", choices=["plot", "csv", "table"],
                       default="plot", help="Output format")

    args = parser.parse_args()

    if not Path(args.metrics).exists():
        print(f"Error: Metrics file not found: {args.metrics}")
        return

    if args.format == "plot":
        plot_training_metrics(args.metrics, args.output)
    elif args.format == "csv":
        output = args.output or "metrics.csv"
        export_csv(args.metrics, output)
    elif args.format == "table":
        print_training_table(args.metrics)


if __name__ == "__main__":
    main()
