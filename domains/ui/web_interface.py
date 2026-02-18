"""
Web Interface - Ported from recovered web_interface.py
Flask-based web application for managing datasets
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import threading
import queue


background_tasks = queue.Queue()
task_results = {}


class DatasetWebManager:
    """Web-based dataset management."""
    
    def __init__(self, datasets_dir: str = "datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(exist_ok=True)
    
    def get_datasets(self) -> List[Dict]:
        """Get all datasets with metadata."""
        datasets = []
        
        for item in self.datasets_dir.iterdir():
            if item.is_dir():
                meta_file = item / "meta.pkl"
                train_file = item / "train.bin"
                val_file = item / "val.bin"
                
                dataset_info = {
                    "name": item.name,
                    "path": str(item),
                    "has_meta": meta_file.exists(),
                    "has_train": train_file.exists(),
                    "has_val": val_file.exists(),
                }
                
                datasets.append(dataset_info)
        
        return datasets
    
    def create_dataset(self, name: str, content: str) -> Dict:
        """Create a new dataset."""
        dataset_path = self.datasets_dir / name
        dataset_path.mkdir(exist_ok=True)
        
        input_file = dataset_path / "input.txt"
        input_file.write_text(content)
        
        return {"name": name, "path": str(dataset_path), "status": "created"}
    
    def delete_dataset(self, name: str) -> bool:
        """Delete a dataset."""
        import shutil
        dataset_path = self.datasets_dir / name
        
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
            return True
        return False


class WebInterface:
    """Flask web interface for dataset management."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.host = host
        self.port = port
        self.manager = DatasetWebManager()
        self.app = None
    
    def create_app(self):
        """Create Flask app."""
        from flask import Flask, render_template, request, jsonify, redirect, url_for
        
        app = Flask(__name__)
        
        @app.route("/")
        def index():
            return {"message": "SloughGPT Web Interface", "version": "2.0"}
        
        @app.route("/api/datasets")
        def list_datasets():
            return jsonify(self.manager.get_datasets())
        
        @app.route("/api/datasets", methods=["POST"])
        def create_dataset():
            data = request.json
            result = self.manager.create_dataset(data.get("name"), data.get("content", ""))
            return jsonify(result)
        
        @app.route("/api/datasets/<name>", methods=["DELETE"])
        def delete_dataset(name):
            success = self.manager.delete_dataset(name)
            return jsonify({"success": success})
        
        self.app = app
        return app
    
    def run(self):
        """Run the web interface."""
        if not self.app:
            self.create_app()
        
        self.app.run(host=self.host, port=self.port)


__all__ = ["WebInterface", "DatasetWebManager"]
