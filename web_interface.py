#!/usr/bin/env python3
"""
Web Interface for Dataset Management - Browser-based UI

Flask-based web application for managing datasets without terminal commands.
"""

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import threading
import queue


app = Flask(__name__)

# Global state for background operations
background_tasks = queue.Queue()
task_results = {}


class DatasetWebManager:
    """Web-based dataset management."""
    
    def __init__(self):
        self.datasets_dir = Path("datasets")
        self.datasets_dir.mkdir(exist_ok=True)
        
    def get_datasets(self) -> List[Dict]:
        """Get all datasets with metadata."""
        datasets = []
        
        for item in self.datasets_dir.iterdir():
            if item.is_dir():
                meta_file = item / "meta.pkl"
                train_file = item / "train.bin"
                
                dataset_info = {
                    "name": item.name,
                    "exists": True,
                    "has_meta": meta_file.exists(),
                    "has_train": train_file.exists(),
                    "size_mb": 0
                }
                
                if meta_file.exists():
                    try:
                        import pickle
                        with open(meta_file, 'rb') as f:
                            meta = pickle.load(f)
                        
                        dataset_info.update({
                            "vocab_size": meta.get("vocab_size", 0),
                            "train_tokens": meta.get("train_tokens", 0),
                            "val_tokens": meta.get("val_tokens", 0),
                            "total_characters": meta.get("total_characters", 0),
                            "source_files": meta.get("source_files", [])
                        })
                        
                        # Calculate size
                        if train_file.exists():
                            dataset_info["size_mb"] = train_file.stat().st_size / (1024 * 1024)
                            
                    except Exception as e:
                        dataset_info["error"] = str(e)
                
                datasets.append(dataset_info)
        
        return sorted(datasets, key=lambda x: x["name"])
    
    def create_dataset(self, name: str, content: str = None, file_data=None) -> Dict:
        """Create new dataset from content or file."""
        try:
            dataset_dir = self.datasets_dir / name
            dataset_dir.mkdir(exist_ok=True)
            
            if file_data:
                # Handle file upload
                if hasattr(file_data, 'filename'):
                    filename = file_data.filename
                    file_data.save(str(dataset_dir / "input.txt"))
                else:
                    # Text content
                    input_file = dataset_dir / "input.txt"
                    with open(input_file, 'w', encoding='utf-8') as f:
                        f.write(content)
            elif content:
                # Direct text input
                input_file = dataset_dir / "input.txt"
                with open(input_file, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Prepare dataset
            cmd = f"python3 create_dataset_fixed.py {name} \"Web UI created dataset\""
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                return {"success": True, "message": "Dataset created successfully"}
            else:
                return {"success": False, "error": result.stderr}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_dataset_preview(self, name: str) -> Dict:
        """Get preview of dataset content."""
        try:
            input_file = self.datasets_dir / name / "input.txt"
            meta_file = self.datasets_dir / name / "meta.pkl"
            
            preview = {"name": name, "exists": False}
            
            if input_file.exists():
                with open(input_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    preview.update({
                        "exists": True,
                        "preview_text": content[:500] + "..." if len(content) > 500 else content,
                        "total_lines": len(content.splitlines()),
                        "total_chars": len(content)
                    })
            
            if meta_file.exists():
                try:
                    import pickle
                    with open(meta_file, 'rb') as f:
                        meta = pickle.load(f)
                    
                    preview.update({
                        "vocab_size": meta.get("vocab_size", 0),
                        "train_tokens": meta.get("train_tokens", 0),
                        "created_at": meta.get("creation_time")
                    })
                except:
                    pass
            
            return preview
            
        except Exception as e:
            return {"error": str(e)}
    
    def delete_dataset(self, name: str) -> Dict:
        """Delete a dataset."""
        try:
            dataset_dir = self.datasets_dir / name
            if dataset_dir.exists():
                import shutil
                shutil.rmtree(dataset_dir)
                return {"success": True, "message": "Dataset deleted successfully"}
            else:
                return {"success": False, "error": "Dataset not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Initialize web manager
web_manager = DatasetWebManager()


@app.route('/')
def index():
    """Main dashboard page."""
    datasets = web_manager.get_datasets()
    return render_template('dashboard.html', datasets=datasets)


@app.route('/api/datasets')
def get_datasets():
    """API endpoint for datasets."""
    return jsonify(web_manager.get_datasets())


@app.route('/api/dataset/<name>')
def get_dataset(name):
    """API endpoint for dataset details."""
    preview = web_manager.get_dataset_preview(name)
    return jsonify(preview)


@app.route('/create', methods=['GET', 'POST'])
def create_dataset():
    """Create dataset page."""
    if request.method == 'POST':
        name = request.form.get('name')
        content = request.form.get('content')
        
        # Handle file upload
        file_data = None
        if 'file' in request.files:
            file_data = request.files['file']
        
        result = web_manager.create_dataset(name, content, file_data)
        
        if result['success']:
            return redirect(url_for('index'))
        else:
            return render_template('create.html', error=result.get('error'))
    
    return render_template('create.html')


@app.route('/dataset/<name>')
def view_dataset(name):
    """View dataset details."""
    preview = web_manager.get_dataset_preview(name)
    return render_template('dataset.html', dataset=preview)


@app.route('/api/dataset/<name>/delete', methods=['POST'])
def delete_dataset(name):
    """Delete dataset."""
    result = web_manager.delete_dataset(name)
    return jsonify(result)


@app.route('/train/<name>')
def train_dataset(name):
    """Trigger training for dataset."""
    cmd = f"python3 train_simple.py {name}"
    
    # Start background task
    task_id = f"train_{int(time.time())}"
    background_tasks.put({"id": task_id, "cmd": cmd})
    
    return jsonify({"task_id": task_id, "message": "Training started"})


@app.route('/api/task/<task_id>')
def get_task_status(task_id):
    """Get status of background task."""
    # This would track actual task status in real implementation
    return jsonify({"task_id": task_id, "status": "running", "progress": 0})


# HTML Templates
@app.route('/templates/dashboard.html')
def dashboard_template():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SloGPT Dataset Management</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: #2563eb; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .header h1 { margin: 0; font-size: 28px; }
        .header p { margin: 10px 0 0 0; opacity: 0.8; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .stat-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }
        .stat-card h3 { margin: 0 0 10px 0; color: #333; }
        .stat-card .number { font-size: 32px; font-weight: bold; color: #2563eb; }
        .datasets { background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .datasets-header { padding: 20px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }
        .datasets h2 { margin: 0; color: #333; }
        .btn { background: #2563eb; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; text-decoration: none; display: inline-block; }
        .btn:hover { background: #1d4ed8; }
        .btn-danger { background: #dc3545; }
        .btn-danger:hover { background: #c82333; }
        .dataset-list { padding: 20px; }
        .dataset-item { padding: 15px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }
        .dataset-item:hover { background: #f8f9fa; }
        .dataset-info h3 { margin: 0 0 5px 0; color: #333; }
        .dataset-info p { margin: 0; color: #666; font-size: 14px; }
        .dataset-stats { display: flex; gap: 20px; margin-top: 10px; }
        .stat { text-align: center; }
        .stat-label { font-size: 12px; color: #666; text-transform: uppercase; }
        .stat-value { font-size: 18px; font-weight: bold; color: #2563eb; }
        .dataset-actions { display: flex; gap: 10px; }
        .actions { display: flex; gap: 10px; }
        .btn-small { padding: 6px 12px; font-size: 12px; }
        .empty { text-align: center; padding: 60px 20px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 SloGPT Dataset Management</h1>
            <p>Manage datasets without terminal commands</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Total Datasets</h3>
                <div class="number">{{ datasets|length }}</div>
            </div>
            <div class="stat-card">
                <h3>Total Tokens</h3>
                <div class="number">{{ datasets|sum(attribute='train_tokens') }}</div>
            </div>
            <div class="stat-card">
                <h3>Ready to Train</h3>
                <div class="number">{{ datasets|selectattr('has_train')|list|length }}</div>
            </div>
        </div>
        
        <div class="datasets">
            <div class="datasets-header">
                <h2>📁 Datasets</h2>
                <a href="{{ url_for('create') }}" class="btn">+ Create Dataset</a>
            </div>
            
            {% if datasets %}
            <div class="dataset-list">
                {% for dataset in datasets %}
                <div class="dataset-item">
                    <div class="dataset-info">
                        <h3>{{ dataset.name }}</h3>
                        {% if dataset.has_meta %}
                        <p>{{ dataset.train_tokens or 0 }} tokens • {{ dataset.vocab_size or 0 }} vocab size</p>
                        {% else %}
                        <p>Dataset not prepared</p>
                        {% endif %}
                    </div>
                    
                    <div class="dataset-actions">
                        <a href="{{ url_for('view_dataset', name=dataset.name) }}" class="btn btn-small">View</a>
                        {% if dataset.has_train %}
                        <a href="{{ url_for('train_dataset', name=dataset.name) }}" class="btn btn-small">Train</a>
                        {% endif %}
                        <button onclick="deleteDataset('{{ dataset.name }}')" class="btn btn-danger btn-small">Delete</button>
                    </div>
                </div>
                {% endfor %}
            </div>
            {% else %}
            <div class="empty">
                <h3>No datasets yet</h3>
                <p>Create your first dataset to get started</p>
                <a href="{{ url_for('create') }}" class="btn">Create Dataset</a>
            </div>
            {% endif %}
        </div>
    </div>
    
    <script>
        function deleteDataset(name) {
            if (confirm(`Delete dataset "${name}"? This action cannot be undone.`)) {
                fetch(`/api/dataset/${name}/delete`, {
                    method: 'POST'
                }).then(response => response.json())
                .then(data => {
                    if (data.success) {
                        location.reload();
                    } else {
                        alert('Error: ' + data.error);
                    }
                });
            }
        }
    </script>
</body>
</html>'''


@app.route('/templates/create.html')
def create_template():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Create Dataset - SloGPT</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; }
        .header { background: #2563eb; color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px; text-align: center; }
        .header h1 { margin: 0; font-size: 32px; }
        .header p { margin: 15px 0 0 0; opacity: 0.9; font-size: 16px; }
        .form-container { background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 40px; }
        .form-group { margin-bottom: 25px; }
        .form-group label { display: block; margin-bottom: 8px; font-weight: 500; color: #333; }
        .form-group input[type="text"], .form-group textarea { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; }
        .form-group textarea { min-height: 200px; resize: vertical; }
        .file-input { border: 2px dashed #ddd; border-radius: 6px; padding: 40px; text-align: center; margin-bottom: 20px; }
        .file-input:hover { border-color: #2563eb; }
        .file-input input { display: none; }
        .btn { background: #2563eb; color: white; border: none; padding: 12px 30px; border-radius: 6px; cursor: pointer; font-size: 16px; width: 100%; }
        .btn:hover { background: #1d4ed8; }
        .btn-secondary { background: #6c757d; }
        .btn-secondary:hover { background: #5a6268; }
        .error { background: #f8d7da; color: #721c24; padding: 15px; border-radius: 6px; margin-bottom: 20px; }
        .tabs { display: flex; margin-bottom: 20px; border-bottom: 2px solid #eee; }
        .tab { padding: 15px 25px; cursor: pointer; background: none; border: none; font-size: 16px; border-bottom: 3px solid transparent; }
        .tab.active { border-bottom-color: #2563eb; color: #2563eb; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📝 Create Dataset</h1>
            <p>Create a new dataset from text content or upload files</p>
        </div>
        
        <div class="form-container">
            {% if error %}
            <div class="error">
                <strong>Error:</strong> {{ error }}
            </div>
            {% endif %}
            
            <form method="POST" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="name">Dataset Name</label>
                    <input type="text" id="name" name="name" required placeholder="mydataset">
                </div>
                
                <div class="tabs">
                    <button type="button" class="tab active" onclick="showTab('text')">Text Content</button>
                    <button type="button" class="tab" onclick="showTab('file')">File Upload</button>
                </div>
                
                <div id="text-tab" class="tab-content active">
                    <div class="form-group">
                        <label for="content">Training Text</label>
                        <textarea id="content" name="content" placeholder="Enter your training text here...">Your training data goes here. This can be any text content that you want the model to learn from.</textarea>
                    </div>
                </div>
                
                <div id="file-tab" class="tab-content">
                    <div class="file-input">
                        <input type="file" id="file" name="file" accept=".txt,.json,.csv,.py,.js,.md">
                        <p>📁 Drop files here or click to upload</p>
                        <p>Supports: .txt, .json, .csv, .py, .js, .md</p>
                    </div>
                </div>
                
                <button type="submit" class="btn">🚀 Create Dataset</button>
                <a href="{{ url_for('index') }}" class="btn btn-secondary">Cancel</a>
            </form>
        </div>
    </div>
    
    <script>
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
        }
        
        // Handle file upload preview
        document.getElementById('file').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                document.querySelector('.file-input p').textContent = `📄 Selected: ${file.name} (${(file.size/1024/1024).toFixed(2)} MB)`;
            }
        });
    </script>
</body>
</html>'''


@app.route('/templates/dataset.html')
def dataset_template():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset View - SloGPT</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; padding: 20px; }
        .header { background: #2563eb; color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px; text-align: center; }
        .header h1 { margin: 0; font-size: 32px; }
        .dataset-info { background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 40px; margin-bottom: 30px; }
        .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 30px; }
        .info-item { text-align: center; }
        .info-label { font-size: 12px; color: #666; text-transform: uppercase; margin-bottom: 5px; }
        .info-value { font-size: 24px; font-weight: bold; color: #2563eb; }
        .actions { background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 30px; text-align: center; }
        .btn { background: #2563eb; color: white; border: none; padding: 12px 30px; border-radius: 6px; cursor: pointer; font-size: 16px; text-decoration: none; display: inline-block; margin: 0 10px; }
        .btn:hover { background: #1d4ed8; }
        .btn-danger { background: #dc3545; }
        .btn-danger:hover { background: #c82333; }
        .btn-success { background: #28a745; }
        .btn-success:hover { background: #218838; }
        .preview { background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 30px; margin-bottom: 30px; }
        .preview-text { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; padding: 20px; font-family: 'Courier New', monospace; font-size: 14px; white-space: pre-wrap; max-height: 300px; overflow-y: auto; }
        .error { background: #f8d7da; color: #721c24; padding: 15px; border-radius: 6px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {{ dataset.name or 'Unknown Dataset' }}</h1>
            <p>Dataset details and training options</p>
        </div>
        
        {% if dataset.error %}
            <div class="error">
                <strong>Error:</strong> {{ dataset.error }}
            </div>
        {% else %}
            <div class="dataset-info">
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Vocabulary Size</div>
                        <div class="info-value">{{ dataset.vocab_size or 0 }}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Training Tokens</div>
                        <div class="info-value">{{ dataset.train_tokens or 0 }}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Validation Tokens</div>
                        <div class="info-value">{{ dataset.val_tokens or 0 }}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Total Characters</div>
                        <div class="info-value">{{ dataset.total_characters or 0 }}</div>
                    </div>
                </div>
            </div>
            
            {% if dataset.preview_text %}
            <div class="preview">
                <h3>📄 Content Preview</h3>
                <div class="preview-text">{{ dataset.preview_text }}</div>
            </div>
            {% endif %}
            
            <div class="actions">
                <a href="{{ url_for('train_dataset', name=dataset.name) }}" class="btn btn-success">🚀 Start Training</a>
                <a href="{{ url_for('index') }}" class="btn">← Back to Dashboard</a>
            </div>
        {% endif %}
    </div>
</body>
</html>'''


if __name__ == '__main__':
    print("🌐 Starting SloGPT Web Interface...")
    print("🚀 Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=True)