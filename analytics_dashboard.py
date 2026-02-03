#!/usr/bin/env python3
"""
Analytics Dashboard for Training Monitoring

Real-time web dashboard for monitoring dataset training, performance metrics, and system health.
"""

from flask import Flask, render_template_string, jsonify
import time
import json
import threading
import psutil
import subprocess
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt
import io
import base64


app = Flask(__name__)

# Global state
performance_data = deque(maxlen=100)
training_logs = deque(maxlen=50)
system_stats = {}
training_status = {"status": "idle", "current_epoch": 0, "loss": 0.0, "learning_rate": 0.0003}


class AnalyticsManager:
    """Manages training analytics and monitoring."""
    
    def __init__(self):
        self.start_time = time.time()
        self.epochs_completed = 0
        self.total_training_time = 0
        
    def get_system_metrics(self) -> dict:
        """Get current system metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "disk_usage": psutil.disk_usage('/').percent,
            "timestamp": time.time()
        }
    
    def get_performance_metrics(self) -> dict:
        """Get training performance metrics."""
        return {
            "epoch": training_status["current_epoch"],
            "loss": training_status["loss"],
            "learning_rate": training_status["learning_rate"],
            "tokens_per_second": 1000 if training_status["status"] == "training" else 0,
            "timestamp": time.time()
        }
    
    def update_training_status(self, status: str, epoch: int = 0, loss: float = 0.0, lr: float = 0.0003):
        """Update training status."""
        training_status.update({
            "status": status,
            "current_epoch": epoch,
            "loss": loss,
            "learning_rate": lr
        })
        
        # Log the update
        log_entry = {
            "timestamp": time.time(),
            "status": status,
            "epoch": epoch,
            "loss": loss,
            "learning_rate": lr
        }
        training_logs.append(log_entry)
    
    def generate_performance_chart(self) -> str:
        """Generate performance chart as base64 image."""
        if not performance_data:
            return ""
        
        # Extract data for chart
        epochs = [d["epoch"] for d in performance_data if d["epoch"] > 0]
        losses = [d["loss"] for d in performance_data if d["epoch"] > 0]
        
        if not epochs:
            return ""
        
        # Create matplotlib chart
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Performance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"


analytics = AnalyticsManager()


@app.route('/')
def dashboard():
    """Main analytics dashboard."""
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/system-metrics')
def get_system_metrics_api():
    """API endpoint for system metrics."""
    metrics = analytics.get_system_metrics()
    performance_data.append(analytics.get_performance_metrics())
    return jsonify(metrics)


@app.route('/api/performance-data')
def get_performance_data_api():
    """API endpoint for performance data."""
    return jsonify(list(performance_data))


@app.route('/api/training-logs')
def get_training_logs_api():
    """API endpoint for training logs."""
    return jsonify(list(training_logs))


@app.route('/api/performance-chart')
def get_performance_chart_api():
    """API endpoint for performance chart."""
    chart_data = analytics.generate_performance_chart()
    return jsonify({"chart": chart_data})


@app.route('/api/training-control', methods=['POST'])
def control_training():
    """Control training (start/pause/resume)."""
    action = request.json.get('action')
    
    if action == 'start':
        analytics.update_training_status('training')
        return jsonify({"status": "started"})
    elif action == 'pause':
        analytics.update_training_status('paused')
        return jsonify({"status": "paused"})
    elif action == 'resume':
        analytics.update_training_status('training')
        return jsonify({"status": "resumed"})
    else:
        return jsonify({"error": "Invalid action"})


# HTML Template for Dashboard
DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SloGPT Training Analytics</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8f9fa; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 30px; text-align: center; }
        .header h1 { margin: 0; font-size: 36px; }
        .header p { margin: 15px 0 0 0; opacity: 0.9; font-size: 16px; }
        .dashboard { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 30px; }
        .card { background: white; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); padding: 25px; }
        .card h2 { margin: 0 0 20px 0; color: #333; font-size: 20px; display: flex; align-items: center; }
        .card h2 .icon { font-size: 24px; margin-right: 10px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }
        .metric { text-align: center; }
        .metric-value { font-size: 32px; font-weight: bold; color: #2563eb; margin-bottom: 5px; }
        .metric-label { font-size: 14px; color: #666; text-transform: uppercase; }
        .metric-change { font-size: 12px; margin-top: 5px; }
        .metric-change.positive { color: #28a745; }
        .metric-change.negative { color: #dc3545; }
        .controls { display: flex; gap: 10px; margin-top: 20px; }
        .btn { background: #2563eb; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-size: 14px; }
        .btn:hover { background: #1d4ed8; }
        .btn.success { background: #28a745; }
        .btn.success:hover { background: #218838; }
        .btn.danger { background: #dc3545; }
        .btn.danger:hover { background: #c82333; }
        .btn:disabled { background: #6c757d; cursor: not-allowed; }
        .status { padding: 15px; border-radius: 8px; margin-top: 15px; font-weight: 500; }
        .status.training { background: #d4edda; color: #155724; }
        .status.paused { background: #fff3cd; color: #856404; }
        .status.idle { background: #f8d7da; color: #721c24; }
        .chart-container { grid-column: 1 / -1; grid-row: 1 / -1; }
        .logs { grid-column: 1 / -1; grid-row: 2 / -1; max-height: 400px; }
        .log-container { background: #2d3748; color: #e9ecef; border-radius: 8px; padding: 20px; height: 350px; overflow-y: auto; font-family: 'Courier New', monospace; font-size: 12px; }
        .log-entry { margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #495057; }
        .log-entry:last-child { border-bottom: none; }
        .log-timestamp { color: #adb5bd; font-size: 10px; margin-bottom: 5px; }
        .log-content { color: #e9ecef; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 SloGPT Training Analytics</h1>
            <p>Real-time monitoring and performance insights</p>
        </div>
        
        <div class="dashboard">
            <div class="card">
                <h2><span class="icon">🖥</span> System Metrics</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value" id="cpu-percent">0%</div>
                        <div class="metric-label">CPU Usage</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="memory-percent">0%</div>
                        <div class="metric-label">Memory Usage</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="disk-usage">0%</div>
                        <div class="metric-label">Disk Usage</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2><span class="icon">🚀</span> Training Status</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value" id="current-epoch">0</div>
                        <div class="metric-label">Current Epoch</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="current-loss">0.000</div>
                        <div class="metric-label">Current Loss</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="learning-rate-display">0.0003</div>
                        <div class="metric-label">Learning Rate</div>
                    </div>
                </div>
                <div class="status" id="training-status">
                    ⏸️ Status: Idle
                </div>
                <div class="controls">
                    <button class="btn success" onclick="startTraining()">Start</button>
                    <button class="btn" onclick="pauseTraining()">Pause</button>
                    <button class="btn danger" onclick="stopTraining()">Stop</button>
                </div>
            </div>
            
            <div class="card">
                <h2><span class="icon">📈</span> Performance</h2>
                <div class="chart-container">
                    <canvas id="performanceChart" width="400" height="200"></canvas>
                </div>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value" id="tokens-per-sec">0</div>
                        <div class="metric-label">Tokens/Sec</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="total-epochs">0</div>
                        <div class="metric-label">Total Epochs</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2><span class="icon">📋</span> Training Logs</h2>
                <div class="logs">
                    <div class="log-container" id="log-container">
                        <div class="log-entry">
                            <div class="log-timestamp">Waiting for logs...</div>
                            <div class="log-content">System ready for training</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let performanceChart;
        let updateInterval;
        
        // Initialize chart
        function initChart() {
            const ctx = document.getElementById('performanceChart').getContext('2d');
            performanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Training Loss',
                        data: [],
                        borderColor: '#2563eb',
                        backgroundColor: 'rgba(37, 99, 235, 0.1)',
                        borderWidth: 2,
                        pointRadius: 3,
                        pointBackgroundColor: '#2563eb'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            title: 'Epoch',
                            grid: {
                                color: 'rgba(0, 0, 0, 0.1)'
                            }
                        },
                        y: {
                            title: 'Loss',
                            grid: {
                                color: 'rgba(0, 0, 0, 0.1)'
                            }
                        }
                    }
                }
            });
        }
        
        // Update metrics
        function updateMetrics() {
            fetch('/api/system-metrics')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('cpu-percent').textContent = data.cpu_percent.toFixed(1) + '%';
                    document.getElementById('memory-percent').textContent = data.memory_percent.toFixed(1) + '%';
                    document.getElementById('disk-usage').textContent = data.disk_usage.toFixed(1) + '%';
                });
            
            fetch('/api/performance-data')
                .then(response => response.json())
                .then(data => {
                    if (data.length > 0) {
                        const latest = data[data.length - 1];
                        document.getElementById('current-epoch').textContent = latest.epoch;
                        document.getElementById('current-loss').textContent = latest.loss.toFixed(4);
                        document.getElementById('learning-rate-display').textContent = latest.learning_rate.toFixed(6);
                        document.getElementById('tokens-per-sec').textContent = latest.tokens_per_second;
                        document.getElementById('total-epochs').textContent = latest.epoch;
                        
                        // Update chart
                        updateChart(data);
                    }
                });
        }
        
        // Update chart
        function updateChart(data) {
            if (!performanceChart || data.length === 0) return;
            
            const epochs = data.map(d => d.epoch);
            const losses = data.map(d => d.loss);
            
            performanceChart.data.labels = epochs;
            performanceChart.data.datasets[0].data = losses;
            performanceChart.update();
        }
        
        // Update logs
        function updateLogs() {
            fetch('/api/training-logs')
                .then(response => response.json())
                .then(logs => {
                    const container = document.getElementById('log-container');
                    container.innerHTML = '';
                    
                    logs.slice(-20).reverse().forEach(log => {
                        const entry = document.createElement('div');
                        entry.className = 'log-entry';
                        
                        const timestamp = new Date(log.timestamp * 1000).toLocaleString();
                        entry.innerHTML = `
                            <div class="log-timestamp">${timestamp}</div>
                            <div class="log-content">[${log.status.toUpperCase()}] Epoch ${log.epoch} | Loss: ${log.loss.toFixed(4)} | LR: ${log.learning_rate}</div>
                        `;
                        
                        container.appendChild(entry);
                    });
                });
        }
        
        // Training controls
        function startTraining() {
            fetch('/api/training-control', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'start'})
            }).then(response => response.json())
            .then(data => {
                updateStatus('training');
                console.log('Training started:', data);
            });
        }
        
        function pauseTraining() {
            fetch('/api/training-control', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'pause'})
            }).then(response => response.json())
            .then(data => {
                updateStatus('paused');
                console.log('Training paused:', data);
            });
        }
        
        function stopTraining() {
            fetch('/api/training-control', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'stop'})
            }).then(response => response.json())
            .then(data => {
                updateStatus('idle');
                console.log('Training stopped:', data);
            });
        }
        
        function updateStatus(status) {
            const statusElement = document.getElementById('training-status');
            
            switch(status) {
                case 'training':
                    statusElement.className = 'status training';
                    statusElement.innerHTML = '🚀 Status: Training';
                    break;
                case 'paused':
                    statusElement.className = 'status paused';
                    statusElement.innerHTML = '⏸️ Status: Paused';
                    break;
                case 'idle':
                    statusElement.className = 'status idle';
                    statusElement.innerHTML = '⏸️ Status: Idle';
                    break;
            }
        }
        
        // Initialize
        initChart();
        
        // Update every 2 seconds
        updateInterval = setInterval(() => {
            updateMetrics();
            updateLogs();
        }, 2000);
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            if (updateInterval) {
                clearInterval(updateInterval);
            }
        });
    </script>
</body>
</html>'''


if __name__ == '__main__':
    print("📊 Starting Analytics Dashboard...")
    print("🚀 Open http://localhost:5001 in your browser")
    app.run(host='0.0.0.0', port=5001, debug=True)