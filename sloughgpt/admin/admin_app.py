"""
Admin App - FastAPI-based admin dashboard for SloughGPT
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import Dict, List
import json
import asyncio
from datetime import datetime

from .admin_config import AdminConfig
from .admin_routes import admin_router
from .admin_utils import WebSocketManager
from ..core.logging_system import get_logger

logger = get_logger(__name__)

def create_app(config: AdminConfig = None) -> FastAPI:
    """Create and configure the FastAPI admin application"""
    
    if config is None:
        config = AdminConfig()
    
    # Create FastAPI app
    app = FastAPI(
        title="SloughGPT Admin Dashboard",
        description="Enterprise-grade admin interface for managing SloughGPT",
        version="1.0.0",
        docs_url="/api/docs" if config.enable_docs else None,
        redoc_url="/api/redoc" if config.enable_docs else None
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure based on your security requirements
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include admin routes
    app.include_router(admin_router, prefix="/api")
    
    # WebSocket manager for real-time updates
    ws_manager = WebSocketManager()
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time dashboard updates"""
        await ws_manager.connect(websocket)
        try:
            while True:
                # Send periodic updates
                await ws_manager.broadcast({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat()
                })
                await asyncio.sleep(30)
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
    
    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Main dashboard HTML page"""
        return get_dashboard_html(config.theme)
    
    # Store WebSocket manager for other components to use
    app.state.ws_manager = ws_manager
    
    logger.info("Admin dashboard application created")
    return app

def start_admin_server(host: str = "127.0.0.1", port: int = 8080, **kwargs):
    """Start the admin dashboard server"""
    import uvicorn
    
    config = AdminConfig(**kwargs)
    app = create_app(config)
    
    logger.info(f"Starting admin dashboard on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

def get_dashboard_html(theme: str = "light") -> str:
    """Generate the main dashboard HTML with Tailwind CSS and Alpine.js"""
    
    return f"""
<!DOCTYPE html>
<html lang="en" data-theme="{theme}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SloughGPT Admin Dashboard</title>
    
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Alpine.js for reactive components -->
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    
    <!-- Chart.js for visualizations -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <!-- Custom Tailwind Configuration -->
    <script>
        tailwind.config = {{
            theme: {{
                extend: {{
                    colors: {{
                        primary: {{
                            50: '#eff6ff',
                            500: '#3b82f6',
                            600: '#2563eb',
                            700: '#1d4ed8',
                            900: '#1e3a8a',
                        }},
                        success: {{
                            500: '#10b981',
                            600: '#059669',
                        }},
                        warning: {{
                            500: '#f59e0b',
                            600: '#d97706',
                        }},
                        error: {{
                            500: '#ef4444',
                            600: '#dc2626',
                        }}
                    }}
                }}
            }}
        }}
    </script>
    
    <style>
        /* Dark mode styles */
        [data-theme="dark"] {{
            background-color: #1f2937;
            color: #f9fafb;
        }}
        
        [data-theme="dark"] .bg-white {{
            background-color: #374151 !important;
        }}
        
        [data-theme="dark"] .text-gray-900 {{
            color: #f9fafb !important;
        }}
        
        [data-theme="dark"] .text-gray-600 {{
            color: #d1d5db !important;
        }}
        
        [data-theme="dark"] .border-gray-200 {{
            border-color: #4b5563 !important;
        }}
        
        /* Animations */
        @keyframes pulse-light {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.8; }}
        }}
        
        .animate-pulse-light {{
            animation: pulse-light 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }}
        
        /* Glass morphism effect */
        .glass {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        
        [data-theme="dark"] .glass {{
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
    </style>
</head>
<body class="bg-gray-50 text-gray-900">
    <div x-data="dashboard()" class="min-h-screen">
        <!-- Header -->
        <header class="bg-white shadow-sm border-b border-gray-200">
            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div class="flex justify-between items-center h-16">
                    <div class="flex items-center">
                        <div class="flex-shrink-0">
                            <h1 class="text-2xl font-bold text-primary-600">SloughGPT Admin</h1>
                        </div>
                    </div>
                    
                    <div class="flex items-center space-x-4">
                        <!-- Connection Status -->
                        <div class="flex items-center">
                            <div class="w-2 h-2 rounded-full mr-2"
                                 :class="connectionStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'"></div>
                            <span class="text-sm text-gray-600" x-text="connectionStatus"></span>
                        </div>
                        
                        <!-- Theme Toggle -->
                        <button @click="toggleTheme()" 
                                class="p-2 rounded-lg hover:bg-gray-100 transition-colors">
                            <svg x-show="theme === 'light'" class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                      d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"/>
                            </svg>
                            <svg x-show="theme === 'dark'" class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                      d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"/>
                            </svg>
                        </button>
                        
                        <!-- Refresh Button -->
                        <button @click="refreshData()" 
                                :disabled="loading"
                                class="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50">
                            <span x-show="!loading">Refresh</span>
                            <span x-show="loading">Loading...</span>
                        </button>
                    </div>
                </div>
            </div>
        </header>

        <!-- Main Content -->
        <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <!-- System Status Cards -->
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <div class="bg-white rounded-lg shadow p-6 border border-gray-200">
                    <div class="flex items-center">
                        <div class="flex-shrink-0 bg-success-100 rounded-lg p-3">
                            <svg class="w-6 h-6 text-success-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                            </svg>
                        </div>
                        <div class="ml-4">
                            <p class="text-sm font-medium text-gray-600">System Health</p>
                            <p class="text-2xl font-semibold text-gray-900" x-text="systemStatus"></p>
                        </div>
                    </div>
                </div>

                <div class="bg-white rounded-lg shadow p-6 border border-gray-200">
                    <div class="flex items-center">
                        <div class="flex-shrink-0 bg-primary-100 rounded-lg p-3">
                            <svg class="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                      d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z"/>
                            </svg>
                        </div>
                        <div class="ml-4">
                            <p class="text-sm font-medium text-gray-600">Active Users</p>
                            <p class="text-2xl font-semibold text-gray-900" x-text="userCount"></p>
                        </div>
                    </div>
                </div>

                <div class="bg-white rounded-lg shadow p-6 border border-gray-200">
                    <div class="flex items-center">
                        <div class="flex-shrink-0 bg-warning-100 rounded-lg p-3">
                            <svg class="w-6 h-6 text-warning-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
                            </svg>
                        </div>
                        <div class="ml-4">
                            <p class="text-sm font-medium text-gray-600">Models</p>
                            <p class="text-2xl font-semibold text-gray-900" x-text="modelCount"></p>
                        </div>
                    </div>
                </div>

                <div class="bg-white rounded-lg shadow p-6 border border-gray-200">
                    <div class="flex items-center">
                        <div class="flex-shrink-0 bg-error-100 rounded-lg p-3">
                            <svg class="w-6 h-6 text-error-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                      d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                            </svg>
                        </div>
                        <div class="ml-4">
                            <p class="text-sm font-medium text-gray-600">Total Cost</p>
                            <p class="text-2xl font-semibold text-gray-900" x-text="'$' + totalCost.toFixed(2)"></p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Charts Section -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                <!-- Usage Chart -->
                <div class="bg-white rounded-lg shadow p-6 border border-gray-200">
                    <h3 class="text-lg font-semibold mb-4">API Usage Trends</h3>
                    <canvas id="usageChart"></canvas>
                </div>

                <!-- Cost Chart -->
                <div class="bg-white rounded-lg shadow p-6 border border-gray-200">
                    <h3 class="text-lg font-semibold mb-4">Cost Breakdown</h3>
                    <canvas id="costChart"></canvas>
                </div>
            </div>

            <!-- Recent Activity -->
            <div class="bg-white rounded-lg shadow border border-gray-200">
                <div class="px-6 py-4 border-b border-gray-200">
                    <h3 class="text-lg font-semibold">Recent Activity</h3>
                </div>
                <div class="p-6">
                    <div class="space-y-4">
                        <template x-for="activity in recentActivity" :key="activity.id">
                            <div class="flex items-center space-x-4 p-4 bg-gray-50 rounded-lg">
                                <div class="flex-shrink-0">
                                    <div class="w-8 h-8 rounded-full flex items-center justify-center"
                                         :class="getActivityColor(activity.type)">
                                        <svg class="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                                        </svg>
                                    </div>
                                </div>
                                <div class="flex-1">
                                    <p class="text-sm font-medium text-gray-900" x-text="activity.title"></p>
                                    <p class="text-sm text-gray-600" x-text="activity.description"></p>
                                </div>
                                <div class="text-sm text-gray-500" x-text="activity.time"></div>
                            </div>
                        </template>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <script>
        function dashboard() {{
            return {{
                theme: localStorage.getItem('theme') || 'light',
                connectionStatus: 'connecting',
                loading: false,
                systemStatus: 'Healthy',
                userCount: 0,
                modelCount: 0,
                totalCost: 0,
                recentActivity: [],
                usageChart: null,
                costChart: null,
                
                init() {{
                    this.applyTheme();
                    this.connectWebSocket();
                    this.initCharts();
                    this.refreshData();
                }},
                
                connectWebSocket() {{
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const ws = new WebSocket(`${{protocol}}//${{window.location.host}}/ws`);
                    
                    ws.onopen = () => {{
                        this.connectionStatus = 'connected';
                    }};
                    
                    ws.onmessage = (event) => {{
                        const data = JSON.parse(event.data);
                        if (data.type === 'heartbeat') {{
                            this.connectionStatus = 'connected';
                        }}
                    }};
                    
                    ws.onclose = () => {{
                        this.connectionStatus = 'disconnected';
                        setTimeout(() => this.connectWebSocket(), 5000);
                    }};
                }},
                
                toggleTheme() {{
                    this.theme = this.theme === 'light' ? 'dark' : 'light';
                    this.applyTheme();
                }},
                
                applyTheme() {{
                    document.documentElement.setAttribute('data-theme', this.theme);
                    localStorage.setItem('theme', this.theme);
                }},
                
                async refreshData() {{
                    this.loading = true;
                    try {{
                        const response = await fetch('/api/dashboard/stats');
                        const data = await response.json();
                        
                        this.systemStatus = data.system_status;
                        this.userCount = data.user_count;
                        this.modelCount = data.model_count;
                        this.totalCost = data.total_cost;
                        this.recentActivity = data.recent_activity;
                        
                        this.updateCharts(data.chart_data);
                    }} catch (error) {{
                        console.error('Error refreshing data:', error);
                    }} finally {{
                        this.loading = false;
                    }}
                }},
                
                initCharts() {{
                    // Usage Chart
                    const usageCtx = document.getElementById('usageChart').getContext('2d');
                    this.usageChart = new Chart(usageCtx, {{
                        type: 'line',
                        data: {{
                            labels: [],
                            datasets: [{{
                                label: 'API Requests',
                                data: [],
                                borderColor: 'rgb(59, 130, 246)',
                                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                tension: 0.1
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            scales: {{
                                y: {{
                                    beginAtZero: true
                                }}
                            }}
                        }}
                    }});
                    
                    // Cost Chart
                    const costCtx = document.getElementById('costChart').getContext('2d');
                    this.costChart = new Chart(costCtx, {{
                        type: 'doughnut',
                        data: {{
                            labels: ['Inference', 'Training', 'Storage', 'Other'],
                            datasets: [{{
                                data: [],
                                backgroundColor: [
                                    'rgb(59, 130, 246)',
                                    'rgb(16, 185, 129)',
                                    'rgb(245, 158, 11)',
                                    'rgb(156, 163, 175)'
                                ]
                            }}]
                        }},
                        options: {{
                            responsive: true
                        }}
                    }});
                }},
                
                updateCharts(chartData) {{
                    if (this.usageChart && chartData.usage) {{
                        this.usageChart.data.labels = chartData.usage.labels;
                        this.usageChart.data.datasets[0].data = chartData.usage.data;
                        this.usageChart.update();
                    }}
                    
                    if (this.costChart && chartData.cost) {{
                        this.costChart.data.datasets[0].data = chartData.cost.data;
                        this.costChart.update();
                    }}
                }},
                
                getActivityColor(type) {{
                    const colors = {{
                        'user': 'bg-primary-500',
                        'model': 'bg-success-500',
                        'system': 'bg-warning-500',
                        'error': 'bg-error-500'
                    }};
                    return colors[type] || 'bg-gray-500';
                }}
            }}
        }}
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    # Start the admin server directly
    start_admin_server()