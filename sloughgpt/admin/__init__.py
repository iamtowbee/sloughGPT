"""
Admin Dashboard Module for SloughGPT

A comprehensive FastAPI-based admin dashboard for managing the SloughGPT system
with real-time WebSocket updates and modern UI components.
"""

from .admin_app import create_app, start_admin_server
from .admin_config import AdminConfig, AdminTheme
from .admin_routes import admin_router
from .admin_utils import (
    get_system_metrics, get_user_stats, get_model_stats,
    export_data, cleanup_data
)

__all__ = [
    'create_app', 'start_admin_server', 'admin_router',
    'AdminConfig', 'AdminTheme',
    'get_system_metrics', 'get_user_stats', 'get_model_stats',
    'export_data', 'cleanup_data'
]