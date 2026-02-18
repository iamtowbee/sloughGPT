"""
User Interfaces Domain

This domain contains all components related to user interfaces
including web UI, chat interfaces, API endpoints, and CLI tools.
"""

from .api import APIController
from .chat import ChatInterface
from .cli import CLIInterface
from .components import UIComponents
from .web import WebInterface
from .web_interface import DatasetWebManager
from .analytics import AnalyticsManager, MetricsCollector
from .cli_shortcuts import CLIManager
from . import webui

__all__ = [
    "WebInterface",
    "ChatInterface",
    "APIController",
    "CLIInterface",
    "UIComponents",
    "AnalyticsManager",
    "MetricsCollector",
    "CLIManager",
    "webui",
]
