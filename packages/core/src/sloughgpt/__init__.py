"""SloughGPT - Enterprise AI Framework"""

__version__ = "1.0.0"
__author__ = "SloughGPT Team"

from .user_management import get_user_manager
from .cost_optimization import get_cost_optimizer  
from .data_learning import DatasetPipeline
from .reasoning_engine import ReasoningEngine
from .auth import AuthService
from .monitoring import MonitoringService

__all__ = [
    "get_user_manager",
    "get_cost_optimizer", 
    "DatasetPipeline",
    "ReasoningEngine",
    "AuthService",
    "MonitoringService"
]