"""SloughGPT - Enterprise AI Framework"""

__version__ = "1.0.0"
__author__ = "SloughGPT Team"

# Core modules - always available
from .user_management import get_user_manager
from .cost_optimization import get_cost_optimizer  
from .data_learning import DatasetPipeline
from .reasoning_engine import ReasoningEngine
from .auth import AuthService
from .monitoring import MonitoringService
from .deployment import DeploymentManager
from .performance import PerformanceOptimizer, performance_optimizer, auto_optimizer

# Optional modules - may not be available due to dependencies
try:
    from .model_serving import ModelManager, model_manager
    _MODEL_SERVING_AVAILABLE = True
except ImportError:
    _MODEL_SERVING_AVAILABLE = False
    ModelManager = None
    model_manager = None

try:
    from .distributed_training import TrainingManager, training_manager
    _DISTRIBUTED_TRAINING_AVAILABLE = True
except ImportError:
    _DISTRIBUTED_TRAINING_AVAILABLE = False
    TrainingManager = None
    training_manager = None

try:
    from .api import app
    _API_AVAILABLE = True
except ImportError:
    _API_AVAILABLE = False
    app = None

__all__ = [
    "get_user_manager",
    "get_cost_optimizer", 
    "DatasetPipeline",
    "ReasoningEngine",
    "AuthService",
    "MonitoringService",
    "DeploymentManager",
    "PerformanceOptimizer",
    "performance_optimizer",
    "auto_optimizer"
]

# Add optional modules if available
if _MODEL_SERVING_AVAILABLE:
    __all__.extend(["ModelManager", "model_manager"])

if _DISTRIBUTED_TRAINING_AVAILABLE:
    __all__.extend(["TrainingManager", "training_manager"])

if _API_AVAILABLE:
    __all__.extend(["app"])

# Availability flags
__model_serving_available__ = _MODEL_SERVING_AVAILABLE
__distributed_training_available__ = _DISTRIBUTED_TRAINING_AVAILABLE
__api_available__ = _API_AVAILABLE