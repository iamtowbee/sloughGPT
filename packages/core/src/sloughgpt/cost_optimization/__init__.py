"""Cost optimization module for tracking and managing AI usage costs."""

from enum import Enum
from typing import Dict, List, Optional, Any, NamedTuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid


class CostMetricType(Enum):
    TOKEN_INFERENCE = "token_inference"
    TRAINING_COMPUTE = "training_compute"
    STORAGE_USAGE = "storage_usage"
    API_CALLS = "api_calls"


@dataclass
class CostMetric:
    user_id: int
    metric_type: CostMetricType
    amount: float
    unit: str
    cost: float
    timestamp: datetime
    model_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class UsageAnalysis:
    total_cost: float
    avg_daily_cost: float
    metrics_by_type: Dict[str, float]
    recommendations: List[Dict[str, Any]]


class CostOptimizer:
    def __init__(self):
        self.metrics: List[CostMetric] = []
        self.user_budgets: Dict[int, Dict[str, float]] = {}
        self.cost_per_token = {
            "sloughgpt-base": 0.000001,
            "sloughgpt-large": 0.000002,
            "sloughgpt-xl": 0.000004
        }
    
    def track_metric(self, user_id: int, metric_type: CostMetricType, amount: float, 
                    model_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track a cost metric."""
        cost = 0.0
        
        if metric_type == CostMetricType.TOKEN_INFERENCE:
            cost_per_token = self.cost_per_token.get(model_name or "sloughgpt-base", 0.000001)
            cost = amount * cost_per_token
            unit = "tokens"
        elif metric_type == CostMetricType.STORAGE_USAGE:
            cost = amount * 0.000023  # $0.023 per GB per month
            unit = "GB"
        elif metric_type == CostMetricType.TRAINING_COMPUTE:
            cost = amount * 0.0001  # Example rate
            unit = "hours"
        else:
            cost = amount * 0.001
            unit = "requests"
        
        metric = CostMetric(
            user_id=user_id,
            metric_type=metric_type,
            amount=amount,
            unit=unit,
            cost=cost,
            timestamp=datetime.now(),
            model_name=model_name,
            metadata=metadata
        )
        
        self.metrics.append(metric)
        
        # Check budget alerts
        self._check_budget_alerts(user_id, metric)
    
    def analyze_usage_patterns(self, user_id: int, days: int = 30) -> UsageAnalysis:
        """Analyze usage patterns for a user."""
        cutoff_date = datetime.now() - timedelta(days=days)
        user_metrics = [m for m in self.metrics if m.user_id == user_id and m.timestamp > cutoff_date]
        
        total_cost = sum(m.cost for m in user_metrics)
        avg_daily_cost = total_cost / days
        
        metrics_by_type = {}
        for metric in user_metrics:
            key = metric.metric_type.value
            metrics_by_type[key] = metrics_by_type.get(key, 0) + metric.cost
        
        recommendations = self._generate_recommendations(user_id, user_metrics)
        
        return UsageAnalysis(
            total_cost=total_cost,
            avg_daily_cost=avg_daily_cost,
            metrics_by_type=metrics_by_type,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, user_id: int, metrics: List[CostMetric]) -> List[Dict[str, Any]]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        token_costs = sum(m.cost for m in metrics if m.metric_type == CostMetricType.TOKEN_INFERENCE)
        if token_costs > 10:
            recommendations.append({
                "strategy": "Use smaller model for simple queries",
                "potential_savings_monthly": token_costs * 0.3,
                "description": "Switch to base model for routine requests"
            })
        
        return recommendations
    
    def _check_budget_alerts(self, user_id: int, metric: CostMetric) -> None:
        """Check if user is approaching budget limits."""
        budget = self.user_budgets.get(user_id, {})
        if not budget:
            return
        
        # Calculate monthly usage
        current_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        monthly_metrics = [m for m in self.metrics 
                          if m.user_id == user_id and m.timestamp >= current_month]
        monthly_cost = sum(m.cost for m in monthly_metrics)
        
        monthly_budget = budget.get("monthly", float('inf'))
        if monthly_budget == float('inf'):
            return
        
        usage_percentage = (monthly_cost / monthly_budget) * 100
        
        if usage_percentage >= 95:
            print(f"CRITICAL: User {user_id} has used {usage_percentage:.1f}% of monthly budget")
        elif usage_percentage >= 80:
            print(f"WARNING: User {user_id} has used {usage_percentage:.1f}% of monthly budget")
    
    def set_user_budget(self, user_id: int, monthly: Optional[float] = None, 
                       daily: Optional[float] = None, hourly: Optional[float] = None) -> None:
        """Set budget limits for a user."""
        self.user_budgets[user_id] = {
            "monthly": monthly or float('inf'),
            "daily": daily or float('inf'),
            "hourly": hourly or float('inf')
        }


def get_cost_optimizer() -> CostOptimizer:
    """Get singleton cost optimizer instance."""
    return CostOptimizer()