#!/usr/bin/env python3
"""
SloughGPT Cost Optimization System
Real-time cost tracking, budget management, and optimization recommendations
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np

from .core.database import Base
from .core.db_manager import get_db_session
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean
from .core.exceptions import create_error, SecurityError

class CostMetricType(Enum):
    """Types of cost metrics"""
    TOKEN_INFERENCE = "token_inference"
    TOKEN_TRAINING = "token_training"
    MODEL_STORAGE = "model_storage"
    DATA_STORAGE = "data_storage"
    API_CALL = "api_call"
    COMPUTE_TIME = "compute_time"
    BANDWIDTH = "bandwidth"

class OptimizationStrategy(Enum):
    """Cost optimization strategies"""
    MODEL_QUANTIZATION = "model_quantization"
    REQUEST_BATCHING = "request_batching"
    CACHED_RESPONSES = "cached_responses"
    AUTO_SCALING = "auto_scaling"
    SPOT_INSTANCES = "spot_instances"
    RESERVED_CAPACITY = "reserved_capacity"

@dataclass
class CostPricing:
    """Pricing configuration for different services"""
    # Inference pricing (per 1K tokens)
    inference_input_cost_per_1k: float = 0.001
    inference_output_cost_per_1k: float = 0.002
    
    # Training pricing (per 1K tokens)
    training_cost_per_1k: float = 0.005
    
    # Storage pricing (per GB per month)
    model_storage_cost_per_gb_month: float = 0.10
    data_storage_cost_per_gb_month: float = 0.023
    
    # Compute pricing (per hour)
    compute_cost_per_hour: float = 0.50
    gpu_cost_per_hour: float = 2.50
    
    # Network pricing
    bandwidth_cost_per_gb: float = 0.09
    api_call_cost_per_1k: float = 0.001

@dataclass 
class BudgetConfig:
    """Budget configuration"""
    monthly_limit: float = 100.0
    daily_limit: Optional[float] = None
    hourly_limit: Optional[float] = None
    
    # Alert thresholds
    warning_threshold: float = 0.8  # 80%
    critical_threshold: float = 0.95  # 95%
    
    # Cost alerts
    email_alerts: bool = True
    webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None
    
    # Auto-optimization
    auto_optimization: bool = True
    optimization_strategies: List[OptimizationStrategy] = None
    
    def __post_init__(self):
        if self.optimization_strategies is None:
            self.optimization_strategies = [
                OptimizationStrategy.CACHED_RESPONSES,
                OptimizationStrategy.REQUEST_BATCHING
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class CostMetric(Base):
    """Cost metric tracking"""
    __tablename__ = "cost_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    metric_type = Column(String(50), nullable=False, index=True)
    amount = Column(Float, nullable=False)  # Quantity (tokens, hours, GB, etc.)
    unit_cost = Column(Float, nullable=False)  # Cost per unit
    total_cost = Column(Float, nullable=False)  # Total cost for this metric
    
    # Metadata
    model_name = Column(String(100), nullable=True)
    request_id = Column(String(100), nullable=True, index=True)
    session_id = Column(String(100), nullable=True)
    metric_metadata = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    hour_key = Column(String(20), nullable=False, index=True)  # YYYY-MM-DD-HH
    day_key = Column(String(10), nullable=False, index=True)   # YYYY-MM-DD
    month_key = Column(String(7), nullable=False, index=True)    # YYYY-MM
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "metric_type": self.metric_type,
            "amount": self.amount,
            "unit_cost": self.unit_cost,
            "total_cost": self.total_cost,
            "model_name": self.model_name,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "metadata": self.metric_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "hour_key": self.hour_key,
            "day_key": self.day_key,
            "month_key": self.month_key
        }

class BudgetAlert(Base):
    """Budget alerts and notifications"""
    __tablename__ = "budget_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    alert_type = Column(String(20), nullable=False)  # warning, critical, exceeded
    threshold_percentage = Column(Float, nullable=False)
    current_spending = Column(Float, nullable=False)
    budget_limit = Column(Float, nullable=False)
    period = Column(String(10), nullable=False)  # hourly, daily, monthly
    
    # Notification details
    alert_sent = Column(Boolean, default=False)
    notification_method = Column(String(20), nullable=True)  # email, webhook, slack
    notification_status = Column(String(20), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "alert_type": self.alert_type,
            "threshold_percentage": self.threshold_percentage,
            "current_spending": self.current_spending,
            "budget_limit": self.budget_limit,
            "period": self.period,
            "notification_sent": self.alert_sent,
            "notification_method": self.notification_method,
            "notification_status": self.notification_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }

class OptimizationRecommendation(Base):
    """Cost optimization recommendations"""
    __tablename__ = "optimization_recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    strategy = Column(String(50), nullable=False)
    recommendation = Column(Text, nullable=False)
    
    # Impact estimation
    potential_savings_monthly = Column(Float, nullable=False)
    implementation_complexity = Column(String(20), nullable=False)  # low, medium, high
    priority = Column(String(10), nullable=False)  # low, medium, high
    
    # Status
    rec_status = Column(String(20), default="pending")  # pending, implemented, rejected
    implemented_at = Column(DateTime, nullable=True)
    implementation_notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "strategy": self.strategy,
            "recommendation": self.recommendation,
            "potential_savings_monthly": self.potential_savings_monthly,
            "implementation_complexity": self.implementation_complexity,
            "priority": self.priority,
            "status": self.rec_status,
            "implemented_at": self.implemented_at.isoformat() if self.implemented_at else None,
            "implementation_notes": self.implementation_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class CostOptimizer:
    """Comprehensive cost optimization system"""
    
    def __init__(self, pricing: Optional[CostPricing] = None):
        self.pricing = pricing or CostPricing()
        
    def calculate_inference_cost(self, input_tokens: int, output_tokens: int, model_name: str = "default") -> float:
        """Calculate inference cost"""
        input_cost = (input_tokens / 1000) * self.pricing.inference_input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.pricing.inference_output_cost_per_1k
        return input_cost + output_cost
    
    def calculate_training_cost(self, tokens: int, compute_hours: float = 0) -> float:
        """Calculate training cost"""
        token_cost = (tokens / 1000) * self.pricing.training_cost_per_1k
        compute_cost = compute_hours * self.pricing.gpu_cost_per_hour
        return token_cost + compute_cost
    
    def calculate_storage_cost(self, gb_size: float, duration_days: int) -> float:
        """Calculate storage cost"""
        monthly_cost = gb_size * self.pricing.model_storage_cost_per_gb_month
        daily_multiplier = duration_days / 30
        return monthly_cost * daily_multiplier
    
    def analyze_usage_patterns(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Analyze usage patterns and identify optimization opportunities"""
        with get_db_session() as session:
            # Get recent cost metrics
            start_date = datetime.utcnow() - timedelta(days=days)
            
            metrics = session.query(CostMetric).filter(
                CostMetric.user_id == user_id,
                CostMetric.created_at >= start_date
            ).all()
            
            if not metrics:
                return {"error": "No usage data found"}
            
            # Aggregate by type
            by_type = {}
            for metric in metrics:
                if metric.metric_type not in by_type:
                    by_type[metric.metric_type] = {"amount": 0, "cost": 0}
                by_type[metric.metric_type]["amount"] += metric.amount
                by_type[metric.metric_type]["cost"] += metric.total_cost
            
            # Find patterns
            total_cost = sum(m.total_cost for m in metrics)
            hourly_costs = {}
            daily_costs = {}
            
            for metric in metrics:
                hour_key = metric.hour_key
                day_key = metric.day_key
                
                hourly_costs[hour_key] = hourly_costs.get(hour_key, 0) + metric.total_cost
                daily_costs[day_key] = daily_costs.get(day_key, 0) + metric.total_cost
            
            # Calculate statistics
            avg_hourly_cost = np.mean(list(hourly_costs.values())) if hourly_costs else 0
            peak_hour_cost = max(hourly_costs.values()) if hourly_costs else 0
            avg_daily_cost = np.mean(list(daily_costs.values())) if daily_costs else 0
            peak_day_cost = max(daily_costs.values()) if daily_costs else 0
            
            # Generate insights
            insights = []
            
            # Peak usage analysis
            if peak_hour_cost > avg_hourly_cost * 2:
                insights.append({
                    "type": "peak_usage",
                    "message": f"Peak hourly usage is {peak_hour_cost/avg_hourly_cost:.1f}x higher than average",
                    "recommendation": "Consider request batching or auto-scaling"
                })
            
            # Cost breakdown
            if by_type:
                highest_cost_type = max(by_type.items(), key=lambda x: x[1]["cost"])
                if highest_cost_type[1]["cost"] > total_cost * 0.6:
                    insights.append({
                        "type": "cost_concentration",
                        "message": f"{highest_cost_type[0]} accounts for {highest_cost_type[1]['cost']/total_cost:.1%} of costs",
                        "recommendation": f"Optimize {highest_cost_type[0]} usage"
                    })
            
            return {
                "period_days": days,
                "total_cost": total_cost,
                "avg_daily_cost": avg_daily_cost,
                "peak_daily_cost": peak_day_cost,
                "avg_hourly_cost": avg_hourly_cost,
                "peak_hourly_cost": peak_hour_cost,
                "cost_by_type": by_type,
                "insights": insights,
                "data_points": len(metrics)
            }
    
    def generate_optimization_recommendations(self, user_id: int) -> List[Dict[str, Any]]:
        """Generate personalized optimization recommendations"""
        usage_analysis = self.analyze_usage_patterns(user_id)
        
        if "error" in usage_analysis:
            return []
        
        recommendations = []
        
        # Model quantization recommendation
        if usage_analysis.get("cost_by_type", {}).get("token_inference", {}).get("cost", 0) > 50:
            savings = usage_analysis["total_cost"] * 0.3  # Estimate 30% savings
            recommendations.append({
                "strategy": OptimizationStrategy.MODEL_QUANTIZATION.value,
                "recommendation": "Implement model quantization to reduce inference costs",
                "potential_savings_monthly": savings,
                "implementation_complexity": "medium",
                "priority": "high"
            })
        
        # Caching recommendation
        if usage_analysis.get("peak_hourly_cost", 0) > usage_analysis.get("avg_hourly_cost", 0) * 1.5:
            savings = usage_analysis["total_cost"] * 0.2  # Estimate 20% savings
            recommendations.append({
                "strategy": OptimizationStrategy.CACHED_RESPONSES.value,
                "recommendation": "Implement response caching for frequently requested content",
                "potential_savings_monthly": savings,
                "implementation_complexity": "low",
                "priority": "medium"
            })
        
        # Request batching recommendation
        if usage_analysis.get("avg_hourly_cost", 0) > 1.0:
            savings = usage_analysis["total_cost"] * 0.15  # Estimate 15% savings
            recommendations.append({
                "strategy": OptimizationStrategy.REQUEST_BATCHING.value,
                "recommendation": "Batch smaller requests together for better efficiency",
                "potential_savings_monthly": savings,
                "implementation_complexity": "medium",
                "priority": "medium"
            })
        
        # Auto-scaling recommendation
        peak_to_avg = (usage_analysis.get("peak_hourly_cost", 0) / 
                       max(usage_analysis.get("avg_hourly_cost", 1), 1))
        
        if peak_to_avg > 3:
            savings = usage_analysis["total_cost"] * 0.25  # Estimate 25% savings
            recommendations.append({
                "strategy": OptimizationStrategy.AUTO_SCALING.value,
                "recommendation": "Implement auto-scaling to handle peak loads efficiently",
                "potential_savings_monthly": savings,
                "implementation_complexity": "high",
                "priority": "high"
            })
        
        return recommendations
    
    def track_metric(self, user_id: int, metric_type: CostMetricType, amount: float,
                   model_name: Optional[str] = None, request_id: Optional[str] = None,
                   session_id: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Track a cost metric"""
        now = datetime.utcnow()
        
        # Calculate unit cost
        unit_cost = self.get_unit_cost(metric_type)
        total_cost = amount * unit_cost
        
        # Create time keys
        hour_key = now.strftime("%Y-%m-%d-%H")
        day_key = now.strftime("%Y-%m-%d")
        month_key = now.strftime("%Y-%m")
        
        metric = CostMetric(
            user_id=user_id,
            metric_type=metric_type.value,
            amount=amount,
            unit_cost=unit_cost,
            total_cost=total_cost,
            model_name=model_name,
            request_id=request_id,
            session_id=session_id,
            metric_metadata=metadata or {},
            hour_key=hour_key,
            day_key=day_key,
            month_key=month_key,
            created_at=now
        )
        
        # Store in database
        with get_db_session() as session:
            session.add(metric)
            session.commit()
            session.refresh(metric)
        
        return {
            "metric": metric.to_dict(),
            "cost": total_cost,
            "period": f"{month_key} (hour: {hour_key})"
        }
    
    def get_unit_cost(self, metric_type: CostMetricType) -> float:
        """Get unit cost for a metric type"""
        cost_map = {
            CostMetricType.TOKEN_INFERENCE: self.pricing.inference_input_cost_per_1k / 1000,
            CostMetricType.TOKEN_TRAINING: self.pricing.training_cost_per_1k / 1000,
            CostMetricType.MODEL_STORAGE: self.pricing.model_storage_cost_per_gb_month / (30 * 24),  # Per hour
            CostMetricType.DATA_STORAGE: self.pricing.data_storage_cost_per_gb_month / (30 * 24),  # Per hour
            CostMetricType.API_CALL: self.pricing.api_call_cost_per_1k / 1000,
            CostMetricType.COMPUTE_TIME: self.pricing.compute_cost_per_hour,
            CostMetricType.BANDWIDTH: self.pricing.bandwidth_cost_per_gb / 1024  # Per GB
        }
        return cost_map.get(metric_type, 0.001)

class BudgetManager:
    """Budget management and alerting"""
    
    def __init__(self, cost_optimizer: CostOptimizer):
        self.cost_optimizer = cost_optimizer
    
    def check_budget_status(self, user_id: int, config: BudgetConfig) -> Dict[str, Any]:
        """Check budget status and generate alerts if needed"""
        now = datetime.utcnow()
        
        # Get current spending
        monthly_spending = self._get_spending(user_id, "month")
        daily_spending = self._get_spending(user_id, "day")
        hourly_spending = self._get_spending(user_id, "hour")
        
        # Calculate percentages
        monthly_percentage = monthly_spending / max(config.monthly_limit, 1)
        
        daily_percentage = 0
        if config.daily_limit:
            daily_percentage = daily_spending / max(config.daily_limit, 1)
            
        hourly_percentage = 0  
        if config.hourly_limit:
            hourly_percentage = hourly_spending / max(config.hourly_limit, 1)
        
        # Determine status
        status = "healthy"
        alerts = []
        
        # Check critical threshold
        if (monthly_percentage >= config.critical_threshold or
            daily_percentage >= config.critical_threshold or
            hourly_percentage >= config.critical_threshold):
            status = "critical"
            alerts.append({
                "level": "critical",
                "message": "Budget critical threshold exceeded",
                "monthly_percentage": monthly_percentage,
                "daily_percentage": daily_percentage,
                "hourly_percentage": hourly_percentage
            })
        
        # Check warning threshold
        elif (monthly_percentage >= config.warning_threshold or
              daily_percentage >= config.warning_threshold or
              hourly_percentage >= config.warning_threshold):
            status = "warning"
            alerts.append({
                "level": "warning",
                "message": "Budget warning threshold reached",
                "monthly_percentage": monthly_percentage,
                "daily_percentage": daily_percentage,
                "hourly_percentage": hourly_percentage
            })
        
        # Create alert records if needed
        for alert in alerts:
            self._create_budget_alert(user_id, alert, config, now)
        
        return {
            "status": status,
            "monthly_spending": monthly_spending,
            "daily_spending": daily_spending,
            "hourly_spending": hourly_spending,
            "monthly_percentage": monthly_percentage,
            "daily_percentage": daily_percentage,
            "hourly_percentage": hourly_percentage,
            "budget_limits": {
                "monthly": config.monthly_limit,
                "daily": config.daily_limit,
                "hourly": config.hourly_limit
            },
            "alerts": alerts
        }
    
    def _get_spending(self, user_id: int, period: str) -> float:
        """Get spending for a specific period"""
        now = datetime.utcnow()
        
        with get_db_session() as session:
            query = session.query(CostMetric).filter(CostMetric.user_id == user_id)
            
            if period == "hour":
                start_time = now.replace(minute=0, second=0, microsecond=0)
            elif period == "day":
                start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif period == "month":
                start_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                return 0.0
            
            metrics = query.filter(CostMetric.created_at >= start_time).all()
            return sum(m.total_cost for m in metrics)
    
    def _create_budget_alert(self, user_id: int, alert: Dict[str, Any], 
                           config: BudgetConfig, timestamp: datetime):
        """Create budget alert record"""
        with get_db_session() as session:
            # Check if similar alert already exists in last hour
            recent_alert = session.query(BudgetAlert).filter(
                BudgetAlert.user_id == user_id,
                BudgetAlert.alert_type == alert["level"],
                BudgetAlert.created_at >= timestamp - timedelta(hours=1)
            ).first()
            
            if recent_alert:
                return  # Don't create duplicate alerts
            
            # Create new alert
            budget_alert = BudgetAlert(
                user_id=user_id,
                alert_type=alert["level"],
                threshold_percentage=alert.get("monthly_percentage", 0) * 100,
                current_spending=alert.get("monthly_percentage", 0) * config.monthly_limit,
                budget_limit=config.monthly_limit,
                period="monthly",
                alert_sent=False
            )
            
            session.add(budget_alert)
            session.commit()
            
            # Send notifications (simplified)
            self._send_notification(budget_alert, config)
    
    def _send_notification(self, alert: BudgetAlert, config: BudgetConfig):
        """Send notification for budget alert"""
        # This is a placeholder for actual notification implementation
        logger.info(f"Budget Alert: {alert.alert_type} for user {alert.user_id}")
        
        if config.email_alerts:
            # Send email notification
            pass
            
        if config.webhook_url:
            # Send webhook notification
            pass
            
        if config.slack_channel:
            # Send Slack notification  
            pass
    
    def get_cost_forecast(self, user_id: int, days_ahead: int = 30) -> Dict[str, Any]:
        """Generate cost forecast based on historical patterns"""
        analysis = self.cost_optimizer.analyze_usage_patterns(user_id, days=30)
        
        if "error" in analysis:
            return {"error": "Insufficient data for forecast"}
        
        # Simple linear forecast
        avg_daily_cost = analysis.get("avg_daily_cost", 0)
        forecast_total = avg_daily_cost * days_ahead
        
        # Add seasonal/daily variations
        peak_multiplier = analysis.get("peak_daily_cost", avg_daily_cost) / max(avg_daily_cost, 1)
        
        return {
            "forecast_days": days_ahead,
            "avg_daily_cost": avg_daily_cost,
            "peak_daily_cost": analysis.get("peak_daily_cost", 0),
            "forecast_total": forecast_total,
            "peak_multiplier": peak_multiplier,
            "confidence": "medium",  # Based on data quality and patterns
            "recommendations": [
                "Consider cost optimization if forecast exceeds budget",
                "Monitor peak usage periods for optimization opportunities"
            ]
        }

# Global instances
_cost_optimizer = None
_budget_manager = None

def get_cost_optimizer() -> CostOptimizer:
    """Get global cost optimizer instance"""
    global _cost_optimizer
    if _cost_optimizer is None:
        _cost_optimizer = CostOptimizer()
    return _cost_optimizer

def get_budget_manager() -> BudgetManager:
    """Get global budget manager instance"""
    global _budget_manager
    if _budget_manager is None:
        _budget_manager = BudgetManager(get_cost_optimizer())
    return _budget_manager

# Convenience functions
def track_inference_cost(user_id: int, input_tokens: int, output_tokens: int, 
                       model_name: Optional[str] = None, request_id: Optional[str] = None):
    """Track inference cost"""
    optimizer = get_cost_optimizer()
    
    # Track input and output tokens separately
    optimizer.track_metric(
        user_id=user_id,
        metric_type=CostMetricType.TOKEN_INFERENCE,
        amount=input_tokens + output_tokens,
        model_name=model_name,
        request_id=request_id,
        metadata={"input_tokens": input_tokens, "output_tokens": output_tokens}
    )

def check_user_budget(user_id: int, config: BudgetConfig) -> Dict[str, Any]:
    """Check user budget status"""
    manager = get_budget_manager()
    return manager.check_budget_status(user_id, config)

# CLI interface
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="SloughGPT Cost Optimization")
    parser.add_argument("command", choices=["analyze", "forecast", "recommendations", "track"],
                       help="Command to execute")
    parser.add_argument("--user-id", type=int, help="User ID")
    parser.add_argument("--amount", type=float, help="Amount for tracking")
    parser.add_argument("--type", help="Metric type")
    parser.add_argument("--days", type=int, default=30, help="Analysis period in days")
    parser.add_argument("--forecast-days", type=int, default=30, help="Forecast period in days")
    
    def main():
        args = parser.parse_args()
        
        if not args.user_id and args.command != "track":
            print("❌ User ID required for this command")
            sys.exit(1)
        
        optimizer = get_cost_optimizer()
        manager = get_budget_manager()
        
        if args.command == "analyze":
            analysis = optimizer.analyze_usage_patterns(args.user_id, args.days)
            print(f"📊 Cost Analysis for User {args.user_id} ({args.days} days):")
            if "error" in analysis:
                print(f"   ❌ {analysis['error']}")
            else:
                print(f"   💰 Total Cost: ${analysis['total_cost']:.2f}")
                print(f"   📈 Avg Daily: ${analysis['avg_daily_cost']:.2f}")
                print(f"   ⚡ Peak Daily: ${analysis['peak_daily_cost']:.2f}")
                print(f"   🔍 Data Points: {analysis['data_points']}")
                
                for insight in analysis.get("insights", []):
                    print(f"   💡 {insight['message']}")
        
        elif args.command == "forecast":
            forecast = manager.get_cost_forecast(args.user_id, args.forecast_days)
            print(f"🔮 Cost Forecast for User {args.user_id} ({args.forecast_days} days):")
            if "error" in forecast:
                print(f"   ❌ {forecast['error']}")
            else:
                print(f"   💰 Forecast Total: ${forecast['forecast_total']:.2f}")
                print(f"   📈 Avg Daily: ${forecast['avg_daily_cost']:.2f}")
                print(f"   ⚡ Peak Daily: ${forecast['peak_daily_cost']:.2f}")
        
        elif args.command == "recommendations":
            recommendations = optimizer.generate_optimization_recommendations(args.user_id)
            print(f"💡 Optimization Recommendations for User {args.user_id}:")
            if not recommendations:
                print("   ✅ No optimizations needed at this time")
            else:
                for i, rec in enumerate(recommendations, 1):
                    print(f"\n   {i}. {rec['strategy']}")
                    print(f"      📝 {rec['recommendation']}")
                    print(f"      💰 Potential Savings: ${rec['potential_savings_monthly']:.2f}/month")
                    print(f"      🔧 Complexity: {rec['implementation_complexity']}")
                    print(f"      🎯 Priority: {rec['priority']}")
        
        elif args.command == "track":
            if not all([args.user_id, args.amount, args.type]):
                print("❌ User ID, amount, and type required for tracking")
                sys.exit(1)
                
            try:
                metric_type = CostMetricType(args.type)
                result = optimizer.track_metric(args.user_id, metric_type, args.amount)
                print(f"✅ Tracked {args.amount} {args.type} - Cost: ${result['cost']:.4f}")
            except ValueError:
                print(f"❌ Invalid metric type. Valid types: {[t.value for t in CostMetricType]}")
    
    main()