#!/usr/bin/env python3
"""
SloughGPT Cost Optimization Demo
Demonstrates cost tracking, budget management, and optimization features
"""

import asyncio
import logging
from datetime import datetime, timedelta
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_cost_optimization():
    """Demonstrate cost optimization capabilities"""
    
    try:
        from sloughgpt.cost_optimization import (
            CostOptimizer, BudgetManager, BudgetConfig, CostMetricType,
            get_cost_optimizer, get_budget_manager, track_inference_cost
        )
    except ImportError as e:
        logger.error(f"Cost optimization not available: {e}")
        logger.info("Install cost optimization dependencies with: pip install numpy")
        return
        
    logger.info("💰 SloughGPT Cost Optimization Demo")
    logger.info("=" * 50)
    
    # Initialize components
    optimizer = get_cost_optimizer()
    budget_manager = get_budget_manager()
    
    # Create sample user ID for demo
    user_id = 1
    
    # Demo: Track various cost metrics
    logger.info("📊 Tracking cost metrics...")
    
    # Simulate inference costs
    for i in range(10):
        input_tokens = random.randint(50, 200)
        output_tokens = random.randint(20, 100)
        
        result = optimizer.track_metric(
            user_id=user_id,
            metric_type=CostMetricType.TOKEN_INFERENCE,
            amount=input_tokens + output_tokens,
            model_name="sloughgpt-base",
            request_id=f"req_{i}",
            metadata={"input_tokens": input_tokens, "output_tokens": output_tokens}
        )
        
        logger.info(f"   Request {i+1}: {input_tokens}+{output_tokens} tokens = ${result['cost']:.4f}")
    
    # Simulate training costs
    training_result = optimizer.track_metric(
        user_id=user_id,
        metric_type=CostMetricType.TOKEN_TRAINING,
        amount=50000,  # 50K tokens
        model_name="sloughgpt-base",
        metadata={"epoch": 1, "batch_size": 32}
    )
    logger.info(f"   Training: 50K tokens = ${training_result['cost']:.2f}")
    
    # Simulate storage costs
    storage_result = optimizer.track_metric(
        user_id=user_id,
        metric_type=CostMetricType.MODEL_STORAGE,
        amount=2.5,  # 2.5 GB
        metadata={"model_size": "2.5GB", "compression": "quantized"}
    )
    logger.info(f"   Storage: 2.5 GB = ${storage_result['cost']:.3f}")
    
    # Demo: Cost Analysis
    logger.info("\n📈 Analyzing usage patterns...")
    
    analysis = optimizer.analyze_usage_patterns(user_id, days=30)
    
    if "error" not in analysis:
        logger.info(f"   Total Cost (30 days): ${analysis['total_cost']:.2f}")
        logger.info(f"   Average Daily: ${analysis['avg_daily_cost']:.2f}")
        logger.info(f"   Peak Daily: ${analysis['peak_daily_cost']:.2f}")
        logger.info(f"   Average Hourly: ${analysis['avg_hourly_cost']:.3f}")
        logger.info(f"   Peak Hourly: ${analysis['peak_hourly_cost']:.3f}")
        logger.info(f"   Data Points: {analysis['data_points']}")
        
        logger.info("\n💡 Usage Insights:")
        for insight in analysis.get("insights", []):
            logger.info(f"   • {insight['message']}")
            logger.info(f"     Recommendation: {insight['recommendation']}")
        
        logger.info("\n📊 Cost Breakdown:")
        for metric_type, data in analysis.get("cost_by_type", {}).items():
            logger.info(f"   {metric_type}: ${data['cost']:.2f} ({data['amount']:.0f} units)")
    else:
        logger.info(f"   ❌ {analysis['error']}")
    
    # Demo: Budget Management
    logger.info("\n💵 Testing budget management...")
    
    # Create budget configuration
    budget_config = BudgetConfig(
        monthly_limit=50.0,
        daily_limit=5.0,
        warning_threshold=0.7,
        critical_threshold=0.9,
        auto_optimization=True
    )
    
    budget_status = budget_manager.check_budget_status(user_id, budget_config)
    
    logger.info(f"   Status: {budget_status['status']}")
    logger.info(f"   Monthly: ${budget_status['monthly_spending']:.2f} / ${budget_status['budget_limits']['monthly']:.2f}")
    logger.info(f"   Percentage: {budget_status['monthly_percentage']:.1%}")
    
    if budget_status.get("alerts"):
        logger.info("   🚨 Alerts:")
        for alert in budget_status["alerts"]:
            logger.info(f"      • {alert['level'].upper()}: {alert['message']}")
    else:
        logger.info("   ✅ No budget alerts")
    
    # Demo: Cost Forecasting
    logger.info("\n🔮 Generating cost forecast...")
    
    forecast = budget_manager.get_cost_forecast(user_id, days_ahead=30)
    
    if "error" not in forecast:
        logger.info(f"   Forecast (30 days): ${forecast['forecast_total']:.2f}")
        logger.info(f"   Average Daily: ${forecast['avg_daily_cost']:.2f}")
        logger.info(f"   Peak Daily: ${forecast['peak_daily_cost']:.2f}")
        logger.info(f"   Peak Multiplier: {forecast['peak_multiplier']:.1f}x")
        logger.info(f"   Confidence: {forecast['confidence']}")
        
        logger.info("\n📝 Forecast Recommendations:")
        for rec in forecast.get("recommendations", []):
            logger.info(f"   • {rec}")
    else:
        logger.info(f"   ❌ {forecast['error']}")
    
    # Demo: Optimization Recommendations
    logger.info("\n💡 Generating optimization recommendations...")
    
    recommendations = optimizer.generate_optimization_recommendations(user_id)
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"\n   {i}. {rec['strategy']}")
            logger.info(f"      📝 {rec['recommendation']}")
            logger.info(f"      💰 Potential Savings: ${rec['potential_savings_monthly']:.2f}/month")
            logger.info(f"      🔧 Complexity: {rec['implementation_complexity']}")
            logger.info(f"      🎯 Priority: {rec['priority']}")
    else:
        logger.info("   ✅ No optimizations needed at this time")
    
    logger.info("\n🎉 Cost Optimization Demo Complete!")

def demo_advanced_features():
    """Demonstrate advanced cost optimization features"""
    
    logger.info("\n🚀 Advanced Cost Optimization Features")
    logger.info("=" * 40)
    
    try:
        from sloughgpt.cost_optimization import (
            CostOptimizer, BudgetConfig, CostMetricType, OptimizationStrategy
        )
        
        optimizer = get_cost_optimizer()
        user_id = 2
        
        # Demo: Different cost calculation scenarios
        logger.info("💰 Cost Calculation Examples:")
        
        # Inference costs
        inference_costs = optimizer.calculate_inference_cost(
            input_tokens=150, output_tokens=75, model_name="sloughgpt-large"
        )
        logger.info(f"   Inference (150+75 tokens): ${inference_costs:.4f}")
        
        # Training costs
        training_costs = optimizer.calculate_training_cost(
            tokens=100000, compute_hours=8.0
        )
        logger.info(f"   Training (100K tokens, 8 hours): ${training_costs:.2f}")
        
        # Storage costs
        storage_costs = optimizer.calculate_storage_cost(gb_size=5.0, duration_days=30)
        logger.info(f"   Storage (5GB, 30 days): ${storage_costs:.2f}")
        
        # Demo: Track multiple metric types
        logger.info("\n📊 Tracking Multiple Metric Types:")
        
        metrics_to_track = [
            (CostMetricType.TOKEN_INFERENCE, 250, "High-volume inference"),
            (CostMetricType.API_CALL, 1000, "API usage burst"),
            (CostMetricType.COMPUTE_TIME, 2.5, "GPU compute hours"),
            (CostMetricType.BANDWIDTH, 10.0, "Data transfer")
        ]
        
        for metric_type, amount, description in metrics_to_track:
            result = optimizer.track_metric(
                user_id=user_id,
                metric_type=metric_type,
                amount=amount,
                metadata={"description": description, "demo": True}
            )
            logger.info(f"   {description}: ${result['cost']:.4f}")
        
        # Demo: Budget with different thresholds
        logger.info("\n💵 Advanced Budget Configuration:")
        
        advanced_budget = BudgetConfig(
            monthly_limit=100.0,
            daily_limit=10.0,
            hourly_limit=1.0,
            warning_threshold=0.6,
            critical_threshold=0.85,
            auto_optimization=True,
            optimization_strategies=[
                OptimizationStrategy.MODEL_QUANTIZATION,
                OptimizationStrategy.CACHED_RESPONSES,
                OptimizationStrategy.AUTO_SCALING
            ]
        )
        
        logger.info(f"   Monthly Limit: ${advanced_budget.monthly_limit}")
        logger.info(f"   Daily Limit: ${advanced_budget.daily_limit}")
        logger.info(f"   Hourly Limit: ${advanced_budget.hourly_limit}")
        logger.info(f"   Warning Threshold: {advanced_budget.warning_threshold:.0%}")
        logger.info(f"   Critical Threshold: {advanced_budget.critical_threshold:.0%}")
        logger.info(f"   Auto-Optimization: {advanced_budget.auto_optimization}")
        logger.info(f"   Strategies: {len(advanced_budget.optimization_strategies)} active")
        
        # Demo: Custom pricing
        logger.info("\n🏷️ Custom Pricing Configuration:")
        
        from sloughgpt.cost_optimization import CostPricing
        custom_pricing = CostPricing(
            inference_input_cost_per_1k=0.0005,  # Discounted rate
            inference_output_cost_per_1k=0.0015,
            training_cost_per_1k=0.003,
            model_storage_cost_per_gb_month=0.05,
            compute_cost_per_hour=0.25
        )
        
        custom_optimizer = CostOptimizer(custom_pricing)
        
        custom_cost = custom_optimizer.calculate_inference_cost(1000, 500)
        standard_cost = optimizer.calculate_inference_cost(1000, 500)
        
        logger.info(f"   Standard Pricing: ${standard_cost:.4f}")
        logger.info(f"   Custom Pricing: ${custom_cost:.4f}")
        logger.info(f"   Savings: {((standard_cost - custom_cost) / standard_cost * 100):.1f}%")
        
    except Exception as e:
        logger.error(f"❌ Advanced features error: {str(e)}")

if __name__ == "__main__":
    print("💰 SloughGPT Cost Optimization Demo")
    print("=" * 50)
    
    # Run basic demo
    demo_cost_optimization()
    
    # Run advanced demo
    demo_advanced_features()
    
    print("\n" + "=" * 50)
    print("🎯 Try these CLI commands:")
    print("   python3 sloughgpt/cost_optimization.py analyze --user-id 1")
    print("   python3 sloughgpt/cost_optimization.py forecast --user-id 1 --days 30")
    print("   python3 sloughgpt/cost_optimization.py recommendations --user-id 1")
    print("   python3 sloughgpt/cost_optimization.py track --user-id 1 --amount 1000 --type token_inference")
    print("\n🔧 Cost Optimization Features:")
    print("   • Real-time cost tracking")
    print("   • Budget management and alerts")
    print("   • Usage pattern analysis")
    print("   • Cost forecasting")
    print("   • Optimization recommendations")
    print("   • Custom pricing support")
    print("   • Multi-metric tracking")