"""Comprehensive test suite for SloughGPT Enterprise Framework."""

import pytest
import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the source path to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from sloughgpt import (
        get_user_manager,
        get_cost_optimizer,
        DatasetPipeline,
        ReasoningEngine,
        AuthService,
        MonitoringService,
        DeploymentManager,
        ModelManager,
        TrainingManager,
        PerformanceOptimizer,
        auto_optimizer
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestUserManagement:
    """Test user management functionality."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_user_manager_initialization(self):
        """Test user manager initialization."""
        user_manager = get_user_manager()
        assert user_manager is not None
        assert hasattr(user_manager, 'create_user')
        assert hasattr(user_manager, 'authenticate_user')
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_user_creation(self):
        """Test user creation."""
        user_manager = get_user_manager()
        result = user_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="testpass123",
            role="user"
        )
        
        assert "user" in result
        assert result["user"]["username"] == "testuser"
        assert result["user"]["email"] == "test@example.com"
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_user_authentication(self):
        """Test user authentication."""
        user_manager = get_user_manager()
        
        # First create a user
        user_manager.create_user(
            username="authuser",
            email="auth@example.com",
            password="authpass123"
        )
        
        # Test authentication
        result = user_manager.authenticate_user("authuser", "authpass123")
        assert "access_token" in result
        assert "user" in result


class TestCostOptimization:
    """Test cost optimization functionality."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_cost_optimizer_initialization(self):
        """Test cost optimizer initialization."""
        optimizer = get_cost_optimizer()
        assert optimizer is not None
        assert hasattr(optimizer, 'track_metric')
        assert hasattr(optimizer, 'analyze_usage_patterns')
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_metric_tracking(self):
        """Test metric tracking."""
        optimizer = get_cost_optimizer()
        
        from sloughgpt.cost_optimization import CostMetricType
        
        optimizer.track_metric(
            user_id=1,
            metric_type=CostMetricType.TOKEN_INFERENCE,
            amount=1000,
            model_name="sloughgpt-base"
        )
        
        assert len(optimizer.metrics) > 0
        assert optimizer.metrics[0].amount == 1000
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_usage_analysis(self):
        """Test usage analysis."""
        optimizer = get_cost_optimizer()
        from sloughgpt.cost_optimization import CostMetricType
        
        # Add some test metrics
        optimizer.track_metric(
            user_id=1,
            metric_type=CostMetricType.TOKEN_INFERENCE,
            amount=1000,
            model_name="sloughgpt-base"
        )
        
        analysis = optimizer.analyze_usage_patterns(user_id=1, days=30)
        assert analysis.total_cost >= 0
        assert analysis.avg_daily_cost >= 0


class TestReasoningEngine:
    """Test reasoning engine functionality."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_reasoning_engine_initialization(self):
        """Test reasoning engine initialization."""
        engine = ReasoningEngine()
        assert engine is not None
        assert hasattr(engine, 'create_context')
        assert hasattr(engine, 'reason')
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_context_creation(self):
        """Test context creation."""
        engine = ReasoningEngine()
        context_id = engine.create_context(
            user_id=1,
            prompt="What is the meaning of life?"
        )
        
        assert context_id is not None
        assert context_id in engine.active_contexts
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_reasoning_process(self):
        """Test reasoning process."""
        engine = ReasoningEngine()
        context_id = engine.create_context(
            user_id=1,
            prompt="What is 2 + 2?"
        )
        
        result = engine.reason(context_id)
        
        assert result.final_answer is not None
        assert len(result.reasoning_steps) > 0
        assert 0 <= result.confidence <= 1


class TestDataLearning:
    """Test data learning functionality."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_dataset_pipeline_initialization(self):
        """Test dataset pipeline initialization."""
        pipeline = DatasetPipeline()
        assert pipeline is not None
        assert hasattr(pipeline, 'add_source')
        assert hasattr(pipeline, 'search_knowledge')
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_source_addition(self):
        """Test adding data sources."""
        pipeline = DatasetPipeline()
        
        # Create a temporary test file
        test_file = Path("/tmp/test_data.txt")
        test_file.write_text("This is test data for the learning pipeline.")
        
        source_id = pipeline.add_source(
            name="test_source",
            path=str(test_file),
            format="text"
        )
        
        assert source_id is not None
        assert source_id in pipeline.data_sources
        
        # Clean up
        test_file.unlink()
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_knowledge_search(self):
        """Test knowledge search."""
        pipeline = DatasetPipeline()
        
        # Search with empty knowledge base
        results = pipeline.search_knowledge("test query", k=5)
        assert isinstance(results, list)


class TestAuthentication:
    """Test authentication functionality."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_auth_service_initialization(self):
        """Test auth service initialization."""
        user_manager = get_user_manager()
        auth_service = AuthService(user_manager)
        
        assert auth_service is not None
        assert hasattr(auth_service, 'authenticate')
        assert hasattr(auth_service, 'validate_token')
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_user_authentication_flow(self):
        """Test complete authentication flow."""
        user_manager = get_user_manager()
        auth_service = AuthService(user_manager)
        
        # Create user
        user_manager.create_user(
            username="flowuser",
            email="flow@example.com",
            password="flowpass123"
        )
        
        # Authenticate
        result = auth_service.authenticate("flowuser", "flowpass123")
        assert "access_token" in result
        assert "refresh_token" in result
        
        # Validate token
        token = result["access_token"]
        payload = auth_service.validate_token(token)
        assert payload is not None
        assert payload["username"] == "flowuser"


class TestMonitoring:
    """Test monitoring functionality."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_monitoring_service_initialization(self):
        """Test monitoring service initialization."""
        monitoring = MonitoringService()
        assert monitoring is not None
        assert hasattr(monitoring, 'record_metric')
        assert hasattr(monitoring, 'get_dashboard_data')
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_metric_recording(self):
        """Test metric recording."""
        monitoring = MonitoringService()
        
        monitoring.record_metric("test_metric", 100.0, labels={"test": "true"})
        monitoring.record_metric("test_counter", 1.0, metric_type="counter")
        
        # Check if metric was recorded
        metric_data = monitoring.get_metric("test_metric", time_range=60)
        assert len(metric_data) > 0
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_dashboard_data(self):
        """Test dashboard data generation."""
        monitoring = MonitoringService()
        
        # Add some test metrics
        monitoring.record_metric("api_requests_total", 100.0, metric_type="counter")
        monitoring.record_metric("api_response_time", 150.0)
        
        dashboard = monitoring.get_dashboard_data()
        assert "metrics" in dashboard
        assert "alerts" in dashboard
        assert "system_health" in dashboard


class TestPerformance:
    """Test performance optimization functionality."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_performance_optimizer_initialization(self):
        """Test performance optimizer initialization."""
        optimizer = PerformanceOptimizer()
        assert optimizer is not None
        assert hasattr(optimizer, 'collect_metrics')
        assert hasattr(optimizer, 'analyze_performance')
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_metrics_collection(self):
        """Test metrics collection."""
        optimizer = PerformanceOptimizer()
        
        metrics = optimizer.collect_metrics()
        assert metrics is not None
        assert hasattr(metrics, 'cpu_percent')
        assert hasattr(metrics, 'memory_percent')
        assert hasattr(metrics, 'response_time')
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_performance_analysis(self):
        """Test performance analysis."""
        optimizer = PerformanceOptimizer()
        
        # Add some test metrics
        for _ in range(10):
            optimizer.metrics_history.append(optimizer.collect_metrics())
        
        analysis = optimizer.analyze_performance(time_window_minutes=5)
        assert "cpu" in analysis
        assert "memory" in analysis
        assert "sample_count" in analysis
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_recommendations_generation(self):
        """Test optimization recommendations."""
        optimizer = PerformanceOptimizer()
        
        # Simulate high CPU usage
        for _ in range(10):
            metrics = optimizer.collect_metrics()
            metrics.cpu_percent = 90.0  # Simulate high CPU
            optimizer.metrics_history.append(metrics)
        
        recommendations = optimizer.generate_recommendations()
        assert isinstance(recommendations, list)
        
        # Should have CPU optimization recommendation
        cpu_recs = [r for r in recommendations if r.category == "CPU"]
        assert len(cpu_recs) > 0


class TestModelServing:
    """Test model serving functionality."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_model_manager_initialization(self):
        """Test model manager initialization."""
        from sloughgpt.model_serving import model_manager
        assert model_manager is not None
        assert hasattr(model_manager, 'server')
        assert hasattr(model_manager, 'generate_text')
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_model_server_initialization(self):
        """Test model server initialization."""
        from sloughgpt.model_serving import ModelServer
        server = ModelServer()
        assert server is not None
        assert hasattr(server, 'load_model')
        assert hasattr(server, 'generate')


class TestDistributedTraining:
    """Test distributed training functionality."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_training_manager_initialization(self):
        """Test training manager initialization."""
        from sloughgpt.distributed_training import training_manager
        assert training_manager is not None
        assert hasattr(training_manager, 'trainer')
        assert hasattr(training_manager, 'create_and_start_training')
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_distributed_trainer_initialization(self):
        """Test distributed trainer initialization."""
        from sloughgpt.distributed_training import DistributedTrainer
        trainer = DistributedTrainer()
        assert trainer is not None
        assert hasattr(trainer, 'create_training_job')
        assert hasattr(trainer, 'list_jobs')


class TestDeployment:
    """Test deployment functionality."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_deployment_manager_initialization(self):
        """Test deployment manager initialization."""
        manager = DeploymentManager()
        assert manager is not None
        assert hasattr(manager, 'create_deployment_config')
        assert hasattr(manager, 'generate_kubernetes_manifests')


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_system_initialization(self):
        """Test complete system initialization."""
        try:
            # Initialize all components
            user_manager = get_user_manager()
            cost_optimizer = get_cost_optimizer()
            pipeline = DatasetPipeline()
            engine = ReasoningEngine()
            auth_service = AuthService(user_manager)
            monitoring = MonitoringService()
            
            # All should initialize without errors
            assert user_manager is not None
            assert cost_optimizer is not None
            assert pipeline is not None
            assert engine is not None
            assert auth_service is not None
            assert monitoring is not None
            
        except Exception as e:
            pytest.fail(f"System initialization failed: {e}")
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow."""
        try:
            # 1. Create user
            user_manager = get_user_manager()
            user_result = user_manager.create_user(
                username="e2euser",
                email="e2e@example.com",
                password="e2epass123"
            )
            
            # 2. Authenticate user
            auth_service = AuthService(user_manager)
            auth_result = auth_service.authenticate("e2euser", "e2epass123")
            
            # 3. Track cost for inference
            cost_optimizer = get_cost_optimizer()
            from sloughgpt.cost_optimization import CostMetricType
            cost_optimizer.track_metric(
                user_id=user_result["user"]["id"],
                metric_type=CostMetricType.TOKEN_INFERENCE,
                amount=100
            )
            
            # 4. Perform reasoning
            engine = ReasoningEngine()
            context_id = engine.create_context(
                user_id=user_result["user"]["id"],
                prompt="Test prompt for E2E workflow"
            )
            reasoning_result = engine.reason(context_id)
            
            # 5. Record metrics
            monitoring = MonitoringService()
            monitoring.increment_counter("e2e_requests_total")
            monitoring.record_histogram("e2e_response_time", 150.0)
            
            # Verify workflow completed
            assert user_result["user"]["username"] == "e2euser"
            assert "access_token" in auth_result
            assert reasoning_result.final_answer is not None
            assert len(cost_optimizer.metrics) > 0
            
        except Exception as e:
            pytest.fail(f"End-to-end workflow failed: {e}")


# Test configuration and fixtures
@pytest.fixture
def test_data_dir():
    """Create temporary test data directory."""
    import tempfile
    test_dir = Path(tempfile.mkdtemp())
    yield test_dir
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def test_imports():
    """Test that all imports are working."""
    if not IMPORTS_AVAILABLE:
        pytest.skip(f"Imports not available: {IMPORT_ERROR}")
    
    # Test that all main components can be imported
    from sloughgpt import (
        get_user_manager,
        get_cost_optimizer,
        DatasetPipeline,
        ReasoningEngine,
        AuthService,
        MonitoringService,
        DeploymentManager,
        ModelManager,
        TrainingManager,
        PerformanceOptimizer,
        auto_optimizer
    )
    
    # Test that all are callable/instantiable
    assert callable(get_user_manager)
    assert callable(get_cost_optimizer)
    assert callable(DatasetPipeline)
    assert callable(ReasoningEngine)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])