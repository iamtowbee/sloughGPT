"""
Simple Test Runner for SloughGPT

A basic test runner that can validate the core functionality
without complex import issues.
"""

import sys
import os
import asyncio
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all major modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        # Test basic imports
        import sloughgpt.config
        print("✅ Config module imported")
        
        import sloughgpt.neural_network
        print("✅ Neural network module imported")
        
        import sloughgpt.auth
        print("✅ Auth module imported")
        
        import sloughgpt.user_management
        print("✅ User management module imported")
        
        import sloughgpt.cost_optimization
        print("✅ Cost optimization module imported")
        
        import sloughgpt.data_learning
        print("✅ Data learning module imported")
        
        import sloughgpt.reasoning_engine
        print("✅ Reasoning engine module imported")
        
        import sloughgpt.api_server
        print("✅ API server module imported")
        
        import sloughgpt.admin
        print("✅ Admin dashboard module imported")
        
        import sloughgpt.trainer
        print("✅ Trainer module imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

async def test_basic_functionality():
    """Test basic functionality of core components"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test configuration
        from sloughgpt.config import SloughGPTConfig
        
        config = SloughGPTConfig()
        assert config.model_config is not None
        assert config.learning_config is not None
        print("✅ Configuration creation works")
        
        # Test database manager
        from sloughgpt.core.database import DatabaseManager
        
        db_config = {"database_url": "sqlite:///:memory:"}
        db_manager = DatabaseManager(db_config)
        
        await db_manager.initialize()
        health = await db_manager.health_check()
        assert health["status"] == "healthy"
        await db_manager.close()
        print("✅ Database operations work")
        
        # Test logging system
        from sloughgpt.core.logging_system import get_logger
        
        logger = get_logger("test")
        logger.info("Test log message")
        print("✅ Logging system works")
        
        # Test security middleware
        from sloughgpt.core.security import SecurityMiddleware
        
        security = SecurityMiddleware()
        print("✅ Security middleware works")
        
        # Test performance optimizer
        from sloughgpt.core.performance import PerformanceOptimizer
        
        perf = PerformanceOptimizer()
        print("✅ Performance optimizer works")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

async def test_api_server():
    """Test API server functionality"""
    print("\n🌐 Testing API server...")
    
    try:
        from sloughgpt.api_server import app
        
        # Test FastAPI app creation
        assert app is not None
        assert app.title == "SloughGPT API Server"
        print("✅ API server app created successfully")
        
        # Test with test client
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        print("✅ Health endpoint works")
        
        # Test info endpoint
        response = client.get("/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "version" in data
        print("✅ Info endpoint works")
        
        return True
        
    except Exception as e:
        print(f"❌ API server test failed: {e}")
        return False

async def test_admin_dashboard():
    """Test admin dashboard functionality"""
    print("\n📊 Testing admin dashboard...")
    
    try:
        from sloughgpt.admin import create_app, get_system_metrics
        
        # Test admin app creation
        admin_app = create_app()
        assert admin_app is not None
        print("✅ Admin dashboard app created successfully")
        
        # Test system metrics
        metrics = await get_system_metrics()
        assert metrics is not None
        assert "status" in metrics
        print("✅ System metrics collection works")
        
        # Test with test client
        from fastapi.testclient import TestClient
        
        client = TestClient(admin_app)
        
        # Test dashboard stats endpoint
        response = client.get("/api/admin/dashboard/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "system_status" in data
        assert "user_count" in data
        print("✅ Admin dashboard stats endpoint works")
        
        return True
        
    except Exception as e:
        print(f"❌ Admin dashboard test failed: {e}")
        return False

async def test_integration_workflow():
    """Test simple integration workflow"""
    print("\n🔄 Testing integration workflow...")
    
    try:
        # Create configuration
        from sloughgpt.config import SloughGPTConfig
        
        config = SloughGPTConfig()
        
        # Initialize database
        from sloughgpt.core.database import DatabaseManager
        
        db_manager = DatabaseManager(config.database_config)
        await db_manager.initialize()
        
        # Test database operations
        async with db_manager.get_session() as session:
            # Create learning experience
            experience = await db_manager.create_learning_experience(
                session=session,
                prompt="Test prompt",
                response="Test response",
                user_id="test_user",
                session_id="test_session",
                metadata={"test": True}
            )
            
            assert experience.id is not None
            assert experience.prompt == "Test prompt"
        
        await db_manager.close()
        print("✅ Integration workflow completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration workflow test failed: {e}")
        return False

def test_performance():
    """Test performance characteristics"""
    print("\n⚡ Testing performance...")
    
    try:
        # Test import time
        start_time = time.time()
        
        import sloughgpt
        import sloughgpt.config
        import sloughgpt.neural_network
        import sloughgpt.auth
        import sloughgpt.user_management
        import sloughgpt.cost_optimization
        import sloughgpt.data_learning
        import sloughgpt.reasoning_engine
        import sloughgpt.api_server
        import sloughgpt.admin
        
        import_time = time.time() - start_time
        assert import_time < 5.0, f"Import took too long: {import_time:.2f}s"
        print(f"✅ All modules imported in {import_time:.2f}s")
        
        # Test memory usage (basic check)
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        assert memory_mb < 500, f"Memory usage too high: {memory_mb:.1f}MB"
        print(f"✅ Memory usage: {memory_mb:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

async def run_all_tests():
    """Run all tests and return results"""
    print("🚀 Starting SloughGPT Test Suite")
    print("=" * 50)
    
    test_results = {
        "imports": test_imports(),
        "basic_functionality": await test_basic_functionality(),
        "api_server": await test_api_server(),
        "admin_dashboard": await test_admin_dashboard(),
        "integration_workflow": await test_integration_workflow(),
        "performance": test_performance()
    }
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SloughGPT is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_all_tests())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)