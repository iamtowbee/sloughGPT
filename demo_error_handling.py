#!/usr/bin/env python3
"""
SloughGPT Error Handling & Logging Demo
Demonstrates comprehensive error handling, structured logging, and resilience patterns
"""

import sys
import os
import time
import random
import asyncio

# Add sloughgpt to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sloughgpt.core.logging_system import (
    get_logger, setup_logging, timer, LogLevel, LogFormat,
    get_error_tracker
)
from sloughgpt.core.error_handling import (
    ErrorHandler, circuit_breaker, retry, 
    CircuitBreakerConfig, RetryConfig, DEFAULT_RETRY, DATABASE_RETRY,
    get_error_handler, DEFAULT_CIRCUIT_BREAKER
)
from sloughgpt.core.exceptions import (
    DatabaseError, ModelError, APIError, 
    SloughGPTErrorCode, create_error
)

def demo_structured_logging():
    """Demonstrate structured logging capabilities"""
    print("📝 SloughGPT Structured Logging Demo")
    print("=" * 50)
    
    # Setup logging
    log_config = {
        "level": LogLevel.INFO,
        "format": LogFormat.STRUCTURED,
        "enable_console": True,
        "enable_performance": True,
        "file_path": "logs/demo_structured.log"
    }
    
    logger = setup_logging(log_config)
    
    print("\n1. Basic Logging Levels...")
    logger.debug("This is a debug message", debug_info=True)
    logger.info("This is an info message", info_type="basic")
    logger.warning("This is a warning message", warning_code=1001)
    logger.error("This is an error message", error_type="demo_error")
    logger.critical("This is a critical message", system_status="failing")
    
    print("\n2. Structured Data Logging...")
    logger.info("User action performed", 
                user_id="user_123",
                action="login",
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0...",
                session_id="sess_abc123",
                timestamp="2026-01-31T12:30:00Z")
    
    print("\n3. Performance Timing...")
    with logger.timer("database_query"):
        time.sleep(0.1)  # Simulate work
        logger.info("Query completed", rows_affected=42)
    
    with logger.timer("model_inference"):
        time.sleep(0.05)  # Simulate work
        logger.info("Inference completed", 
                   model="sloughgpt_small",
                   tokens_generated=150,
                   confidence=0.87)
    
    print("\n4. Performance Metrics...")
    logger.performance("api_request", 125.5, 
                    endpoint="/api/chat",
                    method="POST",
                    status_code=200,
                    user_id="user_123")
    
    logger.performance("model_inference", 75.2,
                    model_name="sloughgpt_small",
                    input_tokens=50,
                    output_tokens=100,
                    tokens_per_second=1.33)
    
    print("\n5. Exception Logging...")
    try:
        raise create_error(
            DatabaseError,
            "Connection to database failed",
            SloughGPTErrorCode.DATABASE_CONNECTION_FAILED,
            context={"database_url": "postgresql://localhost/sloughgpt"}
        )
    except Exception as e:
        logger.exception("Database error occurred", exception=e, 
                       operation="connect", retry_count=2)
    
    return logger

def demo_circuit_breaker():
    """Demonstrate circuit breaker functionality"""
    print("\n🔌 SloughGPT Circuit Breaker Demo")
    print("=" * 50)
    
    logger = get_logger("circuit_demo")
    error_handler = get_error_handler()
    
    # Create a failing function
    call_count = 0
    
    @error_handler.circuit_breaker(DEFAULT_CIRCUIT_BREAKER, "demo_service")
    def failing_service():
        nonlocal call_count
        call_count += 1
        
        logger.info(f"Service call attempt {call_count}")
        
        if call_count <= 7:  # Fail first 7 calls
            raise create_error(
                APIError,
                f"Service call {call_count} failed",
                SloughGPTErrorCode.INVALID_REQUEST_FORMAT
            )
        else:
            return f"Success on call {call_count}"
    
    print("Testing circuit breaker with failing service...")
    
    # Test the circuit breaker
    for i in range(12):
        try:
            result = failing_service()
            print(f"   ✅ Call {i+1}: {result}")
        except Exception as e:
            print(f"   ❌ Call {i+1}: {e.message}")
        
        time.sleep(0.1)  # Small delay between calls
    
    # Show circuit breaker status
    cb_keys = [key for key in error_handler.circuit_breakers.keys() if key.startswith("demo_service")]
    if cb_keys:
        status = error_handler.circuit_breakers[cb_keys[0]].get_status()
        print(f"\nCircuit Breaker Status: {status['state']}")
        print(f"Failure Count: {status['failure_count']}")
        print(f"Success Rate: {status['metrics']['success_rate']:.2%}")
    else:
        print("\nNo circuit breakers found")

def demo_retry_handler():
    """Demonstrate retry functionality"""
    print("\n🔄 SloughGPT Retry Handler Demo")
    print("=" * 50)
    
    logger = get_logger("retry_demo")
    error_handler = get_error_handler()
    
    # Create a function that fails initially then succeeds
    attempt_count = 0
    
    @error_handler.retry(DATABASE_RETRY, "database_operation")
    def flaky_database_operation():
        nonlocal attempt_count
        attempt_count += 1
        
        logger.info(f"Database operation attempt {attempt_count}")
        
        if attempt_count < 3:  # Fail first 2 attempts
            raise create_error(
                DatabaseError,
                f"Temporary connection issue (attempt {attempt_count})",
                SloughGPTErrorCode.DATABASE_TIMEOUT
            )
        
        return f"Database operation succeeded on attempt {attempt_count}"
    
    print("Testing retry handler with flaky database...")
    
    try:
        result = flaky_database_operation()
        print(f"   ✅ Final result: {result}")
    except Exception as e:
        print(f"   ❌ Final failure: {e.message}")
    
    print(f"\nTotal attempts made: {attempt_count}")

async def demo_async_error_handling():
    """Demonstrate async error handling"""
    print("\n⚡ SloughGPT Async Error Handling Demo")
    print("=" * 50)
    
    logger = get_logger("async_demo")
    error_handler = get_error_handler()
    
    # Create async function with circuit breaker and retry
    call_count = 0
    
    @error_handler.circuit_breaker(DEFAULT_CIRCUIT_BREAKER, "async_service")
    @error_handler.retry(DEFAULT_RETRY, "async_operation")
    async def async_service():
        nonlocal call_count
        call_count += 1
        
        logger.info(f"Async service call {call_count}")
        
        # Simulate async work
        await asyncio.sleep(0.05)
        
        if call_count <= 2:  # Fail first 2 calls
            raise create_error(
                ModelError,
                f"Model inference failed (attempt {call_count})",
                SloughGPTErrorCode.MODEL_INFERENCE_FAILED
            )
        
        return f"Async service succeeded on attempt {call_count}"
    
    print("Testing async error handling...")
    
    for i in range(5):
        try:
            result = await async_service()
            print(f"   ✅ Async call {i+1}: {result}")
        except Exception as e:
            print(f"   ❌ Async call {i+1}: {e.message}")
        
        await asyncio.sleep(0.1)

def demo_error_tracking():
    """Demonstrate error tracking and analytics"""
    print("\n📊 SloughGPT Error Tracking Demo")
    print("=" * 50)
    
    logger = get_logger("tracking_demo")
    error_tracker = get_error_tracker()
    
    # Simulate various types of errors
    errors = [
        create_error(DatabaseError, "Connection lost", SloughGPTErrorCode.DATABASE_CONNECTION_FAILED),
        create_error(ModelError, "Model not loaded", SloughGPTErrorCode.MODEL_NOT_INITIALIZED),
        create_error(APIError, "Invalid request", SloughGPTErrorCode.INVALID_REQUEST_FORMAT),
        create_error(DatabaseError, "Query timeout", SloughGPTErrorCode.DATABASE_TIMEOUT),
        create_error(ModelError, "Inference failed", SloughGPTErrorCode.MODEL_INFERENCE_FAILED),
    ]
    
    print("Simulating various errors...")
    
    for i, error in enumerate(errors):
        context = {
            "request_id": f"req_{i+1:03d}",
            "user_id": f"user_{random.randint(1, 100):03d}",
            "component": random.choice(["api", "model", "database", "training"])
        }
        
        error_tracker.track_error(error, context)
    
    # Get error statistics
    stats = error_tracker.get_error_statistics()
    
    print("\nError Tracking Statistics:")
    print(f"   Total Errors: {stats['total_errors']}")
    print(f"   Error Types: {stats['error_types']}")
    print(f"   Critical Errors: {stats['critical_errors']}")
    print(f"   Recent Errors (1hr): {len(stats['recent_errors'])}")
    
    print("\nMost Common Errors:")
    for error_type, count in stats['most_common_errors']:
        print(f"   • {error_type}: {count}")

def demo_comprehensive_status():
    """Show comprehensive error handling status"""
    print("\n📈 SloughGPT Error Handler Status")
    print("=" * 50)
    
    error_handler = get_error_handler()
    status = error_handler.get_status()
    
    print("Error Handler Configuration:")
    print(f"   Name: {status['error_handler']['name']}")
    print(f"   Registered Handlers: {status['error_handler']['registered_handlers']}")
    
    print("\nCircuit Breakers:")
    for name, cb_status in status['circuit_breakers'].items():
        print(f"   • {name}:")
        print(f"     - State: {cb_status['state']}")
        print(f"     - Success Rate: {cb_status['metrics']['success_rate']:.2%}")
        print(f"     - Failure Threshold: {cb_status['config']['failure_threshold']}")
    
    print("\nRetry Handlers:")
    for name, retry_status in status['retry_handlers'].items():
        print(f"   • {name}:")
        print(f"     - Max Attempts: {retry_status['max_attempts']}")
        print(f"     - Base Delay: {retry_status['base_delay']}s")
        print(f"     - Retry On: {', '.join(retry_status['retry_on'])}")

async def main():
    """Main demo function"""
    print("🚀 SloughGPT Error Handling & Logging Comprehensive Demo")
    print("=" * 60)
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Demo 1: Structured Logging
    logger = demo_structured_logging()
    
    # Demo 2: Circuit Breaker
    demo_circuit_breaker()
    
    # Demo 3: Retry Handler
    demo_retry_handler()
    
    # Demo 4: Async Error Handling
    await demo_async_error_handling()
    
    # Demo 5: Error Tracking
    demo_error_tracking()
    
    # Demo 6: Comprehensive Status
    demo_comprehensive_status()
    
    print("\n" + "=" * 60)
    print("🎉 Error Handling & Logging Demo Completed Successfully!")
    print("\n📁 Log files created:")
    print("   • logs/demo_structured.log - Main application logs")
    print("   • logs/performance.log - Performance metrics")
    print("\n🔍 Check the log files to see structured logging in action!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()