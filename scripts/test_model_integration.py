#!/usr/bin/env python3
"""
Demo script to test model integration functionality.
Run this to verify Day 2 implementation is working correctly.
"""
import asyncio
import sys
import time
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import setup_logging, get_logger, set_correlation_id
from app.models.ollama_client import OllamaClient, OllamaException
from app.models.manager import ModelManager, TaskType, QualityLevel


async def test_ollama_client():
    """Test OllamaClient functionality."""
    print("🔧 Testing OllamaClient...")
    
    client = OllamaClient()
    
    try:
        await client.initialize()
        print("  ✓ Client initialized")
        
        # Test health check
        is_healthy = await client.health_check()
        if is_healthy:
            print("  ✓ Ollama service is healthy")
            
            # Test model listing
            models = await client.list_models()
            print(f"  ✓ Found {len(models)} available models:")
            for model in models[:3]:  # Show first 3
                name = model.get('name', 'unknown')
                size = model.get('size', 'unknown size')
                print(f"    - {name} ({size})")
            
            if len(models) > 3:
                print(f"    ... and {len(models) - 3} more")
            
            # Test simple generation if models available
            if models:
                test_model = models[0]['name']
                print(f"  🧪 Testing generation with {test_model}...")
                
                result = await client.generate(
                    model_name=test_model,
                    prompt="Hello, world!",
                    max_tokens=10
                )
                
                if result.success:
                    print(f"  ✓ Generation successful:")
                    print(f"    Response: {result.text[:50]}...")
                    print(f"    Time: {result.execution_time:.2f}s")
                    print(f"    Tokens/sec: {result.tokens_per_second:.1f}")
                else:
                    print(f"  ❌ Generation failed: {result.error}")
            
        else:
            print("  ⚠️  Ollama service not available")
            print("     Make sure Ollama is running: docker-compose up ollama")
            return False
    
    except Exception as e:
        print(f"  ❌ OllamaClient test failed: {e}")
        return False
    
    finally:
        await client.close()
    
    return True


async def test_model_manager():
    """Test ModelManager functionality."""
    print("\n🧠 Testing ModelManager...")
    
    manager = ModelManager()
    
    try:
        await manager.initialize()
        print("  ✓ ModelManager initialized")
        
        # Test model selection
        print("  🎯 Testing model selection:")
        
        test_cases = [
            (TaskType.SIMPLE_CLASSIFICATION, QualityLevel.MINIMAL),
            (TaskType.QA_AND_SUMMARY, QualityLevel.BALANCED),
            (TaskType.ANALYTICAL_REASONING, QualityLevel.HIGH),
            (TaskType.CONVERSATION, QualityLevel.BALANCED)
        ]
        
        for task_type, quality_level in test_cases:
            selected_model = manager.select_optimal_model(task_type, quality_level)
            print(f"    {task_type.value} ({quality_level.value}): {selected_model}")
        
        # Test text generation
        print("  🧪 Testing text generation...")
        
        conversation_model = manager.select_optimal_model(TaskType.CONVERSATION)
        result = await manager.generate(
            model_name=conversation_model,
            prompt="What is artificial intelligence in one sentence?",
            max_tokens=30,
            temperature=0.7
        )
        
        if result.success:
            print(f"  ✓ Generation successful:")
            print(f"    Model: {result.model_used}")
            print(f"    Response: {result.text}")
            print(f"    Time: {result.execution_time:.2f}s")
            print(f"    Cost: ₹{result.cost:.4f}")
        else:
            print(f"  ❌ Generation failed: {result.error}")
            return False
        
        # Test model statistics
        print("  📊 Model statistics:")
        stats = manager.get_model_stats()
        print(f"    Total models: {stats['total_models']}")
        print(f"    Loaded models: {stats['loaded_models']}")
        print(f"    Total requests: {stats['performance_summary']['total_requests']}")
        
        # Show top performing models
        if stats['model_details']:
            print("    Top models by performance:")
            model_scores = [
                (name, details.get('performance_score', 0))
                for name, details in stats['model_details'].items()
                if details.get('total_requests', 0) > 0
            ]
            model_scores.sort(key=lambda x: x[1], reverse=True)
            
            for name, score in model_scores[:3]:
                print(f"      {name}: {score:.3f}")
        
        # Test recommendations
        recommendations = manager.get_model_recommendations()
        if any(recommendations.values()):
            print("  💡 Optimization recommendations:")
            for category, items in recommendations.items():
                if items:
                    print(f"    {category.replace('_', ' ').title()}:")
                    for item in items[:2]:  # Show first 2
                        print(f"      - {item}")
        
    except Exception as e:
        print(f"  ❌ ModelManager test failed: {e}")
        return False
    
    finally:
        await manager.cleanup()
    
    return True


async def test_performance_scenarios():
    """Test performance and edge case scenarios."""
    print("\n⚡ Testing performance scenarios...")
    
    manager = ModelManager()
    
    try:
        await manager.initialize()
        
        # Test concurrent generations
        print("  🏃 Testing concurrent generations...")
        
        start_time = time.time()
        tasks = []
        
        for i in range(3):
            model = manager.select_optimal_model(TaskType.SIMPLE_CLASSIFICATION)
            task = manager.generate(
                model_name=model,
                prompt=f"Test prompt {i + 1}",
                max_tokens=5
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        successful_results = [r for r in results if hasattr(r, 'success') and r.success]
        print(f"  ✓ Completed {len(successful_results)}/{len(tasks)} generations in {total_time:.2f}s")
        
        # Test fallback behavior
        print("  🔄 Testing fallback behavior...")
        
        result = await manager.generate(
            model_name="nonexistent:model",
            prompt="This should fallback",
            max_tokens=10,
            fallback=True
        )
        
        if result.success:
            print(f"  ✓ Fallback successful to: {result.model_used}")
        else:
            print(f"  ❌ Fallback failed: {result.error}")
        
        # Test model preloading
        print("  📦 Testing model preloading...")
        
        available_models = list(manager.models.keys())[:2]  # Test with first 2 models
        if available_models:
            preload_results = await manager.preload_models(available_models)
            successful_preloads = sum(preload_results.values())
            print(f"  ✓ Preloaded {successful_preloads}/{len(available_models)} models")
    
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        return False
    
    finally:
        await manager.cleanup()
    
    return True


async def run_integration_demo():
    """Run complete integration demo."""
    print("🚀 AI Search System - Model Integration Demo")
    print("=" * 50)
    
    # Setup logging
    setup_logging(log_level="INFO", log_format="text", enable_file_logging=False)
    set_correlation_id("demo-integration-001")
    
    logger = get_logger("demo")
    logger.info("Starting model integration demo")
    
    start_time = time.time()
    
    try:
        # Test individual components
        client_success = await test_ollama_client()
        
        if client_success:
            manager_success = await test_model_manager()
            
            if manager_success:
                performance_success = await test_performance_scenarios()
                
                # Summary
                print("\n" + "=" * 50)
                total_time = time.time() - start_time
                
                if performance_success:
                    print("🎉 ALL TESTS PASSED!")
                    print(f"✓ OllamaClient: Working")
                    print(f"✓ ModelManager: Working") 
                    print(f"✓ Performance: Working")
                    print(f"⏱️  Total time: {total_time:.2f}s")
                    
                    print("\n🎯 Next Steps:")
                    print("1. ✅ Day 1: Dummy code elimination - COMPLETED")
                    print("2. ✅ Day 2: Model integration - COMPLETED")
                    print("3. ⏳ Day 3: Graph system implementation - READY TO START")
                    
                    logger.info("Integration demo completed successfully")
                    return True
                else:
                    print("⚠️  Some performance tests failed")
            else:
                print("❌ ModelManager tests failed")
        else:
            print("❌ OllamaClient tests failed")
            print("\n🔧 Troubleshooting:")
            print("1. Ensure Ollama is running: docker-compose up ollama")
            print("2. Check if models are available: ollama list")
            print("3. Try pulling a basic model: ollama pull phi:mini")
    
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        logger.error("Integration demo failed", exc_info=True)
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Demo completed in {total_time:.2f}s")
    return False


if __name__ == "__main__":
    # Run the demo
    success = asyncio.run(run_integration_demo())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
