"""
Performance Optimization Framework
Main performance optimization coordinator with caching and monitoring integration
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import logging

from .monitoring import PerformanceTracker
from .caching import SmartCache

logger = logging.getLogger(__name__)

def cached(ttl: float = 3600, cache_instance: Optional[SmartCache] = None):
    """Decorator for caching function results"""
    if cache_instance is None:
        cache_instance = SmartCache()
    
    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_instance._generate_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache_instance.set(cache_key, result, ttl)
            return result
        
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_instance._generate_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result, ttl)
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

class PerformanceOptimizer:
    """Main performance optimization coordinator"""
    
    def __init__(self):
        self.tracker = PerformanceTracker()
        self.cache = SmartCache(max_size=50000, default_ttl=1800)  # 30 min default TTL
        self.search_cache = SmartCache(max_size=10000, default_ttl=3600)  # 1 hour for search
        self.response_cache = SmartCache(max_size=5000, default_ttl=900)   # 15 min for responses
        
        # Performance monitoring
        self.monitoring_active = True
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        if self.monitoring_active:
            # Start cache cleanup task
            asyncio.create_task(self._periodic_cache_cleanup())
            asyncio.create_task(self._performance_monitoring())
    
    async def _periodic_cache_cleanup(self):
        """Periodic cache cleanup task"""
        while self.monitoring_active:
            try:
                # Clean expired entries every 5 minutes
                await asyncio.sleep(300)
                
                expired_main = self.cache.clear_expired()
                expired_search = self.search_cache.clear_expired()
                expired_response = self.response_cache.clear_expired()
                
                logger.info(f"Cache cleanup: {expired_main + expired_search + expired_response} expired entries removed")
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {str(e)}")
    
    async def _performance_monitoring(self):
        """Continuous performance monitoring"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Get performance summary
                summary = self.tracker.get_performance_summary(hours=1)
                
                # Check for performance issues
                if summary.get("target_compliance"):
                    compliance = summary["target_compliance"]
                    
                    if not compliance.get("response_time", True):
                        logger.warning("⚠️ Response time target not met!")
                    
                    if not compliance.get("success_rate", True):
                        logger.warning("⚠️ Success rate target not met!")
                    
                    if not compliance.get("cache_hit_rate", True):
                        logger.warning("⚠️ Cache hit rate target not met!")
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {str(e)}")
    
    @cached(ttl=1800)  # 30 minutes cache
    async def optimized_search(self, search_func: Callable, *args, **kwargs) -> Any:
        """Optimized search with caching and performance tracking"""
        
        # Generate specific cache key for search
        cache_key = self.search_cache._generate_key("search", *args, **kwargs)
        
        # Check search cache first
        cached_result = self.search_cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"🎯 Search cache hit for key: {cache_key[:16]}...")
            return cached_result
        
        # Track search performance
        async with self.tracker.track_operation("search_execution") as request_id:
            try:
                # Execute search with timeout
                result = await asyncio.wait_for(
                    search_func(*args, **kwargs),
                    timeout=8.0  # 8 second timeout (buffer for 5s target)
                )
                
                # Cache successful results
                self.search_cache.set(cache_key, result, ttl=3600)
                
                self.tracker.finish_operation(
                    request_id,
                    success=True,
                    cache_hit=False,
                    cost=getattr(result, 'total_cost', 0.0)
                )
                
                return result
                
            except asyncio.TimeoutError:
                logger.error("⏰ Search timeout exceeded 8 seconds")
                raise
            except Exception as e:
                logger.error(f"🔍 Search execution failed: {str(e)}")
                raise
    
    async def optimized_response_generation(self, response_func: Callable, *args, **kwargs) -> Any:
        """Optimized response generation with caching"""
        
        cache_key = self.response_cache._generate_key("response", *args, **kwargs)
        
        # Check response cache
        cached_response = self.response_cache.get(cache_key)
        if cached_response is not None:
            logger.info(f"💬 Response cache hit for key: {cache_key[:16]}...")
            return cached_response
        
        async with self.tracker.track_operation("response_generation") as request_id:
            try:
                result = await response_func(*args, **kwargs)
                
                # Cache response for 15 minutes
                self.response_cache.set(cache_key, result, ttl=900)
                
                self.tracker.finish_operation(
                    request_id,
                    success=True,
                    cache_hit=False
                )
                
                return result
                
            except Exception as e:
                logger.error(f"💬 Response generation failed: {str(e)}")
                raise
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        performance_summary = self.tracker.get_performance_summary(hours=24)
        cache_stats = {
            "main_cache": self.cache.get_stats(),
            "search_cache": self.search_cache.get_stats(),
            "response_cache": self.response_cache.get_stats()
        }
        
        # Calculate overall cache efficiency
        total_hits = sum(stats["hits"] for stats in cache_stats.values())
        total_requests = sum(stats["hits"] + stats["misses"] for stats in cache_stats.values())
        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        return {
            "performance_summary": performance_summary,
            "cache_statistics": cache_stats,
            "overall_cache_hit_rate": overall_hit_rate,
            "system_health": self._assess_system_health(performance_summary),
            "optimization_recommendations": self._get_optimization_recommendations(performance_summary, cache_stats)
        }
    
    def _assess_system_health(self, summary: Dict[str, Any]) -> Dict[str, str]:
        """Assess overall system health"""
        health = {}
        
        if not summary.get("target_compliance"):
            return {"overall": "unknown", "details": "Insufficient data"}
        
        compliance = summary["target_compliance"]
        
        # Response time health
        if compliance.get("response_time", False):
            health["response_time"] = "healthy"
        else:
            avg_time = summary.get("response_times", {}).get("avg", 0)
            if avg_time > 10:
                health["response_time"] = "critical"
            elif avg_time > 7:
                health["response_time"] = "warning"
            else:
                health["response_time"] = "degraded"
        
        # Success rate health
        success_rate = summary.get("success_rate", 0)
        if success_rate >= 0.98:
            health["reliability"] = "excellent"
        elif success_rate >= 0.95:
            health["reliability"] = "healthy"
        elif success_rate >= 0.90:
            health["reliability"] = "warning"
        else:
            health["reliability"] = "critical"
        
        # Cache health
        cache_rate = summary.get("cache_hit_rate", 0)
        if cache_rate >= 0.85:
            health["caching"] = "excellent"
        elif cache_rate >= 0.70:
            health["caching"] = "healthy"
        elif cache_rate >= 0.50:
            health["caching"] = "warning"
        else:
            health["caching"] = "poor"
        
        # Cost efficiency
        avg_cost = summary.get("avg_cost_per_query", 0)
        if avg_cost <= 0.05:
            health["cost_efficiency"] = "excellent"
        elif avg_cost <= 0.10:
            health["cost_efficiency"] = "healthy"
        elif avg_cost <= 0.20:
            health["cost_efficiency"] = "warning"
        else:
            health["cost_efficiency"] = "expensive"
        
        # Overall health
        health_scores = {
            "excellent": 4, "healthy": 3, "warning": 2, 
            "degraded": 1, "poor": 1, "critical": 0, "expensive": 1
        }
        
        avg_score = sum(health_scores.get(status, 0) for status in health.values()) / len(health)
        
        if avg_score >= 3.5:
            health["overall"] = "excellent"
        elif avg_score >= 2.5:
            health["overall"] = "healthy"
        elif avg_score >= 1.5:
            health["overall"] = "warning"
        else:
            health["overall"] = "critical"
        
        return health
    
    def _get_optimization_recommendations(self, performance: Dict[str, Any], cache_stats: Dict[str, Any]) -> List[str]:
        """Get optimization recommendations based on metrics"""
        recommendations = []
        
        # Response time recommendations
        avg_response_time = performance.get("response_times", {}).get("avg", 0)
        if avg_response_time > 5:
            recommendations.append("🚀 Consider increasing cache TTL or implementing predictive caching")
            recommendations.append("⚡ Review search provider routing - consider using faster providers")
        
        if avg_response_time > 3:
            recommendations.append("🔄 Implement parallel search execution for multiple providers")
        
        # Cache recommendations
        overall_hit_rate = sum(stats["hits"] for stats in cache_stats.values()) / \
                          sum(stats["hits"] + stats["misses"] for stats in cache_stats.values()) \
                          if any(stats["hits"] + stats["misses"] > 0 for stats in cache_stats.values()) else 0
        
        if overall_hit_rate < 0.70:
            recommendations.append("💾 Increase cache size or adjust TTL settings")
            recommendations.append("🎯 Implement smarter cache warming strategies")
        
        # Cost recommendations
        avg_cost = performance.get("avg_cost_per_query", 0)
        if avg_cost > 0.15:
            recommendations.append("💰 Optimize provider routing to use more free/cheaper providers")
            recommendations.append("🎛️ Implement more aggressive local processing")
        
        # Success rate recommendations
        success_rate = performance.get("success_rate", 0)
        if success_rate < 0.95:
            recommendations.append("🛡️ Implement better fallback mechanisms")
            recommendations.append("🔧 Add circuit breakers for external API calls")
        
        # Memory recommendations
        total_cache_size = sum(stats.get("total_size_bytes", 0) for stats in cache_stats.values())
        if total_cache_size > 100_000_000:  # 100MB
            recommendations.append("🗜️ Consider implementing cache compression")
            recommendations.append("🧹 Implement more aggressive cache eviction policies")
        
        return recommendations

class OptimizedSearchSystem:
    """Complete optimized search system"""
    
    def __init__(self, search_router, search_graph):
        self.search_router = search_router
        self.search_graph = search_graph
        self.optimizer = PerformanceOptimizer()
        
        # Wrap search functions with optimization
        self._wrap_search_functions()
    
    def _wrap_search_functions(self):
        """Wrap search functions with performance optimization"""
        
        # Store original methods
        original_search = self.search_router.search
        original_execute = self.search_graph.execute
        
        # Wrap router search
        async def optimized_router_search(*args, **kwargs):
            return await self.optimizer.optimized_search(original_search, *args, **kwargs)
        
        # Wrap graph execution
        async def optimized_graph_execute(*args, **kwargs):
            return await self.optimizer.optimized_response_generation(original_execute, *args, **kwargs)
        
        # Replace methods
        self.search_router.search = optimized_router_search
        self.search_graph.execute = optimized_graph_execute
    
    async def execute_optimized_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute optimized search with full performance tracking"""
        
        async with self.optimizer.tracker.track_operation("complete_search_flow") as request_id:
            try:
                # Execute optimized search
                result = await self.search_graph.execute(query, **kwargs)
                
                # Add performance metrics to result
                result["performance_metrics"] = self.optimizer.get_comprehensive_metrics()
                
                self.optimizer.tracker.finish_operation(
                    request_id,
                    success=True,
                    cost=result.get("metadata", {}).get("total_cost", 0),
                    metadata={
                        "query_length": len(query),
                        "results_count": len(result.get("citations", [])),
                        "execution_path": result.get("metadata", {}).get("execution_path", [])
                    }
                )
                
                return result
                
            except Exception as e:
                self.optimizer.tracker.finish_operation(
                    request_id,
                    success=False,
                    error=str(e)
                )
                raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        metrics = self.optimizer.get_comprehensive_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": metrics["system_health"],
            "performance_summary": metrics["performance_summary"],
            "cache_efficiency": {
                "overall_hit_rate": metrics["overall_cache_hit_rate"],
                "cache_breakdown": metrics["cache_statistics"]
            },
            "recommendations": metrics["optimization_recommendations"],
            "targets": {
                "response_time": "< 5 seconds",
                "cache_hit_rate": "> 80%",
                "success_rate": "> 95%",
                "cost_per_query": "< ₹0.105"
            }
        }