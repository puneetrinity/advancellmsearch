"""
Performance Monitoring
Advanced performance tracking and monitoring for search operations
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import logging
from contextlib import asynccontextmanager
import threading

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    operation: str = ""
    success: bool = True
    error: Optional[str] = None
    cache_hit: bool = False
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self, success: bool = True, error: Optional[str] = None):
        """Mark operation as finished"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error = error

class PerformanceTracker:
    """Advanced performance tracking and monitoring"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.active_requests: Dict[str, PerformanceMetrics] = {}
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # Performance targets
        self.targets = {
            "search_response_time": 5.0,  # 5 seconds max
            "cache_hit_rate": 0.8,        # 80% cache hit rate
            "success_rate": 0.95,         # 95% success rate
            "cost_per_query": 0.105       # ₹0.105 average cost
        }
        
    def start_operation(self, operation: str, request_id: Optional[str] = None) -> str:
        """Start tracking an operation"""
        if not request_id:
            request_id = f"{operation}_{int(time.time()*1000000)}"
        
        metrics = PerformanceMetrics(
            request_id=request_id,
            start_time=time.time(),
            operation=operation
        )
        
        with self._lock:
            self.active_requests[request_id] = metrics
        
        return request_id
    
    def finish_operation(self, request_id: str, success: bool = True, 
                        error: Optional[str] = None, cost: float = 0.0,
                        cache_hit: bool = False, metadata: Optional[Dict] = None):
        """Finish tracking an operation"""
        with self._lock:
            if request_id not in self.active_requests:
                return
            
            metrics = self.active_requests.pop(request_id)
            metrics.finish(success, error)
            metrics.cost = cost
            metrics.cache_hit = cache_hit
            if metadata:
                metrics.metadata.update(metadata)
            
            # Store in history
            self.metrics_history.append(metrics)
            
            # Update operation stats
            if success and metrics.duration:
                self.operation_stats[metrics.operation].append(metrics.duration)
                
                # Keep only recent stats
                if len(self.operation_stats[metrics.operation]) > 1000:
                    self.operation_stats[metrics.operation] = \
                        self.operation_stats[metrics.operation][-500:]
    
    @asynccontextmanager
    async def track_operation(self, operation: str, **kwargs):
        """Context manager for tracking operations"""
        request_id = self.start_operation(operation)
        try:
            yield request_id
            self.finish_operation(request_id, success=True, **kwargs)
        except Exception as e:
            self.finish_operation(request_id, success=False, error=str(e), **kwargs)
            raise
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.metrics_history if m.start_time > cutoff_time]
        
        if not recent_metrics:
            return {"error": "No recent metrics available"}
        
        total_requests = len(recent_metrics)
        successful_requests = sum(1 for m in recent_metrics if m.success)
        cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
        total_cost = sum(m.cost for m in recent_metrics)
        
        # Calculate percentiles for response times
        response_times = [m.duration for m in recent_metrics if m.duration]
        response_times.sort()
        
        def percentile(data, p):
            if not data:
                return 0
            index = int(len(data) * p / 100)
            return data[min(index, len(data) - 1)]
        
        summary = {
            "time_period_hours": hours,
            "total_requests": total_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
            "cache_hit_rate": cache_hits / total_requests if total_requests > 0 else 0,
            "avg_cost_per_query": total_cost / total_requests if total_requests > 0 else 0,
            "response_times": {
                "avg": sum(response_times) / len(response_times) if response_times else 0,
                "p50": percentile(response_times, 50),
                "p90": percentile(response_times, 90),
                "p95": percentile(response_times, 95),
                "p99": percentile(response_times, 99)
            },
            "target_compliance": self._check_target_compliance(recent_metrics),
            "operation_breakdown": self._get_operation_breakdown(recent_metrics)
        }
        
        return summary
    
    def _check_target_compliance(self, metrics: List[PerformanceMetrics]) -> Dict[str, bool]:
        """Check compliance with performance targets"""
        if not metrics:
            return {}
        
        successful = sum(1 for m in metrics if m.success)
        cache_hits = sum(1 for m in metrics if m.cache_hit)
        avg_response_time = sum(m.duration for m in metrics if m.duration) / len(metrics)
        avg_cost = sum(m.cost for m in metrics) / len(metrics)
        
        return {
            "response_time": avg_response_time <= self.targets["search_response_time"],
            "cache_hit_rate": (cache_hits / len(metrics)) >= self.targets["cache_hit_rate"],
            "success_rate": (successful / len(metrics)) >= self.targets["success_rate"],
            "cost_per_query": avg_cost <= self.targets["cost_per_query"]
        }
    
    def _get_operation_breakdown(self, metrics: List[PerformanceMetrics]) -> Dict[str, Dict[str, Any]]:
        """Get breakdown by operation type"""
        operations = defaultdict(list)
        for metric in metrics:
            operations[metric.operation].append(metric)
        
        breakdown = {}
        for op, op_metrics in operations.items():
            durations = [m.duration for m in op_metrics if m.duration]
            breakdown[op] = {
                "count": len(op_metrics),
                "success_rate": sum(1 for m in op_metrics if m.success) / len(op_metrics),
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "total_cost": sum(m.cost for m in op_metrics)
            }
        
        return breakdown