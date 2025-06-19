# app/optimization/performance_tuner.py
"""
Performance Tuning System with Advanced Cache Strategies
Optimizes response times through intelligent caching and pre-computation
"""

import asyncio
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import json

import structlog
import numpy as np
from collections import defaultdict, deque

from app.cache.redis_client import CacheManager
from app.models.manager import ModelManager
from app.core.config import get_settings

logger = structlog.get_logger(__name__)


class CacheStrategy(Enum):
    """Cache strategy types"""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL_BASED = "ttl_based"        # Time To Live based
    ADAPTIVE = "adaptive"          # Adaptive based on usage patterns
    PREDICTIVE = "predictive"      # Predictive pre-caching


@dataclass
class CacheMetrics:
    """Detailed cache performance metrics"""
    key: str
    hit_count: int = 0
    miss_count: int = 0
    last_access: Optional[datetime] = None
    creation_time: Optional[datetime] = None
    access_frequency: float = 0.0
    avg_response_time: float = 0.0
    cache_value_size: int = 0
    ttl: Optional[int] = None
    strategy: CacheStrategy = CacheStrategy.TTL_BASED
    
    def update_hit(self, response_time: float):
        """Update metrics on cache hit"""
        self.hit_count += 1
        self.last_access = datetime.now()
        self._update_frequency()
        self._update_avg_response_time(response_time)
    
    def update_miss(self, response_time: float):
        """Update metrics on cache miss"""
        self.miss_count += 1
        self.last_access = datetime.now()
        self._update_avg_response_time(response_time)
    
    def _update_frequency(self):
        """Update access frequency"""
        if self.creation_time:
            time_since_creation = (datetime.now() - self.creation_time).total_seconds() / 3600
            self.access_frequency = self.hit_count / max(time_since_creation, 0.1)
    
    def _update_avg_response_time(self, response_time: float):
        """Update average response time"""
        total_accesses = self.hit_count + self.miss_count
        if total_accesses == 1:
            self.avg_response_time = response_time
        else:
            alpha = 1.0 / total_accesses
            self.avg_response_time = (1 - alpha) * self.avg_response_time + alpha * response_time
    
    def get_hit_rate(self) -> float:
        """Calculate hit rate"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def get_cache_efficiency_score(self) -> float:
        """Calculate overall cache efficiency score"""
        hit_rate = self.get_hit_rate()
        frequency_score = min(self.access_frequency / 10.0, 1.0)  # Normalize to max 10 accesses/hour
        recency_score = 1.0
        
        if self.last_access:
            hours_since_access = (datetime.now() - self.last_access).total_seconds() / 3600
            recency_score = max(0.1, 1.0 - (hours_since_access / 24.0))  # Decay over 24 hours
        
        return (hit_rate * 0.5 + frequency_score * 0.3 + recency_score * 0.2)


@dataclass
class QueryPattern:
    """Query pattern for predictive caching"""
    pattern_hash: str
    query_keywords: List[str]
    typical_response_time: float
    frequency: int
    last_seen: datetime
    next_predicted_access: Optional[datetime] = None
    confidence: float = 0.0
    
    def should_precompute(self) -> bool:
        """Determine if this pattern should be pre-computed"""
        if not self.next_predicted_access:
            return False
        
        time_until_access = (self.next_predicted_access - datetime.now()).total_seconds()
        return (0 < time_until_access < 3600 and  # Within next hour
                self.confidence > 0.7 and          # High confidence
                self.frequency > 5)                # Sufficient frequency


class AdvancedCacheManager:
    """Advanced cache manager with multiple strategies"""
    
    def __init__(self, base_cache_manager: CacheManager):
        self.base_cache = base_cache_manager
        self.metrics: Dict[str, CacheMetrics] = {}
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.precompute_queue: deque = deque()
        self.cache_strategies: Dict[str, CacheStrategy] = {}
        self.performance_targets = {
            "hit_rate": 0.85,
            "avg_response_time": 0.5,
            "cache_efficiency": 0.8
        }
        
        # Cache warming and prediction
        self.warming_enabled = True
        self.prediction_model = SimplePredictionModel()
        
        # Background tasks
        self._optimization_task = None
        self._warming_task = None
    
    async def initialize(self):
        """Initialize advanced cache manager"""
        await self._load_metrics()
        await self._load_query_patterns()
        
        # Start background optimization tasks
        self._optimization_task = asyncio.create_task(self._background_optimization())
        self._warming_task = asyncio.create_task(self._background_cache_warming())
        
        logger.info("Advanced cache manager initialized")
    
    async def get_with_strategy(
        self, 
        key: str, 
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        default: Any = None
    ) -> Tuple[Any, bool]:
        """Get value from cache with specific strategy"""
        start_time = time.time()
        
        # Try to get from cache
        value = await self.base_cache.get(key, default)
        is_hit = value is not default
        
        response_time = time.time() - start_time
        
        # Update metrics
        if key not in self.metrics:
            self.metrics[key] = CacheMetrics(
                key=key,
                creation_time=datetime.now(),
                strategy=strategy
            )
        
        if is_hit:
            self.metrics[key].update_hit(response_time)
        else:
            self.metrics[key].update_miss(response_time)
        
        # Adaptive strategy adjustments
        if strategy == CacheStrategy.ADAPTIVE:
            await self._adjust_cache_strategy(key)
        
        return value, is_hit
    
    async def set_with_strategy(
        self,
        key: str,
        value: Any,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache with specific strategy"""
        
        # Determine optimal TTL based on strategy
        optimal_ttl = await self._calculate_optimal_ttl(key, strategy, ttl)
        
        # Set in base cache
        success = await self.base_cache.set(key, value, optimal_ttl)
        
        if success:
            # Update metrics
            if key not in self.metrics:
                self.metrics[key] = CacheMetrics(
                    key=key,
                    creation_time=datetime.now(),
                    strategy=strategy
                )
            
            self.metrics[key].cache_value_size = len(json.dumps(value)) if value else 0
            self.metrics[key].ttl = optimal_ttl
            self.cache_strategies[key] = strategy
            
            # Update query patterns for predictive caching
            await self._update_query_patterns(key, value)
        
        return success
    
    async def _calculate_optimal_ttl(
        self, 
        key: str, 
        strategy: CacheStrategy, 
        requested_ttl: Optional[int]
    ) -> Optional[int]:
        """Calculate optimal TTL based on strategy and historical data"""
        
        if requested_ttl:
            return requested_ttl
        
        # Get historical data for this key
        metrics = self.metrics.get(key)
        
        if strategy == CacheStrategy.TTL_BASED:
            return 3600  # Default 1 hour
        
        elif strategy == CacheStrategy.LRU:
            return None  # No TTL for LRU
        
        elif strategy == CacheStrategy.LFU:
            if metrics and metrics.access_frequency > 5:
                return 7200  # 2 hours for frequently accessed
            return 1800  # 30 minutes for less frequent
        
        elif strategy == CacheStrategy.ADAPTIVE:
            if metrics:
                # Base TTL on access frequency and hit rate
                hit_rate = metrics.get_hit_rate()
                frequency = metrics.access_frequency
                
                if hit_rate > 0.8 and frequency > 2:
                    return 7200  # High value items get longer TTL
                elif hit_rate > 0.5:
                    return 3600  # Medium value items
                else:
                    return 900   # Low value items get short TTL
            return 1800  # Default for new items
        
        elif strategy == CacheStrategy.PREDICTIVE:
            # Longer TTL for predicted items
            pattern = self._find_matching_pattern(key)
            if pattern and pattern.confidence > 0.7:
                return 14400  # 4 hours for high-confidence predictions
            return 3600
        
        return 3600  # Default fallback
    
    async def _adjust_cache_strategy(self, key: str):
        """Dynamically adjust cache strategy based on performance"""
        metrics = self.metrics.get(key)
        if not metrics or metrics.hit_count + metrics.miss_count < 10:
            return  # Need more data
        
        hit_rate = metrics.get_hit_rate()
        frequency = metrics.access_frequency
        
        # Strategy adjustment logic
        current_strategy = self.cache_strategies.get(key, CacheStrategy.TTL_BASED)
        new_strategy = current_strategy
        
        if hit_rate < 0.3:
            # Low hit rate - try shorter TTL or LRU
            new_strategy = CacheStrategy.LRU
        elif frequency > 10 and hit_rate > 0.8:
            # High frequency, high hit rate - use LFU
            new_strategy = CacheStrategy.LFU
        elif self._is_predictable_pattern(key):
            # Predictable access pattern - use predictive caching
            new_strategy = CacheStrategy.PREDICTIVE
        
        if new_strategy != current_strategy:
            self.cache_strategies[key] = new_strategy
            logger.debug(f"Adjusted cache strategy for {key}: {current_strategy.value} -> {new_strategy.value}")
    
    def _is_predictable_pattern(self, key: str) -> bool:
        """Check if key follows a predictable access pattern"""
        pattern = self._find_matching_pattern(key)
        return pattern is not None and pattern.confidence > 0.6
    
    def _find_matching_pattern(self, key: str) -> Optional[QueryPattern]:
        """Find matching query pattern for a cache key"""
        # Simple keyword matching - could be enhanced with ML
        key_lower = key.lower()
        
        for pattern in self.query_patterns.values():
            if any(keyword in key_lower for keyword in pattern.query_keywords):
                return pattern
        
        return None
    
    async def _update_query_patterns(self, key: str, value: Any):
        """Update query patterns for predictive caching"""
        # Extract keywords from key (simplified)
        keywords = self._extract_keywords_from_key(key)
        pattern_hash = hashlib.md5(''.join(sorted(keywords)).encode()).hexdigest()[:16]
        
        if pattern_hash in self.query_patterns:
            pattern = self.query_patterns[pattern_hash]
            pattern.frequency += 1
            pattern.last_seen = datetime.now()
            
            # Update prediction
            pattern.next_predicted_access = self.prediction_model.predict_next_access(pattern)
            pattern.confidence = self.prediction_model.calculate_confidence(pattern)
        else:
            # Create new pattern
            self.query_patterns[pattern_hash] = QueryPattern(
                pattern_hash=pattern_hash,
                query_keywords=keywords,
                typical_response_time=1.0,  # Will be updated
                frequency=1,
                last_seen=datetime.now(),
                confidence=0.1
            )
    
    def _extract_keywords_from_key(self, key: str) -> List[str]:
        """Extract keywords from cache key"""
        # Simple keyword extraction
        parts = key.lower().replace(':', ' ').replace('_', ' ').split()
        keywords = [part for part in parts if len(part) > 2]
        return keywords[:5]  # Limit to 5 keywords
