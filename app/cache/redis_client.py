# app/cache/redis_client.py
"""
Redis Cache Manager - Hot layer for speed-optimized caching
Handles routing shortcuts, conversation history, and performance hints
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict

import redis.asyncio as redis
import structlog
from pydantic import BaseModel

from app.core.config import get_settings

logger = structlog.get_logger(__name__)


class CacheKey:
    """Cache key constants and generators"""
    
    # Routing shortcuts
    ROUTE_PREFIX = "route:"
    PATTERN_PREFIX = "pattern:"
    SHORTCUT_PREFIX = "shortcut:"
    
    # Performance hints
    EXPECT_PREFIX = "expect:"
    MODEL_PREFIX = "model:"
    
    # Real-time counters
    BUDGET_PREFIX = "budget:"
    RATE_PREFIX = "rate:"
    
    # Session context
    CONTEXT_PREFIX = "context:"
    PREFS_PREFIX = "prefs:"
    CONVERSATION_PREFIX = "conv:"
    
    # Analytics
    METRICS_PREFIX = "metrics:"
    STATS_PREFIX = "stats:"
    
    @staticmethod
    def query_hash(query: str) -> str:
        """Generate consistent hash for query"""
        return hashlib.md5(query.encode()).hexdigest()[:16]
    
    @staticmethod
    def route_key(query: str) -> str:
        """Generate route cache key"""
        return f"{CacheKey.ROUTE_PREFIX}{CacheKey.query_hash(query)}"
    
    @staticmethod
    def pattern_key(user_id: str) -> str:
        """Generate user pattern key"""
        return f"{CacheKey.PATTERN_PREFIX}{user_id}"
    
    @staticmethod
    def conversation_key(session_id: str) -> str:
        """Generate conversation history key"""
        return f"{CacheKey.CONVERSATION_PREFIX}{session_id}"
    
    @staticmethod
    def budget_key(user_id: str) -> str:
        """Generate budget tracking key"""
        return f"{CacheKey.BUDGET_PREFIX}{user_id}"
    
    @staticmethod
    def rate_key(user_id: str) -> str:
        """Generate rate limiting key"""
        return f"{CacheKey.RATE_PREFIX}{user_id}"


class CacheStrategy:
    """Cache strategy definitions"""
    
    # TTL values (in seconds)
    TTL_SHORT = 300      # 5 minutes
    TTL_MEDIUM = 1800    # 30 minutes
    TTL_LONG = 3600      # 1 hour
    TTL_DAY = 86400      # 24 hours
    
    # Cache strategies by data type
    STRATEGIES = {
        "routing": {"ttl": TTL_SHORT, "max_size": 10000},
        "responses": {"ttl": TTL_MEDIUM, "max_size": 5000},
        "conversations": {"ttl": TTL_DAY, "max_size": 100},
        "patterns": {"ttl": TTL_LONG, "max_size": 1000},
        "metrics": {"ttl": TTL_SHORT, "max_size": 500}
    }


class CacheMetrics(BaseModel):
    """Cache performance metrics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    avg_response_time: float = 0.0
    memory_usage: int = 0
    
    def update_hit(self, response_time: float):
        """Update metrics for cache hit"""
        self.total_requests += 1
        self.cache_hits += 1
        self._update_hit_rate()
        self._update_avg_response_time(response_time)
    
    def update_miss(self, response_time: float):
        """Update metrics for cache miss"""
        self.total_requests += 1
        self.cache_misses += 1
        self._update_hit_rate()
        self._update_avg_response_time(response_time)
    
    def _update_hit_rate(self):
        """Calculate hit rate"""
        if self.total_requests > 0:
            self.hit_rate = self.cache_hits / self.total_requests
    
    def _update_avg_response_time(self, response_time: float):
        """Update average response time"""
        if self.total_requests == 1:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (
                (self.avg_response_time * (self.total_requests - 1) + response_time)
                / self.total_requests
            )


class CacheManager:
    """Redis-based cache manager for hot layer"""
    
    def __init__(self, redis_url: str, max_connections: int = 20):
        self.redis_url = redis_url
        self.max_connections = max_connections
        self.settings = get_settings()
        
        # Redis clients
        self.redis: Optional[redis.Redis] = None
        self.redis_pool: Optional[redis.ConnectionPool] = None
        
        # Performance tracking
        self.metrics = CacheMetrics()
        
        # Local in-memory cache for emergency fallback
        self._local_cache: Dict[str, tuple[Any, datetime]] = {}
        self._local_cache_max_size = 1000
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            # Create connection pool
            self.redis_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Create Redis client
            self.redis = redis.Redis(connection_pool=self.redis_pool)
            
            # Test connection
            await self.redis.ping()
            
            logger.info("Redis cache manager initialized", url=self.redis_url)
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            logger.warning("Falling back to local cache only")
    
    async def cleanup(self):
        """Cleanup Redis connections"""
        if self.redis:
            await self.redis.close()
        if self.redis_pool:
            await self.redis_pool.disconnect()
    
    async def health_check(self) -> bool:
        """Check Redis health"""
        try:
            if self.redis:
                await self.redis.ping()
                return True
            return False
        except Exception:
            return False
    
    # Core caching methods
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        start_time = datetime.now()
        
        try:
            # Try Redis first
            if self.redis:
                value = await self.redis.get(key)
                if value is not None:
                    response_time = (datetime.now() - start_time).total_seconds()
                    self.metrics.update_hit(response_time)
                    return json.loads(value)
            
            # Fall back to local cache
            if key in self._local_cache:
                value, expiry = self._local_cache[key]
                if datetime.now() < expiry:
                    response_time = (datetime.now() - start_time).total_seconds()
                    self.metrics.update_hit(response_time)
                    return value
                else:
                    # Remove expired entry
                    del self._local_cache[key]
            
            # Cache miss
            response_time = (datetime.now() - start_time).total_seconds()
            self.metrics.update_miss(response_time)
            return default
            
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            response_time = (datetime.now() - start_time).total_seconds()
            self.metrics.update_miss(response_time)
            return default
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache"""
        try:
            serialized_value = json.dumps(value)
            
            # Set in Redis
            if self.redis:
                if ttl:
                    await self.redis.setex(key, ttl, serialized_value)
                else:
                    await self.redis.set(key, serialized_value)
            
            # Also set in local cache as backup
            expiry = datetime.now() + timedelta(seconds=ttl or CacheStrategy.TTL_MEDIUM)
            self._local_cache[key] = (value, expiry)
            
            # Cleanup local cache if too large
            self._cleanup_local_cache()
            
            return True
            
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            
            # Fall back to local cache only
            try:
                expiry = datetime.now() + timedelta(seconds=ttl or CacheStrategy.TTL_MEDIUM)
                self._local_cache[key] = (value, expiry)
                self._cleanup_local_cache()
                return True
            except Exception as local_e:
                logger.error(f"Local cache set error: {local_e}")
                return False
    
    def _cleanup_local_cache(self):
        """Cleanup local cache if it exceeds max size"""
        if len(self._local_cache) > self._local_cache_max_size:
            # Remove oldest entries
            now = datetime.now()
            
            # First remove expired entries
            expired_keys = [
                key for key, (_, expiry) in self._local_cache.items()
                if now >= expiry
            ]
            for key in expired_keys:
                del self._local_cache[key]
            
            # If still too large, remove oldest
            if len(self._local_cache) > self._local_cache_max_size:
                sorted_items = sorted(
                    self._local_cache.items(),
                    key=lambda x: x[1][1]  # Sort by expiry time
                )
                # Keep only the newest entries
                self._local_cache = dict(sorted_items[-self._local_cache_max_size:])
    
    # Specialized caching methods
    
    async def get_cached_route(self, query: str) -> Optional[List[str]]:
        """Get cached routing decision"""
        route_key = CacheKey.route_key(query)
        return await self.get(route_key)
    
    async def cache_successful_route(
        self, 
        query: str, 
        route: List[str], 
        cost: float
    ):
        """Cache a successful routing pattern"""
        route_key = CacheKey.route_key(query)
        route_data = {
            "route": route,
            "cost": cost,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
        await self.set(route_key, route_data, CacheStrategy.TTL_SHORT)
    
    async def get_user_pattern(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user behavior pattern"""
        pattern_key = CacheKey.pattern_key(user_id)
        return await self.get(pattern_key)
    
    async def update_user_pattern(
        self, 
        user_id: str, 
        pattern_data: Dict[str, Any]
    ):
        """Update user behavior pattern"""
        pattern_key = CacheKey.pattern_key(user_id)
        
        # Merge with existing pattern
        existing_pattern = await self.get_user_pattern(user_id) or {}
        existing_pattern.update(pattern_data)
        existing_pattern["last_updated"] = datetime.now().isoformat()
        
        await self.set(pattern_key, existing_pattern, CacheStrategy.TTL_LONG)
    
    async def get_conversation_history(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get conversation history"""
        conv_key = CacheKey.conversation_key(session_id)
        return await self.get(conv_key)
    
    async def update_conversation_history(
        self, 
        session_id: str, 
        history: List[Dict[str, Any]]
    ):
        """Update conversation history"""
        conv_key = CacheKey.conversation_key(session_id)
        
        # Limit history size to prevent memory issues
        max_history = 50  # Keep last 50 exchanges
        if len(history) > max_history:
            history = history[-max_history:]
        
        await self.set(conv_key, history, CacheStrategy.TTL_DAY)
    
    async def cache_successful_pattern(
        self, 
        query: str, 
        execution_path: List[str], 
        cost: float
    ):
        """Cache a successful execution pattern for optimization"""
        query_hash = CacheKey.query_hash(query)
        pattern_key = f"{CacheKey.SHORTCUT_PREFIX}{query_hash}"
        
        pattern_data = {
            "execution_path": execution_path,
            "cost": cost,
            "query_sample": query[:100],  # Store sample for debugging
            "timestamp": datetime.now().isoformat(),
            "usage_count": 1
        }
        
        # Check if pattern already exists and increment usage
        existing = await self.get(pattern_key)
        if existing:
            pattern_data["usage_count"] = existing.get("usage_count", 0) + 1
        
        await self.set(pattern_key, pattern_data, CacheStrategy.TTL_LONG)
    
    # Budget and rate limiting
    
    async def get_remaining_budget(self, user_id: str) -> float:
        """Get user's remaining budget"""
        budget_key = CacheKey.budget_key(user_id)
        budget_data = await self.get(budget_key)
        
        if not budget_data:
            # Initialize with default budget
            default_budget = self.settings.default_monthly_budget
            await self.set_budget(user_id, default_budget)
            return default_budget
        
        return budget_data.get("remaining", 0.0)
    
    async def deduct_budget(self, user_id: str, cost: float) -> float:
        """Deduct cost from user budget and return remaining"""
        budget_key = CacheKey.budget_key(user_id)
        budget_data = await self.get(budget_key) or {}
        
        current_remaining = budget_data.get("remaining", self.settings.default_monthly_budget)
        new_remaining = max(0.0, current_remaining - cost)
        
        budget_data.update({
            "remaining": new_remaining,
            "total_spent": budget_data.get("total_spent", 0.0) + cost,
            "last_transaction": datetime.now().isoformat(),
            "transaction_count": budget_data.get("transaction_count", 0) + 1
        })
        
        await self.set(budget_key, budget_data, CacheStrategy.TTL_DAY)
        return new_remaining
    
    async def set_budget(self, user_id: str, budget: float):
        """Set user budget"""
        budget_key = CacheKey.budget_key(user_id)
        budget_data = {
            "remaining": budget,
            "total_budget": budget,
            "total_spent": 0.0,
            "reset_date": datetime.now().isoformat(),
            "transaction_count": 0
        }
        await self.set(budget_key, budget_data, CacheStrategy.TTL_DAY)
    
    async def check_rate_limit(self, user_id: str, limit_per_minute: int = 60) -> tuple[bool, int]:
        """Check if user is within rate limits"""
        rate_key = CacheKey.rate_key(user_id)
        
        try:
            if self.redis:
                # Use Redis for accurate rate limiting
                current_minute = datetime.now().strftime("%Y%m%d%H%M")
                minute_key = f"{rate_key}:{current_minute}"
                
                # Get current count
                current_count = await self.redis.get(minute_key)
                current_count = int(current_count) if current_count else 0
                
                if current_count >= limit_per_minute:
                    return False, current_count
                
                # Increment counter
                pipe = self.redis.pipeline()
                pipe.incr(minute_key)
                pipe.expire(minute_key, 60)  # Expire after 1 minute
                await pipe.execute()
                
                return True, current_count + 1
            else:
                # Fallback to simple check (less accurate)
                rate_data = await self.get(rate_key) or {"count": 0, "window_start": datetime.now().isoformat()}
                
                window_start = datetime.fromisoformat(rate_data["window_start"])
                now = datetime.now()
                
                # Reset if new minute
                if (now - window_start).total_seconds() >= 60:
                    rate_data = {"count": 0, "window_start": now.isoformat()}
                
                if rate_data["count"] >= limit_per_minute:
                    return False, rate_data["count"]
                
                rate_data["count"] += 1
                await self.set(rate_key, rate_data, 60)
                
                return True, rate_data["count"]
                
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # Fail open - allow request
            return True, 0
    
    # Performance hints and optimization
    
    async def cache_performance_hint(
        self, 
        query_type: str, 
        expected_time: float, 
        expected_confidence: float
    ):
        """Cache performance expectations for query types"""
        hint_key = f"{CacheKey.EXPECT_PREFIX}{query_type}"
        hint_data = {
            "expected_time": expected_time,
            "expected_confidence": expected_confidence,
            "sample_count": 1,
            "last_updated": datetime.now().isoformat()
        }
        
        # Update existing hint with moving average
        existing = await self.get(hint_key)
        if existing:
            sample_count = existing.get("sample_count", 0) + 1
            hint_data["expected_time"] = (
                (existing["expected_time"] * (sample_count - 1) + expected_time) / sample_count
            )
            hint_data["expected_confidence"] = (
                (existing["expected_confidence"] * (sample_count - 1) + expected_confidence) / sample_count
            )
            hint_data["sample_count"] = sample_count
        
        await self.set(hint_key, hint_data, CacheStrategy.TTL_LONG)
    
    async def get_performance_hint(self, query_type: str) -> Optional[Dict[str, float]]:
        """Get performance expectations for query type"""
        hint_key = f"{CacheKey.EXPECT_PREFIX}{query_type}"
        return await self.get(hint_key)
    
    async def cache_optimal_model(self, task_type: str, model_name: str, success_rate: float):
        """Cache optimal model choice for task type"""
        model_key = f"{CacheKey.MODEL_PREFIX}{task_type}"
        model_data = {
            "model_name": model_name,
            "success_rate": success_rate,
            "last_updated": datetime.now().isoformat(),
            "usage_count": 1
        }
        
        # Update existing data
        existing = await self.get(model_key)
        if existing:
            model_data["usage_count"] = existing.get("usage_count", 0) + 1
            # Keep model with higher success rate
            if existing.get("success_rate", 0) > success_rate:
                model_data.update({
                    "model_name": existing["model_name"],
                    "success_rate": existing["success_rate"]
                })
        
        await self.set(model_key, model_data, CacheStrategy.TTL_LONG)
    
    async def get_optimal_model(self, task_type: str) -> Optional[str]:
        """Get cached optimal model for task type"""
        model_key = f"{CacheKey.MODEL_PREFIX}{task_type}"
        model_data = await self.get(model_key)
        return model_data.get("model_name") if model_data else None
    
    # Analytics and metrics
    
    async def store_execution_metrics(self, query_id: str, metrics: Dict[str, Any]):
        """Store execution metrics for analytics"""
        metrics_key = f"{CacheKey.METRICS_PREFIX}{query_id}"
        await self.set(metrics_key, metrics, CacheStrategy.TTL_DAY)
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide cache statistics"""
        stats = {
            "cache_metrics": asdict(self.metrics),
            "local_cache_size": len(self._local_cache),
            "redis_connected": await self.health_check()
        }
        
        # Get Redis memory info if available
        try:
            if self.redis:
                info = await self.redis.info("memory")
                stats["redis_memory"] = {
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_human": info.get("used_memory_human", "0B"),
                    "max_memory": info.get("maxmemory", 0)
                }
        except Exception as e:
            logger.warning(f"Could not get Redis memory info: {e}")
        
        return stats
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get cache manager metrics"""
        return {
            "total_requests": self.metrics.total_requests,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "hit_rate": self.metrics.hit_rate,
            "avg_response_time": self.metrics.avg_response_time,
            "local_cache_size": len(self._local_cache),
            "redis_connected": await self.health_check()
        }
    
    # Bulk operations for optimization
    
    async def scan_by_pattern(
        self, 
        pattern: str, 
        count: int = 100
    ) -> List[tuple[str, Any]]:
        """Scan cache keys by pattern"""
        results = []
        
        try:
            if self.redis:
                cursor = 0
                while True:
                    cursor, keys = await self.redis.scan(cursor, match=pattern, count=count)
                    
                    if keys:
                        # Get values for all keys
                        pipe = self.redis.pipeline()
                        for key in keys:
                            pipe.get(key)
                        values = await pipe.execute()
                        
                        for key, value in zip(keys, values):
                            if value:
                                try:
                                    results.append((key, json.loads(value)))
                                except json.JSONDecodeError:
                                    continue
                    
                    if cursor == 0:
                        break
                    
                    if len(results) >= count:
                        break
        except Exception as e:
            logger.error(f"Pattern scan error: {e}")
        
        return results
    
    async def batch_delete(self, keys: List[str]) -> int:
        """Delete multiple keys in batch"""
        try:
            if self.redis and keys:
                deleted = await self.redis.delete(*keys)
                
                # Also remove from local cache
                for key in keys:
                    self._local_cache.pop(key, None)
                
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Batch delete error: {e}")
            return 0
    
    async def clear_expired(self):
        """Clear expired entries from local cache"""
        now = datetime.now()
        expired_keys = [
            key for key, (_, expiry) in self._local_cache.items()
            if now >= expiry
        ]
        
        for key in expired_keys:
            del self._local_cache[key]
        
        logger.info(f"Cleared {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
