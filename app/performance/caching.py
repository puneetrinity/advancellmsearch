"""
Smart Caching System
Multi-strategy caching with TTL and LRU eviction for performance optimization
"""

import time
import json
import hashlib
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import threading

@dataclass
class CacheEntry:
    """Cache entry with TTL and metadata"""
    key: str
    value: Any
    created_at: float
    ttl: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl
    
    def access(self):
        self.access_count += 1
        self.last_accessed = time.time()

class SmartCache:
    """Multi-strategy caching system with TTL and LRU eviction"""
    
    def __init__(self, max_size: int = 10000, default_ttl: float = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            entry = self.cache.get(key)
            
            if entry is None:
                self.misses += 1
                return None
            
            if entry.is_expired():
                del self.cache[key]
                self.misses += 1
                return None
            
            entry.access()
            self.hits += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache"""
        with self._lock:
            if ttl is None:
                ttl = self.default_ttl
            
            # Calculate approximate size
            try:
                size_bytes = len(json.dumps(value, default=str).encode())
            except:
                size_bytes = 1000  # Estimate
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            self.cache[key] = entry
            
            # Evict if necessary
            self._evict_if_needed()
    
    def _evict_if_needed(self):
        """Evict entries if cache is too large"""
        while len(self.cache) > self.max_size:
            # Find LRU entry
            lru_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].last_accessed
            )
            del self.cache[lru_key]
            self.evictions += 1
    
    def clear_expired(self):
        """Clear expired entries"""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "current_size": len(self.cache),
            "max_size": self.max_size,
            "total_size_bytes": sum(entry.size_bytes for entry in self.cache.values())
        }