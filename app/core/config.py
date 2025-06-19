# app/core/config.py
"""
Configuration management for AI Search System
Environment-based configuration with sensible defaults
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "AI Search System"
    debug: bool = Field(False, env="DEBUG")
    environment: str = Field("development", env="ENVIRONMENT")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    allowed_origins: List[str] = Field(
        ["http://localhost:3000", "http://localhost:8000"],
        env="ALLOWED_ORIGINS"
    )
    
    # Ollama Configuration
    ollama_host: str = Field("http://localhost:11434", env="OLLAMA_HOST")
    ollama_timeout: int = Field(60, env="OLLAMA_TIMEOUT")
    ollama_max_retries: int = Field(3, env="OLLAMA_MAX_RETRIES")
    
    # Redis Configuration
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    redis_max_connections: int = Field(20, env="REDIS_MAX_CONNECTIONS")
    redis_timeout: int = Field(5, env="REDIS_TIMEOUT")
    
    # Model Configuration
    default_model: str = Field("phi:mini", env="DEFAULT_MODEL")
    fallback_model: str = Field("llama2:7b", env="FALLBACK_MODEL")
    max_concurrent_models: int = Field(3, env="MAX_CONCURRENT_MODELS")
    model_memory_threshold: float = Field(0.8, env="MODEL_MEMORY_THRESHOLD")
    
    # Cost & Budget Configuration
    cost_per_api_call: float = Field(0.008, env="COST_PER_API_CALL")
    default_monthly_budget: float = Field(20.0, env="DEFAULT_MONTHLY_BUDGET")
    cost_tracking_enabled: bool = Field(True, env="COST_TRACKING_ENABLED")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_burst: int = Field(10, env="RATE_LIMIT_BURST")
    
    # Cache Configuration
    cache_ttl_default: int = Field(3600, env="CACHE_TTL_DEFAULT")  # 1 hour
    cache_ttl_routing: int = Field(300, env="CACHE_TTL_ROUTING")   # 5 minutes
    cache_ttl_responses: int = Field(1800, env="CACHE_TTL_RESPONSES")  # 30 minutes
    
    # Performance Targets
    target_response_time: float = Field(2.5, env="TARGET_RESPONSE_TIME")
    target_local_processing: float = Field(0.85, env="TARGET_LOCAL_PROCESSING")
    target_cache_hit_rate: float = Field(0.80, env="TARGET_CACHE_HIT_RATE")
    
    # LangGraph Configuration
    graph_max_iterations: int = Field(50, env="GRAPH_MAX_ITERATIONS")
    graph_timeout: int = Field(30, env="GRAPH_TIMEOUT")
    graph_retry_attempts: int = Field(2, env="GRAPH_RETRY_ATTEMPTS")
    
    # External APIs
    brave_search_api_key: Optional[str] = Field(None, env="BRAVE_SEARCH_API_KEY")
    zerows_api_key: Optional[str] = Field(None, env="ZEROWS_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")  # json or text
    
    # Security
    jwt_secret_key: str = Field("dev-secret-key", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    jwt_expiry_hours: int = Field(24, env="JWT_EXPIRY_HOURS")
    
    # Database (for future use)
    database_url: Optional[str] = Field(None, env="DATABASE_URL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class DevelopmentSettings(Settings):
    """Development environment settings"""
    debug: bool = True
    log_level: str = "DEBUG"
    environment: str = "development"


class ProductionSettings(Settings):
    """Production environment settings"""
    debug: bool = False
    log_level: str = "INFO"
    environment: str = "production"
    
    # More restrictive settings for production
    redis_max_connections: int = 50
    rate_limit_per_minute: int = 100
    target_response_time: float = 2.0


class TestingSettings(Settings):
    """Testing environment settings"""
    debug: bool = True
    environment: str = "testing"
    
    # Use in-memory or test databases
    redis_url: str = "redis://localhost:6379/1"  # Different DB for tests
    
    # Faster timeouts for tests
    ollama_timeout: int = 10
    redis_timeout: int = 1
    graph_timeout: int = 5


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)"""
    import os
    
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Model configuration constants
MODEL_ASSIGNMENTS = {
    "simple_classification": "phi:mini",
    "qa_and_summary": "llama2:7b",
    "analytical_reasoning": "mistral:7b",
    "deep_research": "llama2:13b",
    "code_tasks": "codellama",
    "multilingual": "aya:8b"
}

PRIORITY_TIERS = {
    "T0": ["phi:mini"],  # Always loaded (hot)
    "T1": ["llama2:7b"],  # Loaded on first use, kept warm
    "T2": ["llama2:13b", "mistral:7b"]  # Load/unload per request
}

# API costs (in INR)
API_COSTS = {
    "phi:mini": 0.0,
    "llama2:7b": 0.0,
    "mistral:7b": 0.0,
    "gpt-4": 0.06,
    "claude-haiku": 0.01,
    "brave_search": 0.008,
    "zerows_scraping": 0.02
}

# Rate limits by tier
RATE_LIMITS = {
    "free": {"requests_per_minute": 10, "cost_per_month": 20.0},
    "pro": {"requests_per_minute": 100, "cost_per_month": 500.0},
    "enterprise": {"requests_per_minute": 1000, "cost_per_month": 5000.0}
}
