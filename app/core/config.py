# app/core/config.py
"""
Configuration management for AI Search System
Environment-based configuration with sensible defaults
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support
    Provider configs (API keys, timeouts, etc) are loaded from environment variables or .env file.
    Use get_settings() to access per-environment config.
    Example usage for providers:
        brave_search_api_key: str = ...
        scrapingbee_api_key: str = ...
    """

    # Application
    app_name: str = "AI Search System"
    debug: bool = False
    environment: str = "development"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Ollama Configuration
    ollama_host: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://ollama:11434")
    )
    ollama_timeout: int = 60
    ollama_max_retries: int = 3

    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    redis_max_connections: int = 20
    redis_timeout: int = 5

    # Model Configuration
    default_model: str = "tinyllama:latest"
    fallback_model: str = "tinyllama:latest"
    max_concurrent_models: int = 3
    model_memory_threshold: float = 0.8

    # Cost & Budget Configuration
    cost_per_api_call: float = 0.008
    default_monthly_budget: float = 20.0
    cost_tracking_enabled: bool = True

    # Rate Limiting
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 10

    # Cache Configuration
    cache_ttl_default: int = 3600  # 1 hour
    cache_ttl_routing: int = 300  # 5 minutes
    cache_ttl_responses: int = 1800  # 30 minutes

    # Performance Targets
    target_response_time: float = 2.5
    target_local_processing: float = 0.85
    target_cache_hit_rate: float = 0.80

    # LangGraph Configuration
    graph_max_iterations: int = 50
    graph_timeout: int = 30
    graph_retry_attempts: int = 2

    # External APIs
    brave_search_api_key: Optional[str] = None
    zerows_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # json or text

    # Security
    jwt_secret_key: str = "dev-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24

    # Database (for future use)
    database_url: Optional[str] = None

    # New search provider settings
    BRAVE_API_KEY: Optional[str] = None
    SCRAPINGBEE_API_KEY: Optional[str] = None
    # Performance settings
    CACHE_TTL: int = 3600
    MAX_CACHE_SIZE: int = 50000
    PERFORMANCE_MONITORING: bool = True
    # Search settings
    DEFAULT_SEARCH_BUDGET: float = 2.0
    DEFAULT_QUALITY_REQUIREMENT: str = "standard"
    MAX_SEARCH_RESULTS: int = 10

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "protected_namespaces": ("settings_",),
        "env_prefix": "",
        "populate_by_name": True,
        "extra": "ignore",
    }


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
    # Use env var if set, else default
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/1")
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
    "simple_classification": "tinyllama:latest",
    "qa_and_summary": "tinyllama:latest",
    "analytical_reasoning": "tinyllama:latest",
    "deep_research": "tinyllama:latest",
    "code_tasks": "tinyllama:latest",
    "multilingual": "tinyllama:latest",
}

PRIORITY_TIERS = {
    "T0": ["tinyllama:latest"],  # Always loaded (hot)
    "T1": ["tinyllama:latest"],  # Loaded on first use, kept warm
    "T2": ["tinyllama:latest"],  # Load/unload per request
}

# API costs (in INR)
API_COSTS = {
    "phi:mini": 0.0,
    "llama2:7b": 0.0,
    "mistral:7b": 0.0,
    "gpt-4": 0.06,
    "claude-haiku": 0.01,
    "brave_search": 0.008,
    "zerows_scraping": 0.02,
}

# Rate limits by tier
RATE_LIMITS = {
    "free": {"requests_per_minute": 10, "cost_per_month": 20.0},
    "pro": {"requests_per_minute": 100, "cost_per_month": 500.0},
    "enterprise": {"requests_per_minute": 1000, "cost_per_month": 5000.0},
}
