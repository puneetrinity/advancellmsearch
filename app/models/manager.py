# app/models/manager.py
"""
Model Manager - Handles all model operations and lifecycle
Manages Ollama local models with intelligent loading/unloading
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import aiohttp
import structlog
from pydantic import BaseModel

from app.core.config import get_settings, MODEL_ASSIGNMENTS, PRIORITY_TIERS, API_COSTS
from app.cache.redis_client import CacheManager

logger = structlog.get_logger(__name__)


class ModelStatus(Enum):
    """Model status enumeration"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    WARMING = "warming"
    READY = "ready"
    ERROR = "error"


@dataclass
class ModelResult:
    """Result from model generation"""
    success: bool
    text: str = ""
    cost: float = 0.0
    execution_time: float = 0.0
    model_used: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ModelInfo:
    """Information about a model"""
    name: str
    status: ModelStatus
    last_used: datetime
    load_time: float = 0.0
    total_requests: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    memory_usage: float = 0.0
    tier: str = "T2"
    
    def update_stats(self, execution_time: float, cost: float):
        """Update model statistics"""
        self.last_used = datetime.now()
        self.total_requests += 1
        self.total_cost += cost
        
        # Update average response time
        if self.avg_response_time == 0:
            self.avg_response_time = execution_time
        else:
            self.avg_response_time = (
                (self.avg_response_time * (self.total_requests - 1) + execution_time) 
                / self.total_requests
            )


class OllamaClient:
    """Async client for Ollama API"""
    
    def __init__(self, host: str, timeout: int = 60):
        self.host = host.rstrip('/')
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self):
        """Initialize the HTTP session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        try:
            async with self.session.get(f"{self.host}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("models", [])
                else:
                    logger.error(f"Failed to list models: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from registry"""
        try:
            payload = {"name": model_name}
            async with self.session.post(
                f"{self.host}/api/pull", 
                json=payload
            ) as response:
                if response.status == 200:
                    # Stream the pull progress (simplified)
                    async for line in response.content:
                        if line:
                            # Log progress (could be enhanced with progress tracking)
                            pass
                    return True
                else:
                    logger.error(f"Failed to pull model {model_name}: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    async def generate(
        self, 
        model_name: str, 
        prompt: str, 
        max_tokens: int = 300,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None
    ) -> ModelResult:
        """Generate text using a model"""
        start_time = time.time()
        
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "stop": stop or []
                },
                "stream": False
            }
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload
            ) as response:
                execution_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    return ModelResult(
                        success=True,
                        text=data.get("response", ""),
                        cost=API_COSTS.get(model_name, 0.0),
                        execution_time=execution_time,
                        model_used=model_name,
                        metadata={
                            "total_duration": data.get("total_duration", 0),
                            "load_duration": data.get("load_duration", 0),
                            "prompt_eval_count": data.get("prompt_eval_count", 0),
                            "eval_count": data.get("eval_count", 0)
                        }
                    )
                else:
                    error_text = await response.text()
                    return ModelResult(
                        success=False,
                        error=f"Model generation failed: {response.status} - {error_text}",
                        execution_time=execution_time,
                        model_used=model_name
                    )
                    
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            return ModelResult(
                success=False,
                error=f"Model generation timed out after {self.timeout}s",
                execution_time=execution_time,
                model_used=model_name
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ModelResult(
                success=False,
                error=f"Model generation error: {str(e)}",
                execution_time=execution_time,
                model_used=model_name
            )
    
    async def check_model_status(self, model_name: str) -> bool:
        """Check if a model is loaded and ready"""
        try:
            # Try a simple generation to check if model is ready
            payload = {
                "model": model_name,
                "prompt": "test",
                "options": {"num_predict": 1},
                "stream": False
            }
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload
            ) as response:
                return response.status == 200
                
        except Exception:
            return False


class ModelManager:
    """Manages model lifecycle and selection"""
    
    def __init__(self, ollama_host: str, cache_manager: Optional[CacheManager] = None):
        self.settings = get_settings()
        self.ollama_client = OllamaClient(ollama_host, self.settings.ollama_timeout)
        self.cache_manager = cache_manager
        
        # Model registry
        self.models: Dict[str, ModelInfo] = {}
        self.loaded_models: Dict[str, datetime] = {}
        
        # Performance tracking
        self.total_requests = 0
        self.total_cost = 0.0
        self.local_requests = 0
        
        # Model loading locks
        self._loading_locks: Dict[str, asyncio.Lock] = {}
    
    async def initialize(self):
        """Initialize the model manager"""
        await self.ollama_client.initialize()
        
        # Initialize model registry
        await self._initialize_model_registry()
        
        logger.info("Model manager initialized", models=list(self.models.keys()))
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.ollama_client.cleanup()
    
    async def _initialize_model_registry(self):
        """Initialize the model registry with available models"""
        try:
            # Get available models from Ollama
            available_models = await self.ollama_client.list_models()
            available_names = [model["name"] for model in available_models]
            
            # Initialize model info for all expected models
            for assignment_type, model_name in MODEL_ASSIGNMENTS.items():
                tier = self._get_model_tier(model_name)
                
                self.models[model_name] = ModelInfo(
                    name=model_name,
                    status=ModelStatus.LOADED if model_name in available_names else ModelStatus.UNLOADED,
                    last_used=datetime.now() - timedelta(hours=1),  # Default to 1 hour ago
                    tier=tier
                )
                
                # Create loading lock
                self._loading_locks[model_name] = asyncio.Lock()
            
            logger.info(
                "Model registry initialized",
                total_models=len(self.models),
                loaded_models=len([m for m in self.models.values() if m.status == ModelStatus.LOADED])
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize model registry: {e}")
            raise
    
    def _get_model_tier(self, model_name: str) -> str:
        """Get the tier for a model"""
        for tier, models in PRIORITY_TIERS.items():
            if model_name in models:
                return tier
        return "T2"  # Default to lowest priority
    
    async def preload_models(self, model_names: List[str]):
        """Preload critical models"""
        for model_name in model_names:
            try:
                await self._ensure_model_loaded(model_name)
                logger.info(f"Preloaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to preload model {model_name}: {e}")
    
    async def _ensure_model_loaded(self, model_name: str) -> bool:
        """Ensure a model is loaded and ready"""
        if model_name not in self.models:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        model_info = self.models[model_name]
        
        # If already loaded and ready, return
        if model_info.status == ModelStatus.READY:
            return True
        
        # Use lock to prevent concurrent loading
        async with self._loading_locks[model_name]:
            # Check again in case another coroutine loaded it
            if model_info.status == ModelStatus.READY:
                return True
            
            try:
                model_info.status = ModelStatus.LOADING
                start_time = time.time()
                
                # Check if model exists locally
                available_models = await self.ollama_client.list_models()
                available_names = [model["name"] for model in available_models]
                
                if model_name not in available_names:
                    logger.info(f"Pulling model {model_name}...")
                    success = await self.ollama_client.pull_model(model_name)
                    if not success:
                        model_info.status = ModelStatus.ERROR
                        return False
                
                # Test if model is ready
                is_ready = await self.ollama_client.check_model_status(model_name)
                
                if is_ready:
                    load_time = time.time() - start_time
                    model_info.status = ModelStatus.READY
                    model_info.load_time = load_time
                    self.loaded_models[model_name] = datetime.now()
                    
                    logger.info(
                        f"Model {model_name} loaded successfully",
                        load_time=load_time
                    )
                    return True
                else:
                    model_info.status = ModelStatus.ERROR
                    logger.error(f"Model {model_name} failed to load properly")
                    return False
                    
            except Exception as e:
                model_info.status = ModelStatus.ERROR
                logger.error(f"Error loading model {model_name}: {e}")
                return False
    
    async def generate(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.7,
        fallback: bool = True
    ) -> ModelResult:
        """Generate text using specified model with fallback"""
        self.total_requests += 1
        
        # Try primary model
        result = await self._generate_with_model(
            model_name, prompt, max_tokens, temperature
        )
        
        if result.success:
            self.local_requests += 1
            return result
        
        # Fallback logic
        if fallback:
            fallback_model = self._get_fallback_model(model_name)
            if fallback_model and fallback_model != model_name:
                logger.warning(
                    f"Falling back from {model_name} to {fallback_model}",
                    original_error=result.error
                )
                
                fallback_result = await self._generate_with_model(
                    fallback_model, prompt, max_tokens, temperature
                )
                
                if fallback_result.success:
                    self.local_requests += 1
                    fallback_result.metadata["fallback_from"] = model_name
                    return fallback_result
        
        return result
    
    async def _generate_with_model(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> ModelResult:
        """Generate with a specific model"""
        # Ensure model is loaded
        if not await self._ensure_model_loaded(model_name):
            return ModelResult(
                success=False,
                error=f"Failed to load model {model_name}",
                model_used=model_name
            )
        
        # Generate
        result = await self.ollama_client.generate(
            model_name, prompt, max_tokens, temperature
        )
        
        # Update statistics
        if model_name in self.models:
            self.models[model_name].update_stats(result.execution_time, result.cost)
        
        self.total_cost += result.cost
        
        return result
    
    def _get_fallback_model(self, original_model: str) -> Optional[str]:
        """Get fallback model for a failed model"""
        fallback_mapping = {
            "phi:mini": "llama2:7b",
            "llama2:7b": "phi:mini",
            "mistral:7b": "llama2:7b",
            "llama2:13b": "mistral:7b",
            "codellama": "llama2:7b"
        }
        
        return fallback_mapping.get(original_model)
    
    def select_optimal_model(self, task_type: str, quality_requirement: str = "balanced") -> str:
        """Select optimal model for a task"""
        # Get base model for task type
        base_model = MODEL_ASSIGNMENTS.get(task_type, "llama2:7b")
        
        # Adjust based on quality requirement
        if quality_requirement == "minimal":
            return "phi:mini"
        elif quality_requirement == "premium":
            if task_type in ["analytical_reasoning", "deep_research"]:
                return "llama2:13b"
            elif task_type == "code_tasks":
                return "codellama"
        
        return base_model
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get model manager metrics"""
        local_percentage = (self.local_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        model_stats = {}
        for name, info in self.models.items():
            model_stats[name] = {
                "status": info.status.value,
                "total_requests": info.total_requests,
                "avg_response_time": info.avg_response_time,
                "last_used": info.last_used.isoformat(),
                "tier": info.tier
            }
        
        return {
            "total_requests": self.total_requests,
            "local_requests": self.local_requests,
            "local_percentage": local_percentage,
            "total_cost": self.total_cost,
            "loaded_models": len([m for m in self.models.values() if m.status == ModelStatus.READY]),
            "model_stats": model_stats
        }
    
    def is_healthy(self) -> bool:
        """Check if model manager is healthy"""
        # At least one model should be ready
        ready_models = [m for m in self.models.values() if m.status == ModelStatus.READY]
        return len(ready_models) > 0
