"""
ModelManager - Intelligent model lifecycle management with cost optimization.
Handles model loading, selection, fallbacks, and performance tracking.
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from app.models.ollama_client import OllamaClient, ModelResult, ModelStatus, OllamaException
from app.core.logging import get_logger, get_correlation_id, log_performance
from app.core.config import get_settings


logger = get_logger("models.manager")
settings = get_settings()


class TaskType(str, Enum):
    """Task types for model selection."""
    SIMPLE_CLASSIFICATION = "simple_classification"
    QA_AND_SUMMARY = "qa_and_summary"
    ANALYTICAL_REASONING = "analytical_reasoning"
    DEEP_RESEARCH = "deep_research"
    CODE_TASKS = "code_tasks"
    MULTILINGUAL = "multilingual"
    CREATIVE_WRITING = "creative_writing"
    CONVERSATION = "conversation"


class QualityLevel(str, Enum):
    """Quality requirements for model selection."""
    MINIMAL = "minimal"      # Fastest response, basic quality
    BALANCED = "balanced"    # Good balance of speed and quality
    HIGH = "high"           # High quality, reasonable speed
    PREMIUM = "premium"     # Best quality, may be slower/expensive


@dataclass
class ModelInfo:
    """Information about a model including performance metrics."""
    name: str
    status: ModelStatus = ModelStatus.UNKNOWN
    last_used: datetime = field(default_factory=datetime.now)
    load_time: float = 0.0
    total_requests: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    avg_tokens_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    tier: str = "T2"  # T0=always loaded, T1=keep warm, T2=load on demand
    success_rate: float = 1.0
    confidence_scores: List[float] = field(default_factory=list)
    
    def update_stats(self, result: ModelResult, confidence: float = 0.0):
        """Update model performance statistics."""
        self.total_requests += 1
        self.last_used = datetime.now()
        
        if result.success:
            # Update response time (exponential moving average)
            alpha = 0.1
            if self.avg_response_time == 0:
                self.avg_response_time = result.execution_time
            else:
                self.avg_response_time = (alpha * result.execution_time + 
                                        (1 - alpha) * self.avg_response_time)
            
            # Update tokens per second
            if result.tokens_per_second:
                if self.avg_tokens_per_second == 0:
                    self.avg_tokens_per_second = result.tokens_per_second
                else:
                    self.avg_tokens_per_second = (alpha * result.tokens_per_second + 
                                                (1 - alpha) * self.avg_tokens_per_second)
            
            # Track confidence scores
            if confidence > 0:
                self.confidence_scores.append(confidence)
                # Keep only last 100 scores
                if len(self.confidence_scores) > 100:
                    self.confidence_scores = self.confidence_scores[-100:]
        
        # Update success rate
        recent_requests = min(self.total_requests, 20)  # Consider last 20 requests
        if hasattr(self, '_recent_successes'):
            self._recent_successes.append(result.success)
            if len(self._recent_successes) > recent_requests:
                self._recent_successes = self._recent_successes[-recent_requests:]
        else:
            self._recent_successes = [result.success]
        
        self.success_rate = sum(self._recent_successes) / len(self._recent_successes)
    
    @property
    def avg_confidence(self) -> float:
        """Calculate average confidence score."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)
    
    @property
    def performance_score(self) -> float:
        """Calculate overall performance score (0-1)."""
        # Weighted combination of metrics
        speed_score = min(1.0, 10.0 / max(self.avg_response_time, 0.1))  # 10s = 0 score
        quality_score = self.avg_confidence
        reliability_score = self.success_rate
        
        return (0.4 * speed_score + 0.3 * quality_score + 0.3 * reliability_score)


class ModelManager:
    """
    Intelligent model lifecycle manager with optimization and fallback strategies.
    
    Features:
    - Smart model selection based on task type and quality requirements
    - Automatic model loading and unloading based on usage patterns
    - Performance tracking and optimization recommendations
    - Fallback strategies for model failures
    - Cost optimization and budget management
    """
    
    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        cache_manager = None  # Optional cache manager for storing metrics
    ):
        self.ollama_client = OllamaClient(base_url=ollama_host)
        self.cache_manager = cache_manager
        
        # Model registry and performance tracking
        self.models: Dict[str, ModelInfo] = {}
        self.loaded_models: Set[str] = set()
        self._loading_locks: Dict[str, asyncio.Lock] = {}
        
        # Model assignment configuration
        self.model_assignments = {
            TaskType.SIMPLE_CLASSIFICATION: "phi:mini",
            TaskType.QA_AND_SUMMARY: "llama2:7b",
            TaskType.ANALYTICAL_REASONING: "mistral:7b",
            TaskType.DEEP_RESEARCH: "llama2:13b",
            TaskType.CODE_TASKS: "codellama",
            TaskType.MULTILINGUAL: "aya:8b",
            TaskType.CREATIVE_WRITING: "llama2:7b",
            TaskType.CONVERSATION: "llama2:7b"
        }
        
        # Priority tiers for memory management
        self.priority_tiers = {
            "T0": ["phi:mini"],  # Always loaded
            "T1": ["llama2:7b"],  # Keep warm
            "T2": ["llama2:13b", "mistral:7b", "codellama", "aya:8b"]  # Load on demand
        }
        
        # Quality-based model overrides
        self.quality_overrides = {
            QualityLevel.MINIMAL: {
                TaskType.QA_AND_SUMMARY: "phi:mini",
                TaskType.ANALYTICAL_REASONING: "llama2:7b",
                TaskType.DEEP_RESEARCH: "llama2:7b"
            },
            QualityLevel.PREMIUM: {
                TaskType.SIMPLE_CLASSIFICATION: "llama2:7b",
                TaskType.QA_AND_SUMMARY: "mistral:7b",
                TaskType.ANALYTICAL_REASONING: "llama2:13b"
            }
        }
        
        logger.info(
            "ModelManager initialized",
            ollama_host=ollama_host,
            model_assignments=len(self.model_assignments),
            correlation_id=get_correlation_id()
        )
    
    async def initialize(self) -> None:
        """Initialize the model manager and load priority models."""
        correlation_id = get_correlation_id()
        
        logger.info("Initializing ModelManager", correlation_id=correlation_id)
        
        try:
            # Initialize Ollama client
            await self.ollama_client.initialize()
            
            # Check Ollama health
            if not await self.ollama_client.health_check():
                logger.error("Ollama service is not healthy", correlation_id=correlation_id)
                raise OllamaException("Ollama service is not available")
            
            # Load available models from Ollama
            await self._discover_models()
            
            # Preload T0 models (always loaded)
            await self._preload_priority_models()
            
            logger.info(
                "ModelManager initialization completed",
                total_models=len(self.models),
                loaded_models=len(self.loaded_models),
                correlation_id=correlation_id
            )
            
        except Exception as e:
            logger.error(
                "ModelManager initialization failed",
                error=str(e),
                correlation_id=correlation_id,
                exc_info=True
            )
            raise
    
    @log_performance("model_discovery")
    async def _discover_models(self) -> None:
        """Discover available models and initialize model info."""
        correlation_id = get_correlation_id()
        
        try:
            available_models = await self.ollama_client.list_models()
            
            for model_data in available_models:
                model_name = model_data.get("name", "unknown")
                
                # Determine tier based on configuration
                tier = "T2"  # Default
                for tier_name, tier_models in self.priority_tiers.items():
                    if any(model_name.startswith(tm.split(':')[0]) for tm in tier_models):
                        tier = tier_name
                        break
                
                self.models[model_name] = ModelInfo(
                    name=model_name,
                    status=ModelStatus.READY,  # Assume ready if listed
                    tier=tier
                )
            
            logger.info(
                "Model discovery completed",
                discovered_models=len(self.models),
                model_names=list(self.models.keys()),
                correlation_id=correlation_id
            )
            
        except Exception as e:
            logger.error(
                "Model discovery failed",
                error=str(e),
                correlation_id=correlation_id
            )
            raise
    
    async def _preload_priority_models(self) -> None:
        """Preload T0 priority models for instant availability."""
        correlation_id = get_correlation_id()
        
        t0_models = self.priority_tiers.get("T0", [])
        
        for model_pattern in t0_models:
            # Find matching models
            matching_models = [
                name for name in self.models.keys()
                if name.startswith(model_pattern.split(':')[0])
            ]
            
            for model_name in matching_models:
                try:
                    await self._ensure_model_loaded(model_name)
                    logger.info(
                        "Priority model preloaded",
                        model_name=model_name,
                        tier="T0",
                        correlation_id=correlation_id
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to preload priority model",
                        model_name=model_name,
                        error=str(e),
                        correlation_id=correlation_id
                    )
    
    def select_optimal_model(
        self,
        task_type: TaskType,
        quality_requirement: QualityLevel = QualityLevel.BALANCED,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Select the optimal model for a given task and quality requirement.
        
        Args:
            task_type: Type of task to perform
            quality_requirement: Required quality level
            context: Additional context for selection (user tier, budget, etc.)
            
        Returns:
            str: Selected model name
        """
        correlation_id = get_correlation_id()
        
        # Start with base assignment
        base_model = self.model_assignments.get(task_type, "llama2:7b")
        
        # Apply quality overrides
        if quality_requirement in self.quality_overrides:
            quality_overrides = self.quality_overrides[quality_requirement]
            if task_type in quality_overrides:
                base_model = quality_overrides[task_type]
        
        # Check if model is available
        available_models = [name for name in self.models.keys() 
                          if name.startswith(base_model.split(':')[0])]
        
        if not available_models:
            # Fallback to any available model
            fallback_model = self._select_fallback_model(task_type)
            logger.warning(
                "Preferred model not available, using fallback",
                preferred_model=base_model,
                fallback_model=fallback_model,
                task_type=task_type.value,
                correlation_id=correlation_id
            )
            return fallback_model
        
        # Select best performing variant if multiple available
        selected_model = self._select_best_variant(available_models)
        
        logger.debug(
            "Model selected",
            task_type=task_type.value,
            quality_requirement=quality_requirement.value,
            selected_model=selected_model,
            correlation_id=correlation_id
        )
        
        return selected_model
    
    def _select_fallback_model(self, task_type: TaskType) -> str:
        """Select a fallback model when preferred model is unavailable."""
        # Fallback hierarchy
        fallback_hierarchy = [
            "llama2:7b", "phi:mini", "mistral:7b"
        ]
        
        for fallback in fallback_hierarchy:
            available = [name for name in self.models.keys() 
                        if name.startswith(fallback.split(':')[0])]
            if available:
                return available[0]
        
        # Last resort - return first available model
        if self.models:
            return list(self.models.keys())[0]
        
        raise OllamaException("No models available")
    
    def _select_best_variant(self, available_models: List[str]) -> str:
        """Select the best performing variant from available models."""
        if len(available_models) == 1:
            return available_models[0]
        
        # Score models based on performance metrics
        scored_models = []
        for model_name in available_models:
            model_info = self.models.get(model_name)
            if model_info:
                score = model_info.performance_score
                scored_models.append((model_name, score))
        
        if scored_models:
            # Sort by score (highest first)
            scored_models.sort(key=lambda x: x[1], reverse=True)
            return scored_models[0][0]
        
        return available_models[0]
    
    @log_performance("model_generation")
    async def generate(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.7,
        fallback: bool = True,
        **kwargs
    ) -> ModelResult:
        """
        Generate text using specified model with automatic fallback.
        
        Args:
            model_name: Model to use for generation
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            fallback: Enable fallback to alternative models
            **kwargs: Additional generation parameters
            
        Returns:
            ModelResult: Generation result with metadata
        """
        correlation_id = get_correlation_id()
        
        logger.debug(
            "Starting text generation",
            model_name=model_name,
            prompt_length=len(prompt),
            max_tokens=max_tokens,
            temperature=temperature,
            correlation_id=correlation_id
        )
        
        # Try primary model first
        try:
            # Ensure model is loaded
            await self._ensure_model_loaded(model_name)
            
            # Generate with primary model
            result = await self.ollama_client.generate(
                model_name=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Update model stats with the result object and default confidence
            if model_name in self.models:
                self.models[model_name].update_stats(result, confidence=0.8)  # pass result object
                result.cost = 0.0  # Local models have no cost
            
            if result.success:
                logger.debug(
                    "Primary model generation successful",
                    model_name=model_name,
                    correlation_id=correlation_id
                )
                return result
            else:
                # Primary model failed, try fallback if enabled
                if fallback:
                    logger.warning(
                        "Primary model generation failed, trying fallback",
                        primary_model=model_name,
                        error=result.error,
                        correlation_id=correlation_id
                    )
                    return await self._try_fallback_generation(
                        original_model=model_name,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **kwargs
                    )
                else:
                    return result
                    
        except Exception as e:
            logger.error(
                "Model generation failed",
                model_name=model_name,
                error=str(e),
                correlation_id=correlation_id
            )
            
            if fallback:
                logger.warning(
                    "Primary model exception, trying fallback",
                    primary_model=model_name,
                    error=str(e),
                    correlation_id=correlation_id
                )
                return await self._try_fallback_generation(
                    original_model=model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
            else:
                return ModelResult(
                    success=False,
                    model_used=model_name,
                    error=str(e)
                )

    async def _try_fallback_generation(
        self,
        original_model: str,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.7,
        **kwargs
    ) -> ModelResult:
        """
        Try fallback models when primary model fails.
        
        Args:
            original_model: The model that failed
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            ModelResult: Generation result from fallback model
        """
        correlation_id = get_correlation_id()
        
        # Get list of fallback models (exclude the failed one)
        fallback_models = [
            name for name in self.models.keys() 
            if name != original_model and self.models[name].status == ModelStatus.READY
        ]
        
        if not fallback_models:
            # Try to use any available model as fallback
            try:
                available_models = await self.ollama_client.list_models()
                fallback_models = [
                    model["name"] for model in available_models 
                    if model["name"] != original_model
                ]
            except Exception:
                fallback_models = []
        
        if not fallback_models:
            return ModelResult(
                success=False,
                model_used=original_model,
                error="No fallback models available"
            )
        
        # Try fallback models one by one
        for fallback_model in fallback_models[:3]:  # Limit to 3 attempts
            try:
                logger.debug(
                    "Trying fallback model",
                    original_model=original_model,
                    fallback_model=fallback_model,
                    correlation_id=correlation_id
                )
                await self._ensure_model_loaded(fallback_model)
                result = await self.ollama_client.generate(
                    model_name=fallback_model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                # Only update stats if result is a ModelResult
                if isinstance(result, ModelResult):
                    if result.success:
                        logger.info(
                            "Fallback generation successful",
                            original_model=original_model,
                            fallback_model=fallback_model,
                            correlation_id=correlation_id
                        )
                        if fallback_model in self.models:
                            self.models[fallback_model].update_stats(result, confidence=0.7)
                            result.cost = 0.0
                        return result
                    else:
                        logger.warning(
                            "Fallback model also failed",
                            fallback_model=fallback_model,
                            error=result.error,
                            correlation_id=correlation_id
                        )
                        continue
                else:
                    logger.warning(
                        "Fallback model returned non-ModelResult",
                        fallback_model=fallback_model,
                        result_type=str(type(result)),
                        correlation_id=correlation_id
                    )
                    continue
            except Exception as e:
                logger.warning(
                    "Fallback model exception",
                    fallback_model=fallback_model,
                    error=str(e),
                    correlation_id=correlation_id
                )
                continue
        # All fallbacks failed
        return ModelResult(
            success=False,
            model_used=original_model,
            error=f"All fallback attempts failed. Original model: {original_model}"
        )
    
    async def _ensure_model_loaded(self, model_name: str) -> None:
        """Ensure a model is loaded and ready for inference."""
        correlation_id = get_correlation_id()
        
        # Check if already loaded
        if model_name in self.loaded_models:
            return
        
        # Use lock to prevent concurrent loading of same model
        if model_name not in self._loading_locks:
            self._loading_locks[model_name] = asyncio.Lock()
        
        async with self._loading_locks[model_name]:
            # Double-check after acquiring lock
            if model_name in self.loaded_models:
                return
            
            logger.info(
                "Loading model",
                model_name=model_name,
                correlation_id=correlation_id
            )
            
            start_time = time.time()
            
            try:
                # Check if model exists in Ollama
                status = await self.ollama_client.check_model_status(model_name)
                
                if status == ModelStatus.UNKNOWN:
                    # Try to pull the model
                    logger.info(
                        "Model not found, attempting to pull",
                        model_name=model_name,
                        correlation_id=correlation_id
                    )
                    await self.ollama_client.pull_model(model_name)
                
                # Verify model is ready
                status = await self.ollama_client.check_model_status(model_name)
                if status != ModelStatus.READY:
                    raise OllamaException(f"Model {model_name} failed to load properly")
                
                # Model successfully loaded
                self.loaded_models.add(model_name)
                load_time = time.time() - start_time
                
                if model_name in self.models:
                    self.models[model_name].load_time = load_time
                    self.models[model_name].status = ModelStatus.READY
                
                logger.info(
                    "Model loaded successfully",
                    model_name=model_name,
                    load_time=round(load_time, 2),
                    correlation_id=correlation_id
                )
                
            except Exception as e:
                logger.error(
                    "Model loading failed",
                    model_name=model_name,
                    error=str(e),
                    correlation_id=correlation_id
                )
                
                if model_name in self.models:
                    self.models[model_name].status = ModelStatus.ERROR
                
                raise
    
    async def preload_models(self, model_names: List[str]) -> Dict[str, bool]:
        """
        Preload multiple models concurrently.
        
        Args:
            model_names: List of model names to preload
            
        Returns:
            Dict mapping model names to success status
        """
        correlation_id = get_correlation_id()
        
        logger.info(
            "Preloading models",
            model_names=model_names,
            correlation_id=correlation_id
        )
        
        results = {}
        
        # Load models concurrently
        tasks = []
        for model_name in model_names:
            task = asyncio.create_task(
                self._preload_single_model(model_name),
                name=f"preload_{model_name}"
            )
            tasks.append((model_name, task))
        
        # Wait for all tasks to complete
        for model_name, task in tasks:
            try:
                await task
                results[model_name] = True
            except Exception as e:
                logger.error(
                    "Model preload failed",
                    model_name=model_name,
                    error=str(e),
                    correlation_id=correlation_id
                )
                results[model_name] = False
        
        successful_loads = sum(results.values())
        
        logger.info(
            "Model preloading completed",
            total_models=len(model_names),
            successful_loads=successful_loads,
            failed_loads=len(model_names) - successful_loads,
            correlation_id=correlation_id
        )
        
        return results
    
    async def _preload_single_model(self, model_name: str) -> None:
        """Preload a single model with error handling."""
        try:
            await self._ensure_model_loaded(model_name)
        except Exception as e:
            logger.warning(
                "Single model preload failed",
                model_name=model_name,
                error=str(e),
                correlation_id=get_correlation_id()
            )
            raise
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        stats = {
            "total_models": len(self.models),
            "loaded_models": len(self.loaded_models),
            "model_details": {},
            "performance_summary": {
                "avg_response_time": 0.0,
                "avg_success_rate": 0.0,
                "total_requests": 0
            }
        }
        
        total_response_time = 0.0
        total_success_rate = 0.0
        total_requests = 0
        
        for model_name, model_info in self.models.items():
            stats["model_details"][model_name] = {
                "status": model_info.status.value,
                "tier": model_info.tier,
                "total_requests": model_info.total_requests,
                "avg_response_time": round(model_info.avg_response_time, 3),
                "avg_tokens_per_second": round(model_info.avg_tokens_per_second, 2),
                "success_rate": round(model_info.success_rate, 3),
                "avg_confidence": round(model_info.avg_confidence, 3),
                "performance_score": round(model_info.performance_score, 3),
                "last_used": model_info.last_used.isoformat(),
                "is_loaded": model_name in self.loaded_models
            }
            
            if model_info.total_requests > 0:
                total_response_time += model_info.avg_response_time
                total_success_rate += model_info.success_rate
                total_requests += model_info.total_requests
        
        # Calculate averages
        if self.models:
            stats["performance_summary"]["avg_response_time"] = round(
                total_response_time / len(self.models), 3
            )
            stats["performance_summary"]["avg_success_rate"] = round(
                total_success_rate / len(self.models), 3
            )
        
        stats["performance_summary"]["total_requests"] = total_requests
        
        return stats
    
    def get_model_recommendations(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get model optimization recommendations."""
        recommendations = {
            "performance_optimizations": [],
            "cost_optimizations": [],
            "reliability_improvements": []
        }
        
        for model_name, model_info in self.models.items():
            # Performance recommendations
            if model_info.avg_response_time > 10.0:
                recommendations["performance_optimizations"].append(
                    f"Model {model_name} has slow response time ({model_info.avg_response_time:.1f}s). "
                    f"Consider using a faster model for simple tasks."
                )
            
            # Reliability recommendations
            if model_info.success_rate < 0.9 and model_info.total_requests > 5:
                recommendations["reliability_improvements"].append(
                    f"Model {model_name} has low success rate ({model_info.success_rate:.1%}). "
                    f"Check model health or consider alternative."
                )
            
            # Cost optimizations (for future API model integration)
            if model_info.total_cost > 0:
                recommendations["cost_optimizations"].append(
                    f"Model {model_name} has incurred costs. Consider local alternatives."
                )
        
        return recommendations
    
    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        correlation_id = get_correlation_id()
        
        logger.info("Cleaning up ModelManager", correlation_id=correlation_id)
        
        try:
            await self.ollama_client.close()
            self.loaded_models.clear()
            self._loading_locks.clear()
            
            logger.info("ModelManager cleanup completed", correlation_id=correlation_id)
            
        except Exception as e:
            logger.error(
                "ModelManager cleanup failed",
                error=str(e),
                correlation_id=correlation_id
            )


# Export main classes and types
__all__ = [
    'ModelManager',
    'ModelInfo',
    'TaskType',
    'QualityLevel'
]
