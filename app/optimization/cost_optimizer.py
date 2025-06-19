# app/optimization/cost_optimizer.py
"""
Cost Optimization System with Intelligent Model Selection
Dynamic model selection and cost-aware execution strategies
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
import numpy as np

from app.models.manager import ModelManager, ModelResult
from app.cache.redis_client import CacheManager
from app.core.config import get_settings, MODEL_ASSIGNMENTS, API_COSTS

logger = structlog.get_logger(__name__)


class OptimizationStrategy(Enum):
    """Cost optimization strategies"""
    COST_FIRST = "cost_first"           # Minimize cost above all
    BALANCED = "balanced"               # Balance cost and quality  
    QUALITY_FIRST = "quality_first"     # Maximize quality within budget
    SPEED_FIRST = "speed_first"         # Minimize response time
    ADAPTIVE = "adaptive"               # Adapt based on user behavior


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a model"""
    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    total_execution_time: float = 0.0
    total_cost: float = 0.0
    avg_confidence: float = 0.0
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    cost_per_request: float = 0.0
    quality_score: float = 0.0
    last_updated: datetime = None
    
    def update_metrics(self, execution_time: float, cost: float, confidence: float, success: bool):
        """Update metrics with new execution data"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        
        self.total_execution_time += execution_time
        self.total_cost += cost
        
        # Update averages
        self.success_rate = self.successful_requests / self.total_requests
        self.avg_response_time = self.total_execution_time / self.total_requests
        self.cost_per_request = self.total_cost / self.total_requests
        
        # Update confidence using exponential moving average
        alpha = 0.1  # Learning rate
        if self.avg_confidence == 0.0:
            self.avg_confidence = confidence
        else:
            self.avg_confidence = (1 - alpha) * self.avg_confidence + alpha * confidence
        
        # Calculate quality score (combination of success rate and confidence)
        self.quality_score = (self.success_rate * 0.7 + self.avg_confidence * 0.3)
        self.last_updated = datetime.now()
    
    def get_efficiency_score(self, strategy: OptimizationStrategy) -> float:
        """Calculate efficiency score based on optimization strategy"""
        if strategy == OptimizationStrategy.COST_FIRST:
            # Lower cost is better
            return 1.0 / (self.cost_per_request + 0.001)
        elif strategy == OptimizationStrategy.QUALITY_FIRST:
            # Higher quality is better
            return self.quality_score
        elif strategy == OptimizationStrategy.SPEED_FIRST:
            # Lower response time is better
            return 1.0 / (self.avg_response_time + 0.001)
        elif strategy == OptimizationStrategy.BALANCED:
            # Balance quality, speed, and cost
            cost_score = 1.0 / (self.cost_per_request + 0.001)
            speed_score = 1.0 / (self.avg_response_time + 0.001)
            return (self.quality_score * 0.4 + cost_score * 0.3 + speed_score * 0.3)
        else:  # ADAPTIVE
            return self.quality_score * self.success_rate


@dataclass
class CostBudget:
    """User cost budget tracking"""
    user_id: str
    total_budget: float
    used_budget: float
    remaining_budget: float
    daily_limit: float
    used_today: float
    reset_date: datetime
    tier: str = "free"
    
    def can_afford(self, estimated_cost: float) -> bool:
        """Check if user can afford the estimated cost"""
        return (self.remaining_budget >= estimated_cost and 
                self.used_today + estimated_cost <= self.daily_limit)
    
    def deduct_cost(self, actual_cost: float):
        """Deduct actual cost from budget"""
        self.used_budget += actual_cost
        self.remaining_budget -= actual_cost
        self.used_today += actual_cost
    
    def should_use_cheaper_models(self) -> bool:
        """Determine if we should prefer cheaper models"""
        # Use cheaper models if budget is running low
        budget_ratio = self.remaining_budget / self.total_budget
        daily_ratio = self.used_today / self.daily_limit
        
        return budget_ratio < 0.2 or daily_ratio > 0.8


class ModelSelectionEngine:
    """Intelligent model selection based on cost optimization"""
    
    def __init__(self, model_manager: ModelManager, cache_manager: CacheManager):
        self.model_manager = model_manager
        self.cache_manager = cache_manager
        self.performance_metrics: Dict[str, ModelPerformanceMetrics] = {}
        self.settings = get_settings()
        
        # Initialize metrics for all known models
        for task_type, model_name in MODEL_ASSIGNMENTS.items():
            if model_name not in self.performance_metrics:
                self.performance_metrics[model_name] = ModelPerformanceMetrics(model_name)
    
    async def initialize(self):
        """Initialize the model selection engine"""
        await self._load_metrics_from_cache()
        logger.info("Model selection engine initialized")
    
    async def _load_metrics_from_cache(self):
        """Load performance metrics from cache"""
        try:
            cached_metrics = await self.cache_manager.get("model_performance_metrics")
            if cached_metrics:
                for model_name, metrics_data in cached_metrics.items():
                    if "last_updated" in metrics_data:
                        metrics_data["last_updated"] = datetime.fromisoformat(metrics_data["last_updated"])
                    self.performance_metrics[model_name] = ModelPerformanceMetrics(**metrics_data)
        except Exception as e:
            logger.warning(f"Failed to load model metrics from cache: {e}")
    
    async def _save_metrics_to_cache(self):
        """Save performance metrics to cache"""
        try:
            metrics_data = {}
            for model_name, metrics in self.performance_metrics.items():
                data = asdict(metrics)
                if data["last_updated"]:
                    data["last_updated"] = data["last_updated"].isoformat()
                metrics_data[model_name] = data
            
            await self.cache_manager.set("model_performance_metrics", metrics_data, ttl=86400)
        except Exception as e:
            logger.error(f"Failed to save model metrics to cache: {e}")
    
    async def select_optimal_model(
        self,
        task_type: str,
        quality_requirement: str,
        budget: CostBudget,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        context: Dict[str, Any] = None
    ) -> Tuple[str, float, str]:
        """
        Select optimal model based on requirements and budget
        Returns: (model_name, estimated_cost, selection_reason)
        """
        try:
            # Get candidate models for the task
            candidates = self._get_candidate_models(task_type, quality_requirement)
            
            # Filter by budget constraints
            affordable_candidates = self._filter_by_budget(candidates, budget)
            
            if not affordable_candidates:
                # Emergency fallback to cheapest local model
                return "phi:mini", 0.0, "Budget constraint - using cheapest model"
            
            # Score candidates based on strategy
            scored_candidates = []
            for model_name in affordable_candidates:
                metrics = self.performance_metrics.get(model_name)
                if metrics and metrics.total_requests > 0:
                    # Use historical performance
                    score = metrics.get_efficiency_score(strategy)
                    estimated_cost = metrics.cost_per_request
                else:
                    # Use default scoring for new models
                    score, estimated_cost = self._default_model_score(model_name, strategy)
                
                scored_candidates.append((model_name, score, estimated_cost))
            
            # Sort by score (higher is better)
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Select best candidate
            selected_model = scored_candidates[0][0]
            estimated_cost = scored_candidates[0][2]
            
            # Generate selection reasoning
            reasoning = self._generate_selection_reasoning(
                selected_model, task_type, strategy, budget, scored_candidates
            )
            
            logger.info(
                "Model selected",
                model=selected_model,
                task_type=task_type,
                strategy=strategy.value,
                estimated_cost=estimated_cost,
                reasoning=reasoning
            )
            
            return selected_model, estimated_cost, reasoning
            
        except Exception as e:
            logger.error(f"Model selection failed: {e}")
            # Safe fallback
            return "phi:mini", 0.0, f"Selection error - using fallback: {str(e)}"
    
    def _get_candidate_models(self, task_type: str, quality_requirement: str) -> List[str]:
        """Get candidate models for a task type and quality requirement"""
        # Base model for task type
        base_model = MODEL_ASSIGNMENTS.get(task_type, "llama2:7b")
        candidates = [base_model]
        
        # Add models based on quality requirement
        if quality_requirement == "minimal":
            candidates = ["phi:mini"]
        elif quality_requirement == "balanced":
            candidates = ["phi:mini", "llama2:7b"]
        elif quality_requirement == "high":
            if task_type == "code_tasks":
                candidates = ["codellama", "llama2:7b"]
            elif task_type == "analytical_reasoning":
                candidates = ["mistral:7b", "llama2:7b"]
            else:
                candidates = ["llama2:7b", "mistral:7b"]
        elif quality_requirement == "premium":
            if task_type in ["analytical_reasoning", "deep_research"]:
                candidates = ["llama2:13b", "mistral:7b", "llama2:7b"]
            elif task_type == "code_tasks":
                candidates = ["codellama", "llama2:13b"]
            else:
                candidates = ["llama2:13b", "mistral:7b", "llama2:7b"]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for model in candidates:
            if model not in seen:
                seen.add(model)
                unique_candidates.append(model)
        
        return unique_candidates
    
    def _filter_by_budget(self, candidates: List[str], budget: CostBudget) -> List[str]:
        """Filter candidates by budget constraints"""
        affordable = []
        
        for model_name in candidates:
            model_cost = API_COSTS.get(model_name, 0.0)
            
            # Local models are always affordable
            if model_cost == 0.0:
                affordable.append(model_name)
                continue
            
            # Check if user can afford API model
            if budget.can_afford(model_cost):
                # Additional check for budget conservation
                if budget.should_use_cheaper_models():
                    # Skip expensive models if budget is low
                    if model_cost <= 0.01:  # Only very cheap API models
                        affordable.append(model_name)
                else:
                    affordable.append(model_name)
        
        return affordable
    
    def _default_model_score(self, model_name: str, strategy: OptimizationStrategy) -> Tuple[float, float]:
        """Generate default score for models without historical data"""
        model_cost = API_COSTS.get(model_name, 0.0)
        
        # Model capability estimates (would be refined with actual data)
        capability_scores = {
            "phi:mini": {"quality": 0.6, "speed": 0.9, "cost": 1.0},
            "llama2:7b": {"quality": 0.8, "speed": 0.7, "cost": 1.0},
            "mistral:7b": {"quality": 0.85, "speed": 0.6, "cost": 1.0},
            "llama2:13b": {"quality": 0.9, "speed": 0.4, "cost": 1.0},
            "codellama": {"quality": 0.85, "speed": 0.6, "cost": 1.0},
            "gpt-4": {"quality": 0.95, "speed": 0.3, "cost": 0.2},
            "claude-haiku": {"quality": 0.8, "speed": 0.7, "cost": 0.6}
        }
        
        scores = capability_scores.get(model_name, {"quality": 0.5, "speed": 0.5, "cost": 0.5})
        
        if strategy == OptimizationStrategy.COST_FIRST:
            score = scores["cost"]
        elif strategy == OptimizationStrategy.QUALITY_FIRST:
            score = scores["quality"]
        elif strategy == OptimizationStrategy.SPEED_FIRST:
            score = scores["speed"]
        else:  # BALANCED or ADAPTIVE
            score = (scores["quality"] * 0.4 + scores["cost"] * 0.3 + scores["speed"] * 0.3)
        
        return score, model_cost
    
    def _generate_selection_reasoning(
        self,
        selected_model: str,
        task_type: str,
        strategy: OptimizationStrategy,
        budget: CostBudget,
        all_candidates: List[Tuple[str, float, float]]
    ) -> str:
        """Generate human-readable reasoning for model selection"""
        reasons = []
        
        # Strategy reasoning
        if strategy == OptimizationStrategy.COST_FIRST:
            reasons.append("optimized for minimum cost")
        elif strategy == OptimizationStrategy.QUALITY_FIRST:
            reasons.append("optimized for maximum quality")
        elif strategy == OptimizationStrategy.SPEED_FIRST:
            reasons.append("optimized for fastest response")
        else:
            reasons.append("balanced optimization")
        
        # Task-specific reasoning
        if task_type == "code_tasks" and selected_model == "codellama":
            reasons.append("specialized for programming tasks")
        elif task_type == "analytical_reasoning" and selected_model in ["mistral:7b", "llama2:13b"]:
            reasons.append("optimized for analytical reasoning")
        
        # Budget reasoning
        if budget.should_use_cheaper_models():
            reasons.append("budget conservation mode")
        
        # Performance reasoning
        metrics = self.performance_metrics.get(selected_model)
        if metrics and metrics.total_requests > 10:
            if metrics.success_rate > 0.9:
                reasons.append(f"high success rate ({metrics.success_rate:.1%})")
            if metrics.avg_response_time < 2.0:
                reasons.append("fast response time")
        
        return "; ".join(reasons)
    
    async def record_execution(
        self,
        model_name: str,
        execution_time: float,
        cost: float,
        confidence: float,
        success: bool
    ):
        """Record execution metrics for model optimization"""
        if model_name not in self.performance_metrics:
            self.performance_metrics[model_name] = ModelPerformanceMetrics(model_name)
        
        self.performance_metrics[model_name].update_metrics(
            execution_time, cost, confidence, success
        )
        
        # Periodically save metrics to cache
        if self.performance_metrics[model_name].total_requests % 10 == 0:
            await self._save_metrics_to_cache()
    
    async def get_model_recommendations(self, budget: CostBudget) -> Dict[str, Any]:
        """Get model usage recommendations based on current performance"""
        recommendations = {
            "cost_savings": [],
            "performance_improvements": [],
            "budget_warnings": []
        }
        
        # Analyze current model performance
        total_cost = sum(metrics.total_cost for metrics in self.performance_metrics.values())
        total_requests = sum(metrics.total_requests for metrics in self.performance_metrics.values())
        
        if total_requests > 0:
            avg_cost_per_request = total_cost / total_requests
            
            # Cost saving recommendations
            if avg_cost_per_request > 0.005:  # Above ₹0.005 per request
                recommendations["cost_savings"].append(
                    "Consider using local models (phi:mini, llama2:7b) for simple tasks"
                )
            
            # Find underperforming expensive models
            for model_name, metrics in self.performance_metrics.items():
                if (metrics.cost_per_request > 0.01 and 
                    metrics.success_rate < 0.8 and 
                    metrics.total_requests > 5):
                    recommendations["performance_improvements"].append(
                        f"Model {model_name} has low success rate ({metrics.success_rate:.1%}) for its cost"
                    )
        
        # Budget warnings
        if budget.remaining_budget < budget.total_budget * 0.1:
            recommendations["budget_warnings"].append(
                "Budget running low - consider using only local models"
            )
        
        if budget.used_today > budget.daily_limit * 0.8:
            recommendations["budget_warnings"].append(
                "Daily spending limit nearly reached"
            )
        
        return recommendations
    
    async def optimize_model_loading(self) -> Dict[str, Any]:
        """Optimize which models should be kept loaded"""
        optimization_result = {
            "keep_loaded": [],
            "can_unload": [],
            "should_preload": []
        }
        
        # Analyze model usage patterns
        usage_scores = {}
        for model_name, metrics in self.performance_metrics.items():
            if metrics.total_requests > 0:
                # Score based on usage frequency and success rate
                recent_usage = 1.0 if metrics.last_updated and (
                    datetime.now() - metrics.last_updated
                ).total_seconds() < 3600 else 0.0
                
                usage_score = (
                    metrics.total_requests * 0.4 +
                    metrics.success_rate * 0.3 +
                    recent_usage * 0.3
                )
                usage_scores[model_name] = usage_score
        
        # Sort by usage score
        sorted_models = sorted(usage_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Categorize models
        for i, (model_name, score) in enumerate(sorted_models):
            if i < 2:  # Top 2 models
                optimization_result["keep_loaded"].append(model_name)
            elif score < 0.1:  # Low usage models
                optimization_result["can_unload"].append(model_name)
            elif score > 0.5:  # Medium-high usage
                optimization_result["should_preload"].append(model_name)
        
        return optimization_result


class CostOptimizer:
    """Main cost optimization system"""
    
    def __init__(self, model_manager: ModelManager, cache_manager: CacheManager):
        self.model_manager = model_manager
        self.cache_manager = cache_manager
        self.model_selector = ModelSelectionEngine(model_manager, cache_manager)
        self.user_budgets: Dict[str, CostBudget] = {}
        self.optimization_stats = {
            "total_savings": 0.0,
            "optimized_requests": 0,
            "fallback_usage": 0
        }
    
    async def initialize(self):
        """Initialize the cost optimization system"""
        await self.model_selector.initialize()
        await self._load_user_budgets()
        logger.info("Cost optimization system initialized")
    
    async def _load_user_budgets(self):
        """Load user budgets from cache"""
        try:
            cached_budgets = await self.cache_manager.get("user_budgets")
            if cached_budgets:
                for user_id, budget_data in cached_budgets.items():
                    if "reset_date" in budget_data:
                        budget_data["reset_date"] = datetime.fromisoformat(budget_data["reset_date"])
                    self.user_budgets[user_id] = CostBudget(**budget_data)
        except Exception as e:
            logger.warning(f"Failed to load user budgets: {e}")
    
    async def _save_user_budgets(self):
        """Save user budgets to cache"""
        try:
            budget_data = {}
            for user_id, budget in self.user_budgets.items():
                data = asdict(budget)
                data["reset_date"] = data["reset_date"].isoformat()
                budget_data[user_id] = data
            
            await self.cache_manager.set("user_budgets", budget_data, ttl=86400)
        except Exception as e:
            logger.error(f"Failed to save user budgets: {e}")
    
    async def get_user_budget(self, user_id: str, tier: str = "free") -> CostBudget:
        """Get or create user budget"""
        if user_id not in self.user_budgets:
            # Create new budget based on tier
            budget_limits = {
                "free": {"total": 20.0, "daily": 2.0},
                "pro": {"total": 500.0, "daily": 25.0},
                "enterprise": {"total": 5000.0, "daily": 200.0}
            }
            
            limits = budget_limits.get(tier, budget_limits["free"])
            
            self.user_budgets[user_id] = CostBudget(
                user_id=user_id,
                total_budget=limits["total"],
                used_budget=0.0,
                remaining_budget=limits["total"],
                daily_limit=limits["daily"],
                used_today=0.0,
                reset_date=datetime.now() + timedelta(days=30),
                tier=tier
            )
            
            await self._save_user_budgets()
        
        return self.user_budgets[user_id]
    
    async def optimize_request(
        self,
        user_id: str,
        task_type: str,
        quality_requirement: str,
        user_tier: str = "free",
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Optimize a request for cost efficiency"""
        
        # Get user budget
        budget = await self.get_user_budget(user_id, user_tier)
        
        # Determine optimization strategy based on user behavior and budget
        strategy = self._determine_strategy(budget, user_tier, context)
        
        # Select optimal model
        selected_model, estimated_cost, reasoning = await self.model_selector.select_optimal_model(
            task_type, quality_requirement, budget, strategy, context
        )
        
        # Check if we can afford the request
        if not budget.can_afford(estimated_cost):
            return {
                "allowed": False,
                "reason": "insufficient_budget",
                "remaining_budget": budget.remaining_budget,
                "estimated_cost": estimated_cost,
                "suggestions": [
                    "Upgrade to a higher tier for more budget",
                    "Wait for budget reset",
                    "Use simpler quality requirements"
                ]
            }
        
        # Provide cost-saving suggestions
        suggestions = await self._generate_cost_suggestions(
            selected_model, task_type, quality_requirement, budget
        )
        
        self.optimization_stats["optimized_requests"] += 1
        
        return {
            "allowed": True,
            "selected_model": selected_model,
            "estimated_cost": estimated_cost,
            "reasoning": reasoning,
            "strategy": strategy.value,
            "budget_remaining": budget.remaining_budget,
            "suggestions": suggestions,
            "optimization_metadata": {
                "strategy_used": strategy.value,
                "budget_status": "low" if budget.should_use_cheaper_models() else "normal",
                "tier": user_tier
            }
        }
    
    def _determine_strategy(
        self, 
        budget: CostBudget, 
        user_tier: str, 
        context: Dict[str, Any]
    ) -> OptimizationStrategy:
        """Determine the best optimization strategy for the user"""
        
        # Budget-driven strategy
        if budget.should_use_cheaper_models():
            return OptimizationStrategy.COST_FIRST
        
        # Tier-based strategy
        if user_tier == "enterprise":
            return OptimizationStrategy.QUALITY_FIRST
        elif user_tier == "pro":
            return OptimizationStrategy.BALANCED
        
        # Context-driven strategy
        if context:
            if context.get("time_critical", False):
                return OptimizationStrategy.SPEED_FIRST
            if context.get("quality_critical", False):
                return OptimizationStrategy.QUALITY_FIRST
        
        # Default to balanced
        return OptimizationStrategy.BALANCED
    
    async def _generate_cost_suggestions(
        self,
        selected_model: str,
        task_type: str,
        quality_requirement: str,
        budget: CostBudget
    ) -> List[str]:
        """Generate cost-saving suggestions"""
        suggestions = []
        
        # Model-specific suggestions
        if selected_model in ["gpt-4", "claude-haiku"]:
            suggestions.append("Consider using local models for similar tasks to save costs")
        
        # Quality-based suggestions
        if quality_requirement == "premium":
            suggestions.append("Try 'high' quality setting for significant cost savings")
        elif quality_requirement == "high" and task_type == "simple_classification":
            suggestions.append("Simple questions can often use 'balanced' quality")
        
        # Budget-based suggestions
        if budget.remaining_budget < budget.total_budget * 0.2:
            suggestions.append("Budget running low - consider upgrading tier or using local models only")
        
        # Task-specific suggestions
        if task_type == "simple_classification" and selected_model != "phi:mini":
            suggestions.append("Use phi:mini model for simple classification tasks")
        
        return suggestions
    
    async def record_execution_cost(
        self,
        user_id: str,
        model_name: str,
        actual_cost: float,
        execution_time: float,
        confidence: float,
        success: bool
    ):
        """Record actual execution cost and update metrics"""
        
        # Update user budget
        if user_id in self.user_budgets:
            budget = self.user_budgets[user_id]
            budget.deduct_cost(actual_cost)
            await self._save_user_budgets()
        
        # Record model performance
        await self.model_selector.record_execution(
            model_name, execution_time, actual_cost, confidence, success
        )
        
        # Update optimization stats
        if actual_cost == 0.0:  # Local model usage
            estimated_api_cost = 0.01  # What an API call might have cost
            self.optimization_stats["total_savings"] += estimated_api_cost
        
        logger.debug(
            "Execution cost recorded",
            user_id=user_id,
            model=model_name,
            cost=actual_cost,
            success=success
        )
    
    async def analyze_cost_efficiency(self) -> Dict[str, Any]:
        """Analyze overall cost efficiency and provide insights"""
        
        # Calculate total costs and savings
        total_model_costs = sum(
            metrics.total_cost for metrics in self.model_selector.performance_metrics.values()
        )
        
        total_requests = sum(
            metrics.total_requests for metrics in self.model_selector.performance_metrics.values()
        )
        
        avg_cost_per_request = total_model_costs / max(total_requests, 1)
        
        # Calculate local vs API usage
        local_requests = 0
        api_requests = 0
        
        for model_name, metrics in self.model_selector.performance_metrics.items():
            if API_COSTS.get(model_name, 0.0) == 0.0:
                local_requests += metrics.total_requests
            else:
                api_requests += metrics.total_requests
        
        local_percentage = (local_requests / max(total_requests, 1)) * 100
        
        # User budget analysis
        budget_utilization = {}
        for user_id, budget in self.user_budgets.items():
            utilization = (budget.used_budget / budget.total_budget) * 100
            budget_utilization[budget.tier] = budget_utilization.get(budget.tier, [])
            budget_utilization[budget.tier].append(utilization)
        
        # Calculate average utilization by tier
        avg_utilization_by_tier = {}
        for tier, utilizations in budget_utilization.items():
            avg_utilization_by_tier[tier] = sum(utilizations) / len(utilizations)
        
        return {
            "cost_metrics": {
                "total_costs": total_model_costs,
                "total_requests": total_requests,
                "avg_cost_per_request": avg_cost_per_request,
                "total_savings": self.optimization_stats["total_savings"],
                "target_cost_per_query": self.settings.cost_per_api_call
            },
            "usage_distribution": {
                "local_requests": local_requests,
                "api_requests": api_requests,
                "local_percentage": local_percentage,
                "target_local_percentage": self.settings.target_local_processing * 100
            },
            "budget_analysis": {
                "total_users": len(self.user_budgets),
                "avg_utilization_by_tier": avg_utilization_by_tier,
                "users_over_budget": sum(1 for b in self.user_budgets.values() if b.remaining_budget <= 0)
            },
            "optimization_stats": self.optimization_stats,
            "recommendations": await self._generate_system_recommendations()
        }
    
    async def _generate_system_recommendations(self) -> List[str]:
        """Generate system-wide optimization recommendations"""
        recommendations = []
        
        # Analyze current performance
        total_requests = sum(
            metrics.total_requests for metrics in self.model_selector.performance_metrics.values()
        )
        
        if total_requests == 0:
            return ["System needs more usage data for meaningful recommendations"]
        
        # Calculate local usage percentage
        local_requests = sum(
            metrics.total_requests for model_name, metrics in self.model_selector.performance_metrics.items()
            if API_COSTS.get(model_name, 0.0) == 0.0
        )
        
        local_percentage = (local_requests / total_requests) * 100
        target_percentage = self.settings.target_local_processing * 100
        
        if local_percentage < target_percentage:
            recommendations.append(
                f"Increase local model usage from {local_percentage:.1f}% to {target_percentage:.1f}%"
            )
        
        # Check for underperforming expensive models
        expensive_models = [
            model_name for model_name, metrics in self.model_selector.performance_metrics.items()
            if API_COSTS.get(model_name, 0.0) > 0.01 and metrics.success_rate < 0.8
        ]
        
        if expensive_models:
            recommendations.append(
                f"Review usage of expensive models with low success rates: {', '.join(expensive_models)}"
            )
        
        # Budget optimization recommendations
        low_budget_users = sum(1 for b in self.user_budgets.values() if b.remaining_budget < b.total_budget * 0.1)
        if low_budget_users > 0:
            recommendations.append(
                f"{low_budget_users} users have low remaining budgets - consider tier upgrades or cost optimizations"
            )
        
        return recommendations
    
    async def optimize_user_tier(self, user_id: str) -> Dict[str, Any]:
        """Recommend optimal tier for a user based on usage patterns"""
        if user_id not in self.user_budgets:
            return {"recommendation": "insufficient_data"}
        
        budget = self.user_budgets[user_id]
        current_tier = budget.tier
        
        # Calculate usage patterns
        monthly_usage = budget.used_budget
        daily_avg = budget.used_today  # Simplified - would use historical data
        
        # Tier recommendations based on usage
        tier_limits = {
            "free": {"monthly": 20.0, "daily": 2.0, "cost": 0.0},
            "pro": {"monthly": 500.0, "daily": 25.0, "cost": 49.0},
            "enterprise": {"monthly": 5000.0, "daily": 200.0, "cost": 199.0}
        }
        
        # Find optimal tier
        recommended_tier = current_tier
        cost_savings = 0.0
        
        for tier, limits in tier_limits.items():
            if monthly_usage <= limits["monthly"] * 0.8:  # 80% utilization threshold
                if tier != current_tier:
                    # Calculate potential savings/costs
                    current_cost = tier_limits[current_tier]["cost"]
                    new_cost = limits["cost"]
                    cost_difference = new_cost - current_cost
                    
                    if cost_difference < 0:  # Downgrade saves money
                        recommended_tier = tier
                        cost_savings = abs(cost_difference)
                        break
                    elif tier == "pro" and current_tier == "free" and monthly_usage > 15.0:
                        # Upgrade to pro might be worth it for heavy free users
                        recommended_tier = tier
                        break
        
        return {
            "current_tier": current_tier,
            "recommended_tier": recommended_tier,
            "monthly_usage": monthly_usage,
            "utilization_percentage": (monthly_usage / tier_limits[current_tier]["monthly"]) * 100,
            "potential_savings": cost_savings,
            "reason": "cost_optimization" if cost_savings > 0 else "usage_optimization"
        }
    
    async def get_cost_optimization_report(self, user_id: str = None) -> Dict[str, Any]:
        """Generate comprehensive cost optimization report"""
        if user_id:
            # User-specific report
            budget = await self.get_user_budget(user_id)
            recommendations = await self.model_selector.get_model_recommendations(budget)
            tier_optimization = await self.optimize_user_tier(user_id)
            
            return {
                "user_id": user_id,
                "budget_status": {
                    "remaining": budget.remaining_budget,
                    "used_percentage": (budget.used_budget / budget.total_budget) * 100,
                    "tier": budget.tier
                },
                "recommendations": recommendations,
                "tier_optimization": tier_optimization,
                "cost_saving_tips": await self._generate_user_cost_tips(user_id)
            }
        else:
            # System-wide report
            return await self.analyze_cost_efficiency()
    
    async def _generate_user_cost_tips(self, user_id: str) -> List[str]:
        """Generate personalized cost-saving tips for a user"""
        tips = []
        
        if user_id not in self.user_budgets:
            return ["No usage data available for personalized tips"]
        
        budget = self.user_budgets[user_id]
        
        # Budget-based tips
        if budget.should_use_cheaper_models():
            tips.append("Your budget is running low - stick to local models (phi:mini, llama2:7b)")
        
        if budget.tier == "free" and budget.used_budget > 15.0:
            tips.append("Consider upgrading to Pro tier for better value at your usage level")
        
        # Usage pattern tips
        tips.append("Use 'balanced' quality setting for most tasks instead of 'high' or 'premium'")
        tips.append("Cache frequently asked questions to avoid repeated processing costs")
        
        return tips


# Helper function to create cost optimizer
async def create_cost_optimizer(
    model_manager: ModelManager,
    cache_manager: CacheManager
) -> CostOptimizer:
    """Create and initialize cost optimizer"""
    optimizer = CostOptimizer(model_manager, cache_manager)
    await optimizer.initialize()
    return optimizer
