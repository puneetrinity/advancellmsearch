# app/graphs/intelligent_router.py
"""
Enhanced Routing System with Pattern Recognition
Intelligent query routing based on learned patterns and optimization
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.graphs.base import GraphState, GraphType
from app.graphs.chat_graph import ChatGraph
from app.graphs.search_graph import SearchGraph
from app.models.manager import ModelManager
from app.cache.redis_client import CacheManager
from app.core.config import get_settings

logger = structlog.get_logger(__name__)


class IntentType(Enum):
    """Query intent types"""
    SIMPLE_CHAT = "simple_chat"
    SEARCH_NEEDED = "search_needed"
    CODE_ASSISTANCE = "code_assistance"
    ANALYSIS_NEEDED = "analysis_needed"
    RESEARCH_MODE = "research_mode"
    MULTILINGUAL = "multilingual"


@dataclass
class QueryPattern:
    """Represents a learned query pattern"""
    pattern_id: str
    intent: IntentType
    keywords: List[str]
    query_features: Dict[str, float]
    optimal_route: List[str]
    success_rate: float
    avg_execution_time: float
    avg_cost: float
    confidence_threshold: float
    usage_count: int
    last_used: datetime
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for caching"""
        data = asdict(self)
        data["intent"] = self.intent.value
        data["last_used"] = self.last_used.isoformat()
        data["created_at"] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryPattern':
        """Create from dictionary"""
        data["intent"] = IntentType(data["intent"])
        data["last_used"] = datetime.fromisoformat(data["last_used"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class RoutingDecision:
    """Represents a routing decision"""
    selected_graph: GraphType
    confidence: float
    reasoning: str
    estimated_cost: float
    estimated_time: float
    pattern_match: Optional[QueryPattern] = None
    fallback_route: Optional[GraphType] = None


class QueryFeatureExtractor:
    """Extracts features from queries for pattern matching"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.is_fitted = False
        
        # Feature keywords for different intents
        self.intent_keywords = {
            IntentType.SIMPLE_CHAT: [
                "hello", "hi", "how are you", "thanks", "thank you", "bye", "goodbye",
                "what is", "who is", "where is", "when is", "explain", "tell me"
            ],
            IntentType.SEARCH_NEEDED: [
                "latest", "recent", "current", "news", "today", "2024", "2025",
                "find", "search", "look up", "information about", "what happened"
            ],
            IntentType.CODE_ASSISTANCE: [
                "code", "programming", "function", "api", "debug", "error", "python",
                "javascript", "react", "sql", "algorithm", "implement", "how to code"
            ],
            IntentType.ANALYSIS_NEEDED: [
                "analyze", "compare", "evaluate", "research", "study", "investigate",
                "pros and cons", "advantages", "disadvantages", "deep dive"
            ],
            IntentType.RESEARCH_MODE: [
                "comprehensive", "detailed research", "thorough analysis", "in-depth",
                "academic", "scholarly", "literature review", "survey"
            ],
            IntentType.MULTILINGUAL: [
                "translate", "language", "español", "français", "deutsch", "中文",
                "japanese", "korean", "russian", "arabic", "hindi"
            ]
        }
    
    def extract_features(self, query: str, context: Optional[GraphState] = None) -> Dict[str, float]:
        """Extract features from a query"""
        query_lower = query.lower()
        features = {}
        
        # Basic text features
        features["length"] = len(query)
        features["word_count"] = len(query.split())
        features["question_marks"] = query.count("?")
        features["exclamation_marks"] = query.count("!")
        features["uppercase_ratio"] = sum(1 for c in query if c.isupper()) / max(len(query), 1)
        
        # Intent-based keyword features
        for intent, keywords in self.intent_keywords.items():
            feature_name = f"intent_{intent.value}_score"
            score = sum(1 for keyword in keywords if keyword in query_lower)
            features[feature_name] = score / max(len(keywords), 1)
        
        # Context features (if available)
        if context:
            features["has_conversation_history"] = len(context.conversation_history) > 0
            features["session_length"] = len(context.conversation_history)
            features["user_tier"] = self._encode_user_tier(context.user_preferences.get("tier", "free"))
            features["time_budget_ratio"] = context.max_execution_time / 30.0  # Normalize to 30s baseline
        else:
            features["has_conversation_history"] = 0
            features["session_length"] = 0
            features["user_tier"] = 0
            features["time_budget_ratio"] = 1.0
        
        # Temporal features
        now = datetime.now()
        features["hour_of_day"] = now.hour / 24.0
        features["day_of_week"] = now.weekday() / 7.0
        
        return features
    
    def _encode_user_tier(self, tier: str) -> float:
        """Encode user tier as numerical feature"""
        tier_mapping = {"free": 0.0, "pro": 0.5, "enterprise": 1.0}
        return tier_mapping.get(tier, 0.0)
    
    def calculate_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Calculate cosine similarity between feature vectors"""
        # Ensure both feature dicts have the same keys
        all_keys = set(features1.keys()) | set(features2.keys())
        
        vec1 = np.array([features1.get(key, 0.0) for key in all_keys])
        vec2 = np.array([features2.get(key, 0.0) for key in all_keys])
        
        # Calculate cosine similarity
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        
        return float(cosine_similarity([vec1], [vec2])[0][0])


class PatternLearningEngine:
    """Learns patterns from successful query executions"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.feature_extractor = QueryFeatureExtractor()
        self.patterns: Dict[str, QueryPattern] = {}
        self.pattern_cache_ttl = 86400 * 7  # 1 week
    
    async def initialize(self):
        """Initialize the pattern learning engine"""
        await self._load_patterns_from_cache()
        logger.info(f"Loaded {len(self.patterns)} query patterns")
    
    async def _load_patterns_from_cache(self):
        """Load existing patterns from cache"""
        try:
            patterns_data = await self.cache_manager.get("routing_patterns:all")
            if patterns_data:
                for pattern_data in patterns_data:
                    pattern = QueryPattern.from_dict(pattern_data)
                    self.patterns[pattern.pattern_id] = pattern
        except Exception as e:
            logger.warning(f"Failed to load patterns from cache: {e}")
    
    async def learn_from_execution(
        self, 
        query: str, 
        state: GraphState, 
        execution_result: Dict[str, Any]
    ):
        """Learn from a successful query execution"""
        try:
            # Only learn from successful executions
            if not execution_result.get("success", False):
                return
            
            # Extract features
            features = self.feature_extractor.extract_features(query, state)
            
            # Classify intent
            intent = self._classify_intent(query, features)
            
            # Create or update pattern
            pattern_id = self._generate_pattern_id(query, features)
            
            if pattern_id in self.patterns:
                # Update existing pattern
                await self._update_pattern(pattern_id, execution_result)
            else:
                # Create new pattern
                await self._create_pattern(
                    pattern_id, query, features, intent, execution_result
                )
            
            # Persist patterns to cache
            await self._save_patterns_to_cache()
            
        except Exception as e:
            logger.error(f"Pattern learning failed: {e}")
    
    def _classify_intent(self, query: str, features: Dict[str, float]) -> IntentType:
        """Classify query intent based on features"""
        intent_scores = {}
        
        for intent in IntentType:
            score_key = f"intent_{intent.value}_score"
            intent_scores[intent] = features.get(score_key, 0.0)
        
        # Add rule-based boosts
        query_lower = query.lower()
        
        # Search indicators
        if any(word in query_lower for word in ["latest", "current", "today", "news", "recent"]):
            intent_scores[IntentType.SEARCH_NEEDED] += 0.3
        
        # Code indicators
        if any(word in query_lower for word in ["code", "function", "api", "debug", "error"]):
            intent_scores[IntentType.CODE_ASSISTANCE] += 0.3
        
        # Analysis indicators
        if any(word in query_lower for word in ["analyze", "compare", "research", "study"]):
            intent_scores[IntentType.ANALYSIS_NEEDED] += 0.3
        
        # Simple chat indicators
        if any(word in query_lower for word in ["hello", "hi", "thanks", "what is"]):
            intent_scores[IntentType.SIMPLE_CHAT] += 0.2
        
        # Return intent with highest score
        return max(intent_scores.items(), key=lambda x: x[1])[0]
    
    def _generate_pattern_id(self, query: str, features: Dict[str, float]) -> str:
        """Generate a unique pattern ID"""
        # Create a hash based on query characteristics
        feature_str = json.dumps(sorted(features.items()), sort_keys=True)
        content = f"{query[:50]}{feature_str}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def _create_pattern(
        self, 
        pattern_id: str, 
        query: str, 
        features: Dict[str, float],
        intent: IntentType,
        execution_result: Dict[str, Any]
    ):
        """Create a new query pattern"""
        keywords = self._extract_keywords(query)
        
        pattern = QueryPattern(
            pattern_id=pattern_id,
            intent=intent,
            keywords=keywords,
            query_features=features,
            optimal_route=execution_result.get("execution_path", []),
            success_rate=1.0,
            avg_execution_time=execution_result.get("execution_time", 0.0),
            avg_cost=execution_result.get("total_cost", 0.0),
            confidence_threshold=0.7,
            usage_count=1,
            last_used=datetime.now(),
            created_at=datetime.now()
        )
        
        self.patterns[pattern_id] = pattern
        
        logger.info(
            "Created new query pattern",
            pattern_id=pattern_id,
            intent=intent.value,
            keywords=keywords[:3]
        )
    
    async def _update_pattern(self, pattern_id: str, execution_result: Dict[str, Any]):
        """Update an existing pattern with new execution data"""
        pattern = self.patterns[pattern_id]
        
        # Update averages using online learning
        alpha = 1.0 / (pattern.usage_count + 1)  # Learning rate
        
        new_time = execution_result.get("execution_time", 0.0)
        new_cost = execution_result.get("total_cost", 0.0)
        
        pattern.avg_execution_time = (1 - alpha) * pattern.avg_execution_time + alpha * new_time
        pattern.avg_cost = (1 - alpha) * pattern.avg_cost + alpha * new_cost
        pattern.usage_count += 1
        pattern.last_used = datetime.now()
        
        # Update success rate (assuming this execution was successful)
        pattern.success_rate = (1 - alpha) * pattern.success_rate + alpha * 1.0
        
        logger.debug(
            "Updated query pattern",
            pattern_id=pattern_id,
            usage_count=pattern.usage_count,
            success_rate=pattern.success_rate
        )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract key terms from query"""
        # Simple keyword extraction (could be enhanced with NLP)
        stop_words = {"the", "is", "at", "which", "on", "a", "an", "and", "or", "but", "in", "with", "to", "for", "of", "as", "by"}
        words = query.lower().split()
        keywords = [word.strip(".,!?") for word in words if word not in stop_words and len(word) > 2]
        return keywords[:10]  # Limit to 10 keywords
    
    async def _save_patterns_to_cache(self):
        """Save all patterns to cache"""
        try:
            patterns_data = [pattern.to_dict() for pattern in self.patterns.values()]
            await self.cache_manager.set(
                "routing_patterns:all", 
                patterns_data, 
                ttl=self.pattern_cache_ttl
            )
        except Exception as e:
            logger.error(f"Failed to save patterns to cache: {e}")
    
    async def find_matching_pattern(self, query: str, state: GraphState) -> Optional[QueryPattern]:
        """Find the best matching pattern for a query"""
        if not self.patterns:
            return None
        
        query_features = self.feature_extractor.extract_features(query, state)
        best_match = None
        best_similarity = 0.0
        
        for pattern in self.patterns.values():
            # Skip patterns with low success rates
            if pattern.success_rate < 0.6:
                continue
            
            # Calculate feature similarity
            similarity = self.feature_extractor.calculate_similarity(
                query_features, pattern.query_features
            )
            
            # Boost similarity for keyword matches
            keyword_matches = sum(1 for keyword in pattern.keywords if keyword in query.lower())
            keyword_boost = keyword_matches / max(len(pattern.keywords), 1) * 0.2
            
            total_similarity = similarity + keyword_boost
            
            if total_similarity > best_similarity and total_similarity >= pattern.confidence_threshold:
                best_similarity = total_similarity
                best_match = pattern
        
        if best_match:
            logger.info(
                "Found matching pattern",
                pattern_id=best_match.pattern_id,
                similarity=best_similarity,
                intent=best_match.intent.value
            )
        
        return best_match


class IntelligentRouter:
    """Intelligent routing system with pattern recognition"""
    
    def __init__(self, model_manager: ModelManager, cache_manager: CacheManager):
        self.model_manager = model_manager
        self.cache_manager = cache_manager
        self.pattern_engine = PatternLearningEngine(cache_manager)
        self.available_graphs = {}
        self.routing_stats = {
            "total_routes": 0,
            "pattern_matches": 0,
            "fallback_routes": 0
        }
    
    async def initialize(self):
        """Initialize the intelligent router"""
        await self.pattern_engine.initialize()
        
        # Initialize available graphs
        from app.graphs.chat_graph import create_chat_graph
        from app.graphs.search_graph import create_search_graph
        
        self.available_graphs[GraphType.CHAT] = await create_chat_graph(
            self.model_manager, self.cache_manager
        )
        self.available_graphs[GraphType.SEARCH] = await create_search_graph(
            self.model_manager, self.cache_manager
        )
        
        logger.info("Intelligent router initialized with pattern recognition")
    
    async def route_query(self, query: str, state: GraphState) -> RoutingDecision:
        """Make intelligent routing decision"""
        self.routing_stats["total_routes"] += 1
        
        try:
            # Try to find matching pattern first
            matching_pattern = await self.pattern_engine.find_matching_pattern(query, state)
            
            if matching_pattern:
                # Use pattern-based routing
                decision = await self._route_with_pattern(matching_pattern, state)
                self.routing_stats["pattern_matches"] += 1
            else:
                # Fallback to rule-based routing
                decision = await self._route_with_rules(query, state)
                self.routing_stats["fallback_routes"] += 1
            
            # Log routing decision
            logger.info(
                "Routing decision made",
                query_id=state.query_id,
                selected_graph=decision.selected_graph.value,
                confidence=decision.confidence,
                pattern_match=bool(matching_pattern)
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            # Emergency fallback to chat
            return RoutingDecision(
                selected_graph=GraphType.CHAT,
                confidence=0.3,
                reasoning="Emergency fallback due to routing error",
                estimated_cost=0.01,
                estimated_time=5.0
            )
    
    async def _route_with_pattern(self, pattern: QueryPattern, state: GraphState) -> RoutingDecision:
        """Route using a matched pattern"""
        # Determine graph type from pattern intent
        graph_mapping = {
            IntentType.SIMPLE_CHAT: GraphType.CHAT,
            IntentType.SEARCH_NEEDED: GraphType.SEARCH,
            IntentType.CODE_ASSISTANCE: GraphType.CHAT,  # Can be enhanced to code-specific graph
            IntentType.ANALYSIS_NEEDED: GraphType.SEARCH,  # Use search for analysis
            IntentType.RESEARCH_MODE: GraphType.SEARCH,
            IntentType.MULTILINGUAL: GraphType.CHAT
        }
        
        selected_graph = graph_mapping.get(pattern.intent, GraphType.CHAT)
        
        # Adjust estimates based on user context
        time_adjustment = self._calculate_time_adjustment(state)
        cost_adjustment = self._calculate_cost_adjustment(state)
        
        return RoutingDecision(
            selected_graph=selected_graph,
            confidence=pattern.success_rate,
            reasoning=f"Pattern match: {pattern.intent.value} (used {pattern.usage_count} times)",
            estimated_cost=pattern.avg_cost * cost_adjustment,
            estimated_time=pattern.avg_execution_time * time_adjustment,
            pattern_match=pattern,
            fallback_route=GraphType.CHAT if selected_graph != GraphType.CHAT else None
        )
    
    async def _route_with_rules(self, query: str, state: GraphState) -> RoutingDecision:
        """Fallback rule-based routing"""
        query_lower = query.lower()
        
        # Search indicators
        search_keywords = ["latest", "current", "recent", "news", "find", "search", "today", "what happened"]
        if any(keyword in query_lower for keyword in search_keywords):
            return RoutingDecision(
                selected_graph=GraphType.SEARCH,
                confidence=0.7,
                reasoning="Rule-based: Search keywords detected",
                estimated_cost=0.02,
                estimated_time=10.0,
                fallback_route=GraphType.CHAT
            )
        
        # Complex analysis indicators
        analysis_keywords = ["analyze", "compare", "research", "study", "comprehensive", "detailed"]
        if any(keyword in query_lower for keyword in analysis_keywords):
            return RoutingDecision(
                selected_graph=GraphType.SEARCH,
                confidence=0.6,
                reasoning="Rule-based: Analysis keywords detected",
                estimated_cost=0.03,
                estimated_time=15.0,
                fallback_route=GraphType.CHAT
            )
        
        # Default to chat for simple queries
        return RoutingDecision(
            selected_graph=GraphType.CHAT,
            confidence=0.8,
            reasoning="Rule-based: Default to chat for simple query",
            estimated_cost=0.005,
            estimated_time=3.0
        )
    
    def _calculate_time_adjustment(self, state: GraphState) -> float:
        """Calculate time adjustment based on context"""
        adjustment = 1.0
        
        # User tier adjustment
        user_tier = state.user_preferences.get("tier", "free")
        if user_tier == "enterprise":
            adjustment *= 0.8  # Faster processing for enterprise
        elif user_tier == "free":
            adjustment *= 1.2  # Slower for free tier
        
        # Quality requirement adjustment
        if state.quality_requirement == "premium":
            adjustment *= 1.5
        elif state.quality_requirement == "minimal":
            adjustment *= 0.7
        
        return adjustment
    
    def _calculate_cost_adjustment(self, state: GraphState) -> float:
        """Calculate cost adjustment based on context"""
        adjustment = 1.0
        
        # Quality requirement adjustment
        if state.quality_requirement == "premium":
            adjustment *= 2.0  # Higher cost for premium quality
        elif state.quality_requirement == "minimal":
            adjustment *= 0.5  # Lower cost for minimal quality
        
        # Budget constraint adjustment
        if state.cost_budget_remaining < 0.1:  # Less than ₹0.10 remaining
            adjustment *= 0.3  # Force cheaper execution
        
        return adjustment
    
    async def execute_with_route(self, decision: RoutingDecision, state: GraphState) -> GraphState:
        """Execute the query using the selected route"""
        try:
            selected_graph = self.available_graphs[decision.selected_graph]
            result_state = await selected_graph.execute(state)
            
            # Learn from successful execution
            if result_state.final_response:
                execution_result = {
                    "success": True,
                    "execution_path": result_state.execution_path,
                    "execution_time": result_state.calculate_total_time(),
                    "total_cost": result_state.calculate_total_cost(),
                    "confidence": result_state.get_avg_confidence()
                }
                
                await self.pattern_engine.learn_from_execution(
                    state.original_query, state, execution_result
                )
            
            return result_state
            
        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            
            # Try fallback route if available
            if decision.fallback_route and decision.fallback_route in self.available_graphs:
                logger.info(f"Attempting fallback to {decision.fallback_route.value}")
                try:
                    fallback_graph = self.available_graphs[decision.fallback_route]
                    return await fallback_graph.execute(state)
                except Exception as fallback_error:
                    logger.error(f"Fallback execution also failed: {fallback_error}")
            
            # Final fallback - return error state
            state.errors.append(f"Graph execution failed: {str(e)}")
            state.final_response = "I apologize, but I encountered an error processing your request."
            return state
    
    async def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        stats = self.routing_stats.copy()
        stats["pattern_count"] = len(self.pattern_engine.patterns)
        stats["pattern_match_rate"] = (
            stats["pattern_matches"] / max(stats["total_routes"], 1)
        )
        
        # Get top patterns by usage
        top_patterns = sorted(
            self.pattern_engine.patterns.values(),
            key=lambda p: p.usage_count,
            reverse=True
        )[:5]
        
        stats["top_patterns"] = [
            {
                "intent": pattern.intent.value,
                "usage_count": pattern.usage_count,
                "success_rate": pattern.success_rate,
                "avg_cost": pattern.avg_cost
            }
            for pattern in top_patterns
        ]
        
        return stats
    
    async def optimize_patterns(self):
        """Optimize patterns by removing low-performing ones"""
        before_count = len(self.pattern_engine.patterns)
        
        # Remove patterns with low success rates or old patterns with low usage
        cutoff_date = datetime.now() - timedelta(days=30)
        patterns_to_remove = []
        
        for pattern_id, pattern in self.pattern_engine.patterns.items():
            # Remove patterns with very low success rate
            if pattern.success_rate < 0.3:
                patterns_to_remove.append(pattern_id)
                continue
            
            # Remove old patterns with low usage
            if pattern.last_used < cutoff_date and pattern.usage_count < 5:
                patterns_to_remove.append(pattern_id)
                continue
        
        # Remove identified patterns
        for pattern_id in patterns_to_remove:
            del self.pattern_engine.patterns[pattern_id]
        
        # Save updated patterns
        await self.pattern_engine._save_patterns_to_cache()
        
        removed_count = len(patterns_to_remove)
        logger.info(
            "Pattern optimization completed",
            removed_patterns=removed_count,
            remaining_patterns=len(self.pattern_engine.patterns)
        )
        
        return {
            "before_count": before_count,
            "removed_count": removed_count,
            "after_count": len(self.pattern_engine.patterns)
        }


# Enhanced Graph Orchestrator
class GraphOrchestrator:
    """Master orchestrator with intelligent routing"""
    
    def __init__(self, model_manager: ModelManager, cache_manager: CacheManager):
        self.model_manager = model_manager
        self.cache_manager = cache_manager
        self.router = IntelligentRouter(model_manager, cache_manager)
        self.execution_history = []
    
    async def initialize(self):
        """Initialize the orchestrator"""
        await self.router.initialize()
        logger.info("Graph orchestrator initialized with intelligent routing")
    
    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a query through intelligent routing"""
        # Create graph state
        state = GraphState(
            original_query=query,
            user_id=context.get("user_id"),
            session_id=context.get("session_id"),
            conversation_history=context.get("conversation_history", []),
            user_preferences=context.get("user_preferences", {}),
            cost_budget_remaining=context.get("cost_budget_remaining", 20.0),
            max_execution_time=context.get("max_execution_time", 30.0),
            quality_requirement=context.get("quality_requirement", "balanced")
        )
        
        try:
            # Get routing decision
            routing_decision = await self.router.route_query(query, state)
            
            # Execute with selected route
            result_state = await self.router.execute_with_route(routing_decision, state)
            
            # Track execution
            execution_record = {
                "query_id": state.query_id,
                "query": query,
                "routing_decision": {
                    "selected_graph": routing_decision.selected_graph.value,
                    "confidence": routing_decision.confidence,
                    "reasoning": routing_decision.reasoning
                },
                "execution_result": {
                    "success": bool(result_state.final_response),
                    "execution_time": result_state.calculate_total_time(),
                    "total_cost": result_state.calculate_total_cost(),
                    "execution_path": result_state.execution_path
                },
                "timestamp": datetime.now().isoformat()
            }
            
            self.execution_history.append(execution_record)
            
            # Keep only recent history (last 100 executions)
            if len(self.execution_history) > 100:
                self.execution_history = self.execution_history[-100:]
            
            # Return comprehensive response
            return {
                "success": bool(result_state.final_response),
                "response": result_state.final_response,
                "query_id": state.query_id,
                "routing": {
                    "selected_graph": routing_decision.selected_graph.value,
                    "confidence": routing_decision.confidence,
                    "reasoning": routing_decision.reasoning,
                    "pattern_match": bool(routing_decision.pattern_match)
                },
                "execution": {
                    "path": result_state.execution_path,
                    "time": result_state.calculate_total_time(),
                    "cost": result_state.calculate_total_cost(),
                    "confidence": result_state.get_avg_confidence()
                },
                "metadata": result_state.response_metadata,
                "errors": result_state.errors,
                "warnings": result_state.warnings
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=e)
            return {
                "success": False,
                "response": "I apologize, but I encountered an error processing your request.",
                "query_id": state.query_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator statistics"""
        routing_stats = await self.router.get_routing_stats()
        
        # Calculate execution statistics
        successful_executions = sum(1 for record in self.execution_history if record["execution_result"]["success"])
        total_executions = len(self.execution_history)
        
        avg_execution_time = 0.0
        avg_cost = 0.0
        if total_executions > 0:
            avg_execution_time = sum(
                record["execution_result"]["execution_time"] for record in self.execution_history
            ) / total_executions
            
            avg_cost = sum(
                record["execution_result"]["total_cost"] for record in self.execution_history
            ) / total_executions
        
        # Graph usage statistics
        graph_usage = {}
        for record in self.execution_history:
            graph = record["routing_decision"]["selected_graph"]
            graph_usage[graph] = graph_usage.get(graph, 0) + 1
        
        return {
            "routing": routing_stats,
            "execution": {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "success_rate": successful_executions / max(total_executions, 1),
                "avg_execution_time": avg_execution_time,
                "avg_cost": avg_cost
            },
            "graph_usage": graph_usage,
            "performance": {
                "target_response_time": get_settings().target_response_time,
                "target_local_processing": get_settings().target_local_processing,
                "target_cache_hit_rate": get_settings().target_cache_hit_rate
            }
        }
    
    async def optimize_system(self) -> Dict[str, Any]:
        """Optimize the entire system based on learned patterns"""
        optimization_results = {}
        
        # Optimize routing patterns
        pattern_optimization = await self.router.optimize_patterns()
        optimization_results["patterns"] = pattern_optimization
        
        # Analyze execution patterns for model optimization
        model_usage = {}
        for record in self.execution_history[-50:]:  # Last 50 executions
            path = record["execution_result"]["execution_path"]
            for step in path:
                if step in ["phi:mini", "llama2:7b", "mistral:7b", "llama2:13b", "codellama"]:
                    model_usage[step] = model_usage.get(step, 0) + 1
        
        optimization_results["model_usage"] = model_usage
        
        # Cache optimization suggestions
        cache_stats = await self.cache_manager.get_metrics()
        optimization_results["cache"] = {
            "current_hit_rate": cache_stats.get("hit_rate", 0.0),
            "target_hit_rate": get_settings().target_cache_hit_rate,
            "suggestions": []
        }
        
        if cache_stats.get("hit_rate", 0.0) < get_settings().target_cache_hit_rate:
            optimization_results["cache"]["suggestions"].append(
                "Consider increasing cache TTL for frequently accessed patterns"
            )
        
        logger.info("System optimization completed", results=optimization_results)
        return optimization_results


# Helper function to create orchestrator
async def create_graph_orchestrator(
    model_manager: ModelManager,
    cache_manager: CacheManager
) -> GraphOrchestrator:
    """Create and initialize graph orchestrator"""
    orchestrator = GraphOrchestrator(model_manager, cache_manager)
    await orchestrator.initialize()
    return orchestrator
