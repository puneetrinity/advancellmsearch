"""
ChatGraph - Intelligent conversation management with context awareness.
Implements sophisticated chat workflows with model selection and optimization.

Complete fixed version addressing LangGraph START/END constants and compilation issues.
"""
import re
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from app.graphs.base import (
    BaseGraph, BaseGraphNode, GraphState, NodeResult, 
    GraphType, NodeType, START, EndNode
)
from app.models.manager import ModelManager, TaskType, QualityLevel
from app.models.ollama_client import ModelResult
from app.core.logging import get_logger

logger = get_logger("graphs.chat")

@dataclass
class ConversationContext:
    """Rich conversation context for better response generation."""
    user_name: Optional[str] = None
    conversation_topic: Optional[str] = None
    user_expertise_level: str = "intermediate"  # beginner, intermediate, expert
    preferred_response_style: str = "balanced"  # concise, balanced, detailed
    conversation_mood: str = "neutral"  # casual, professional, neutral
    key_entities: List[str] = None
    previous_topics: List[str] = None
    
    def __post_init__(self):
        if self.key_entities is None:
            self.key_entities = []
        if self.previous_topics is None:
            self.previous_topics = []

class ContextManagerNode(BaseGraphNode):
    """
    Manages conversation context and history.
    Extracts relevant information from previous messages.
    """
    
    def __init__(self, cache_manager=None):
        super().__init__("context_manager", NodeType.PROCESSING)
        self.cache_manager = cache_manager
    
    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        logger.debug(f"[ContextManagerNode] Enter execute. state.query_id={getattr(state, 'query_id', None)}")
        try:
            # Create conversation context
            context = ConversationContext()
            
            # Analyze conversation history if available
            if state.conversation_history:
                # Extract patterns and preferences
                context.user_expertise_level = self._infer_expertise_level(state.conversation_history)
                context.preferred_response_style = self._infer_response_style(state.conversation_history)
                context.conversation_mood = self._infer_conversation_mood(state.conversation_history)
            
            # Store processed query
            state.processed_query = state.original_query
            
            # Store context in state
            state.intermediate_results["conversation_context"] = context.__dict__
            
            logger.debug(f"[ContextManagerNode] Success. state.query_id={getattr(state, 'query_id', None)}")
            return NodeResult(
                success=True,
                data={"context": context.__dict__},
                confidence=0.8,
                execution_time=0.1
            )
            
        except Exception as e:
            logger.error(f"[ContextManagerNode] Error: {e}")
            return NodeResult(
                success=False,
                error=f"Context management failed: {str(e)}",
                execution_time=0.1
            )
    
    def _infer_expertise_level(self, history: List[Dict]) -> str:
        """Infer user expertise level from conversation history."""
        # Simple heuristic - count technical terms
        technical_terms = 0
        total_words = 0
        
        for msg in history[-5:]:  # Last 5 messages
            if msg.get("role") == "user":
                content = msg.get("content", "").lower()
                words = content.split()
                total_words += len(words)
                
                # Count technical indicators
                for word in words:
                    if len(word) > 8 or word in ["algorithm", "implementation", "architecture"]:
                        technical_terms += 1
        
        if total_words == 0:
            return "intermediate"
        
        ratio = technical_terms / total_words
        if ratio > 0.1:
            return "expert"
        elif ratio > 0.05:
            return "intermediate"
        else:
            return "beginner"
    
    def _infer_response_style(self, history: List[Dict]) -> str:
        """Infer preferred response style from conversation history."""
        # Default to balanced
        return "balanced"
    
    def _infer_conversation_mood(self, history: List[Dict]) -> str:
        """Infer conversation mood from recent messages."""
        # Default to neutral
        return "neutral"

class IntentClassifierNode(BaseGraphNode):
    """
    Classifies user intent and determines optimal processing path.
    """
    
    def __init__(self, model_manager: ModelManager):
        super().__init__("intent_classifier", NodeType.PROCESSING)
        self.model_manager = model_manager
    
    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        logger.debug(f"[IntentClassifierNode] Enter execute. state.query_id={getattr(state, 'query_id', None)}")
        try:
            query = state.processed_query or state.original_query
            model_name = self.model_manager.select_optimal_model(
                TaskType.SIMPLE_CLASSIFICATION, 
                QualityLevel.MINIMAL
            )
            try:
                classification_prompt = f"Classify this query intent: '{query}'\nReturn only one word: question, creative, analysis, code, request, or conversation"
                result = await self.model_manager.generate(
                    model_name=model_name,
                    prompt=classification_prompt,
                    max_tokens=10,
                    temperature=0.1
                )
                if result.success:
                    intent = result.text.strip().lower()
                    if intent in ["question", "creative", "analysis", "code", "request", "conversation"]:
                        classification_method = "model_based"
                    else:
                        intent = self._classify_intent_rule_based(query)
                        classification_method = "rule_based_fallback"
                else:
                    intent = self._classify_intent_rule_based(query)
                    classification_method = "rule_based"
            except Exception:
                intent = self._classify_intent_rule_based(query)
                classification_method = "rule_based"
            complexity = self._calculate_complexity(query)
            state.query_intent = intent
            state.query_complexity = complexity
            logger.debug(f"[IntentClassifierNode] Success. state.query_id={getattr(state, 'query_id', None)}")
            return NodeResult(
                success=True,
                data={
                    "intent": intent,
                    "complexity": complexity,
                    "classification_method": classification_method
                },
                confidence=0.7,
                execution_time=0.05
            )
        except Exception as e:
            logger.error(f"[IntentClassifierNode] Error: {e}")
            return NodeResult(
                success=False,
                error=f"Intent classification failed: {str(e)}",
                execution_time=0.05
            )
    
    def _classify_intent_rule_based(self, query: str) -> str:
        query_lower = query.lower()
        code_terms = ["python", "function", "debug", "code", "script", "programming"]
        code_matches = sum(1 for term in code_terms if term in query_lower)
        if code_matches >= 2:
            return "code"
        elif any(term in query_lower for term in ["debug this python", "python function", "function code"]):
            return "code"
        elif any(word in query_lower for word in ["how", "what", "why", "when", "where"]):
            return "question"
        elif any(word in query_lower for word in ["create", "generate", "write", "make"]):
            return "creative"
        elif any(word in query_lower for word in ["analyze", "compare", "evaluate"]):
            return "analysis"
        elif any(word in query_lower for word in ["help", "can you", "please"]):
            return "request"
        else:
            return "conversation"
    
    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score (0.0 to 1.0)."""
        words = query.split()
        word_count = len(words)
        base_complexity = min(word_count / 50, 0.7)
        complex_indicators = ["analyze", "compare", "comprehensive", "detailed", "research"]
        if any(indicator in query.lower() for indicator in complex_indicators):
            base_complexity += 0.2
        return min(base_complexity, 1.0)

class ResponseGeneratorNode(BaseGraphNode):
    """
    Generates responses using the optimal model based on context and intent.
    """
    
    def __init__(self, model_manager: ModelManager):
        super().__init__("response_generator", NodeType.PROCESSING)
        self.model_manager = model_manager
    
    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        logger.debug(f"[ResponseGeneratorNode] Enter execute. state.query_id={getattr(state, 'query_id', None)}")
        try:
            model_name = self._select_model(state)
            prompt = self._build_prompt(state)
            max_tokens = self._calculate_max_tokens(state)
            temperature = self._calculate_temperature(state)
            try:
                result = await self.model_manager.generate(
                    model_name=model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            except TypeError as e:
                if "object ModelResult can't be used in 'await' expression" in str(e):
                    result = self.model_manager.generate(
                        model_name=model_name,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                else:
                    raise e
            if result.success:
                response = self._post_process_response(result.text, state)
                state.final_response = response
                logger.debug(f"[ResponseGeneratorNode] Success. state.query_id={getattr(state, 'query_id', None)}")
                return NodeResult(
                    success=True,
                    data={"response": response},
                    confidence=0.8,
                    execution_time=result.execution_time,
                    cost=result.cost,
                    model_used=model_name
                )
            else:
                fallback_response = "I'm having trouble generating a response right now."
                state.final_response = fallback_response
                logger.warning(f"[ResponseGeneratorNode] Model generation failed: {result.error}")
                return NodeResult(
                    success=False,
                    data={"response": fallback_response},
                    error=f"Model generation failed: {result.error}",
                    execution_time=getattr(result, 'execution_time', 1.0),
                    cost=getattr(result, 'cost', 0.0)
                )
        except Exception as e:
            fallback_response = "I encountered an error. Please try again."
            state.final_response = fallback_response
            logger.error(f"[ResponseGeneratorNode] Error: {e}")
            return NodeResult(
                success=False,
                data={"response": fallback_response},
                error=f"Response generation failed: {str(e)}",
                execution_time=1.0
            )
    
    def _select_model(self, state: GraphState) -> str:
        """Select optimal model based on intent and complexity."""
        intent = getattr(state, 'query_intent', 'conversation')
        complexity = getattr(state, 'query_complexity', 0.5)
        quality = getattr(state, 'quality_requirement', None)
        
        # Map intent to task type
        intent_to_task = {
            'code': TaskType.CODE_TASKS,
            'analysis': TaskType.ANALYTICAL_REASONING,
            'creative': TaskType.CREATIVE_WRITING,
            'question': TaskType.QA_AND_SUMMARY,
            'request': TaskType.QA_AND_SUMMARY,
            'conversation': TaskType.CONVERSATION
        }
        
        task_type = intent_to_task.get(intent, TaskType.CONVERSATION)
        
        # Map complexity and quality to quality level
        if quality == "premium" or complexity > 0.8:
            quality_level = QualityLevel.PREMIUM
        elif quality == "minimal" or complexity < 0.3:
            quality_level = QualityLevel.MINIMAL
        else:
            quality_level = QualityLevel.BALANCED
        
        return self.model_manager.select_optimal_model(task_type, quality_level)

    def _build_prompt(self, state: GraphState) -> str:
        """Build context-aware prompt for response generation."""
        query = getattr(state, 'processed_query', None) or getattr(state, 'original_query', None)
        
        # Get conversation context if available
        context = state.intermediate_results.get("conversation_context", {})
        expertise_level = context.get("user_expertise_level", "intermediate")
        response_style = context.get("preferred_response_style", "balanced")
        
        # Build system prompt based on intent
        intent = getattr(state, 'query_intent', 'conversation')
        
        system_prompts = {
            'code': f"You are a helpful programming assistant. Provide clear, working code examples with explanations suitable for {expertise_level} level users.",
            'analysis': f"You are an analytical expert. Provide thorough, balanced analysis with evidence and reasoning for {expertise_level} level understanding.",
            'creative': f"You are a creative assistant. Generate original, engaging content that is {response_style} in style.",
            'question': f"You are a knowledgeable assistant. Answer questions clearly and accurately for {expertise_level} level users.",
            'request': f"You are a helpful assistant. Fulfill requests efficiently and provide {response_style} responses.",
            'conversation': f"You are a friendly conversational AI. Engage naturally with {response_style} tone."
        }
        
        system_prompt = system_prompts.get(intent, system_prompts['conversation'])
        
        # Add conversation history if available
        history_text = ""
        if hasattr(state, 'conversation_history') and state.conversation_history:
            recent_history = state.conversation_history[-3:]
            for msg in recent_history:
                role = msg.get('role', 'unknown') if isinstance(msg, dict) else getattr(msg, 'role', 'unknown')
                content = msg.get('content', '')[:200] if isinstance(msg, dict) else getattr(msg, 'content', '')[:200]
                history_text += f"{role}: {content}\n"
        
        # Build final prompt
        if history_text:
            prompt = f"{system_prompt}\n\nConversation history:\n{history_text}\nUser: {query}\nAssistant:"
        else:
            prompt = f"{system_prompt}\n\nUser: {query}\nAssistant:"
        
        return prompt

    def _calculate_max_tokens(self, state: GraphState) -> int:
        """Calculate appropriate max tokens based on query complexity and intent."""
        intent = getattr(state, 'query_intent', 'conversation')
        complexity = getattr(state, 'query_complexity', 0.5)
        
        # Base token counts by intent
        base_tokens = {
            'code': 400,  # Code needs more space for examples
            'analysis': 500,  # Analysis needs detailed explanations
            'creative': 350,  # Creative content needs moderate space
            'question': 300,  # Questions need concise answers
            'request': 250,  # Requests usually need brief responses
            'conversation': 200  # Conversations should be concise
        }
        
        base_count = base_tokens.get(intent, 250)
        
        # Adjust based on complexity
        complexity_multiplier = 0.5 + (complexity * 1.0)  # 0.5x to 1.5x
        adjusted_count = int(base_count * complexity_multiplier)
        
        # Ensure reasonable bounds
        return max(100, min(adjusted_count, 800))

    def _calculate_temperature(self, state: GraphState) -> float:
        """Calculate appropriate temperature based on intent and requirements."""
        intent = getattr(state, 'query_intent', 'conversation')
        
        # Temperature settings by intent
        temperatures = {
            'code': 0.1,  # Code needs to be precise
            'analysis': 0.3,  # Analysis needs to be mostly factual
            'creative': 0.8,  # Creative content benefits from randomness
            'question': 0.2,  # Questions need accurate answers
            'request': 0.4,  # Requests need balanced responses
            'conversation': 0.6  # Conversations can be more natural
        }
        
        return temperatures.get(intent, 0.5)

    def _post_process_response(self, response: str, state: GraphState) -> str:
        """Post-process the generated response."""
        # Clean up the response
        response = response.strip()
        
        # Remove any unwanted prefixes
        prefixes_to_remove = ["Assistant:", "AI:", "Response:", "Answer:"]
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Ensure response ends properly
        if response and not response.endswith(('.', '!', '?', ':', '```')):
            response += '.'
        
        # Add helpful context for code responses
        intent = getattr(state, 'query_intent', 'conversation')
        if intent == 'code' and '```' in response:
            if not response.endswith('\n\nLet me know if you need any clarification or have questions about this code!'):
                response += '\n\nLet me know if you need any clarification or have questions about this code!'
        
        return response

class CacheUpdateNode(BaseGraphNode):
    """
    Updates conversation cache and learns from successful interactions.
    """
    
    def __init__(self, cache_manager=None):
        super().__init__("cache_update", NodeType.PROCESSING)
        self.cache_manager = cache_manager
    
    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        """Update conversation cache with current state."""
        logger.debug(f"[CacheUpdateNode] Enter execute. state.query_id={getattr(state, 'query_id', None)}")
        
        try:
            if not self.cache_manager:
                # No cache manager - skip caching
                return NodeResult(
                    success=True,
                    data={"cached": False, "reason": "no_cache_manager"},
                    confidence=1.0,
                    execution_time=0.01
                )
            
            session_id = getattr(state, 'session_id', None)
            if not session_id:
                # No session ID - skip caching
                return NodeResult(
                    success=True,
                    data={"cached": False, "reason": "no_session_id"},
                    confidence=1.0,
                    execution_time=0.01
                )
            
            # Prepare conversation entry
            conversation_entry = {
                "user_message": getattr(state, 'original_query', None),
                "assistant_response": getattr(state, 'final_response', None),
                "query_id": getattr(state, 'query_id', None),
                "timestamp": datetime.now().isoformat(),
                "intent": getattr(state, 'query_intent', 'unknown'),
                "complexity": getattr(state, 'query_complexity', 0.0),
                "total_cost": state.calculate_total_cost() if hasattr(state, 'calculate_total_cost') else None,
                "execution_time": state.calculate_total_time() if hasattr(state, 'calculate_total_time') else None,
                "models_used": [
                    result["result"].model_used 
                    for result in getattr(state, 'node_results', {}).values() 
                    if hasattr(result["result"], 'model_used')
                ]
            }
            
            # Cache conversation history
            history_key = f"conversation_history:{session_id}"
            try:
                # Get existing history
                existing_history = await self.cache_manager.get(history_key, [])
                if not isinstance(existing_history, list):
                    existing_history = []
                
                # Add new entry
                existing_history.append(conversation_entry)
                
                # Keep only last 50 entries
                if len(existing_history) > 50:
                    existing_history = existing_history[-50:]
                
                # Save updated history (TTL: 7 days)
                await self.cache_manager.set(history_key, existing_history, ttl=604800)
                
            except Exception as cache_error:
                logger.warning(f"Failed to cache conversation history: {cache_error}")
            
            # Cache user context and preferences
            context_key = f"user_context:{session_id}"
            try:
                user_context = state.intermediate_results.get("conversation_context", {})
                if user_context:
                    # Update with current interaction
                    user_context["last_interaction"] = datetime.now().isoformat()
                    user_context["total_interactions"] = user_context.get("total_interactions", 0) + 1
                    
                    # Save context (TTL: 30 days)
                    await self.cache_manager.set(context_key, user_context, ttl=2592000)
            
            except Exception as context_error:
                logger.warning(f"Failed to cache user context: {context_error}")
            
            # Cache query patterns for analytics
            pattern_key = f"query_pattern:{getattr(state, 'query_intent', 'unknown')}"
            try:
                pattern_data = {
                    "query": getattr(state, 'original_query', '')[:100],  # Truncated for privacy
                    "intent": getattr(state, 'query_intent', 'unknown'),
                    "complexity": getattr(state, 'query_complexity', 0.0),
                    "timestamp": datetime.now().isoformat(),
                    "success": bool(getattr(state, 'final_response', None)),
                    "cost": state.calculate_total_cost() if hasattr(state, 'calculate_total_cost') else None
                }
                
                # Get existing patterns
                existing_patterns = await self.cache_manager.get(pattern_key, [])
                if not isinstance(existing_patterns, list):
                    existing_patterns = []
                
                existing_patterns.append(pattern_data)
                
                # Keep only last 100 patterns per intent
                if len(existing_patterns) > 100:
                    existing_patterns = existing_patterns[-100:]
                
                # Save patterns (TTL: 90 days)
                await self.cache_manager.set(pattern_key, existing_patterns, ttl=7776000)
                
            except Exception as pattern_error:
                logger.warning(f"Failed to cache query patterns: {pattern_error}")
            
            logger.debug(f"[CacheUpdateNode] Success. state.query_id={getattr(state, 'query_id', None)}")
            
            return NodeResult(
                success=True,
                data={
                    "cached": True,
                    "conversation_cached": True,
                    "context_cached": bool(state.intermediate_results.get("conversation_context")),
                    "patterns_cached": True
                },
                confidence=1.0,
                execution_time=0.05
            )
            
        except Exception as e:
            logger.error(f"[CacheUpdateNode] Error: {e}")
            return NodeResult(
                success=False,
                error=f"Cache update failed: {str(e)}",
                execution_time=0.05
            )

class ErrorHandlerNode(BaseGraphNode):
    """
    Handles errors and provides fallback responses.
    """
    
    def __init__(self):
        super().__init__("error_handler", NodeType.PROCESSING)
        self.max_executions = 3  # Prevent infinite loops

    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        logger.debug(f"[ErrorHandlerNode] Enter execute. state.query_id={getattr(state, 'query_id', None)}")
        try:
            error_handler_count = state.execution_path.count("error_handler")
            if error_handler_count >= self.max_executions:
                if not state.final_response:
                    state.final_response = "I'm experiencing technical difficulties. Please try again later."
                logger.warning(f"[ErrorHandlerNode] Circuit breaker triggered. count={error_handler_count}")
                return NodeResult(
                    success=True,
                    data={"errors_handled": len(state.errors), "circuit_breaker_triggered": True},
                    confidence=0.1,
                    execution_time=0.01
                )
            if not state.final_response:
                state.final_response = (
                    "I apologize, but I encountered some issues while processing "
                    "your request. Please try rephrasing your question or try again later."
                )
            if len(state.errors) > 5:
                state.errors = state.errors[-3:]
            logger.debug(f"[ErrorHandlerNode] Success. state.query_id={getattr(state, 'query_id', None)}")
            return NodeResult(
                success=True,
                data={"errors_handled": len(state.errors)},
                confidence=0.3,
                execution_time=0.01
            )
        except Exception as e:
            logger.error(f"[ErrorHandlerNode] Error: {e}")
            if not state.final_response:
                state.final_response = "Technical error occurred. Please try again."
            return NodeResult(
                success=True,
                error=f"Error handler failed: {str(e)}",
                execution_time=0.01
            )

class ChatGraph(BaseGraph):
    """
    Main chat graph implementation for intelligent conversation management.
    
    Fixed to properly use LangGraph START/END constants and correct compilation order.
    """
    
    def __init__(self, model_manager: ModelManager, cache_manager=None):
        super().__init__(GraphType.CHAT, "chat_graph")
        self.model_manager = model_manager
        self.cache_manager = cache_manager
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "total_execution_time": 0.0,
            "node_stats": {}
        }
        # Automatically build the graph
        self.build()

    def get_performance_stats(self) -> Dict[str, Any]:
        stats = self.execution_stats.copy()
        total_exec = stats["total_executions"]
        if total_exec > 0:
            stats["success_rate"] = stats["successful_executions"] / total_exec
            stats["avg_execution_time"] = stats["total_execution_time"] / total_exec
        else:
            stats["success_rate"] = 0.0
            stats["avg_execution_time"] = 0.0
        for node_name, node_stats in stats["node_stats"].items():
            if node_stats["executions"] > 0:
                node_stats["success_rate"] = node_stats["success"] / node_stats["executions"]
                node_stats["avg_execution_time"] = node_stats["total_time"] / node_stats["executions"]
            else:
                node_stats["success_rate"] = 0.0
                node_stats["avg_execution_time"] = 0.0
        return {
            "graph_name": self.name,
            "graph_type": self.graph_type.value,
            "execution_count": total_exec,
            "success_rate": stats["success_rate"],
            "avg_execution_time": stats["avg_execution_time"],
            "total_execution_time": stats["total_execution_time"],
            "node_count": len(self.nodes),
            "node_stats": stats["node_stats"]
        }
    
    def define_nodes(self) -> Dict[str, BaseGraphNode]:
        """Define all nodes for the chat graph."""
        from app.graphs.base import EndNode
        return {
            "start": ContextManagerNode(self.cache_manager),  # Entrypoint for LangGraph
            "context_manager": ContextManagerNode(self.cache_manager),
            "intent_classifier": IntentClassifierNode(self.model_manager),
            "response_generator": ResponseGeneratorNode(self.model_manager),
            "cache_update": CacheUpdateNode(self.cache_manager),
            "error_handler": ErrorHandlerNode(),
            "end": EndNode()  # Add end node for LangGraph termination
        }
    
    def define_edges(self) -> List[tuple]:
        """
        Define the flow between nodes using descriptive node keys.
        """
        return [
            ("start", "context_manager"),  # Entry edge for LangGraph
            ("context_manager", "intent_classifier"),
            ("intent_classifier", "response_generator"),
            ("response_generator", "cache_update"),
            ("cache_update", self._check_for_errors, {
                "error_handler": "error_handler",
                "continue": "end"  # End the graph if no error
            }),
            ("error_handler", "end")  # End after error handling
        ]
    
    def _check_for_errors(self, state: GraphState) -> str:
        """Check if there are errors that need handling - prevent infinite loops."""
        if state.errors and "error_handler" not in state.execution_path:
            return "error_handler"
        return "continue"
    
    def build(self) -> None:
        """Build the chat graph with conditional routing."""
        # Call parent build which will handle START/END properly
        # This will compile the graph, so no modifications after this point
        super().build()
    
    async def execute(self, state: GraphState) -> GraphState:
        logger.debug(f"[ChatGraph] Enter execute. state.query_id={getattr(state, 'query_id', None)}")
        import time
        start_time = time.time()
        self.execution_stats["total_executions"] += 1
        try:
            result = await super().execute(state)
            logger.debug(f"[ChatGraph] Success. state.query_id={getattr(state, 'query_id', None)}")
            if len(result.errors) <= 2:
                self.execution_stats["successful_executions"] += 1
            for node_name in result.execution_path:
                if node_name not in self.execution_stats["node_stats"]:
                    self.execution_stats["node_stats"][node_name] = {
                        "executions": 0,
                        "success": 0,
                        "total_time": 0.0
                    }
                node_stats = self.execution_stats["node_stats"][node_name]
                node_stats["executions"] += 1
                if node_name in result.node_results:
                    node_result = result.node_results[node_name]["result"]
                    if node_result.success:
                        node_stats["success"] += 1
                    node_stats["total_time"] += node_result.execution_time
            execution_time = time.time() - start_time
            self.execution_stats["total_execution_time"] += execution_time
            return result
        except Exception as e:
            logger.error(f"[ChatGraph] Error: {e}")
            execution_time = time.time() - start_time
            self.execution_stats["total_execution_time"] += execution_time
            state.errors.append(f"Graph execution failed: {str(e)}")
            return state

# Export main classes
__all__ = [
    'ChatGraph',
    'ContextManagerNode',
    'IntentClassifierNode', 
    'ResponseGeneratorNode',
    'CacheUpdateNode',
    'ConversationContext',
    'ErrorHandlerNode'
]
