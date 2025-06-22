"""
ChatGraph - Intelligent conversation management with context awareness.
Implements sophisticated chat workflows with model selection and optimization.

Complete fixed version addressing LangGraph START/END constants and compilation issues.
"""
import re
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from app.graphs.base import (
    BaseGraph, BaseGraphNode, GraphState, NodeResult, 
    GraphType, NodeType
)
from app.models.manager import ModelManager, TaskType, QualityLevel
from app.models.ollama_client import ModelResult
from app.core.logging import get_logger
from langgraph.constants import START, END

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
        
        # Simple model selection logic
        if complexity > 0.7:
            return "llama2:13b"  # Use larger model for complex queries
        elif intent == "code":
            return "codellama:7b"  # Use code-specific model
        else:
            return "llama2:7b"  # Default model
    
    def _build_prompt(self, state: GraphState) -> str:
        """Build prompt from state context."""
        prompt_parts = []
        
        # Add system context
        prompt_parts.append("You are a helpful AI assistant.")
        
        # Add conversation history
        if state.conversation_history:
            prompt_parts.append("\nConversation history:")
            for msg in state.conversation_history[-3:]:  # Last 3 exchanges
                if isinstance(msg, dict):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                else:
                    # Assume ChatMessage object or similar
                    role = getattr(msg, "role", "unknown")
                    content = getattr(msg, "content", "")
                prompt_parts.append(f"{role.title()}: {content}")
        
        # Add style preferences
        context = state.intermediate_results.get("conversation_context", {})
        style = context.get("preferred_response_style", "balanced")
        if style == "concise":
            prompt_parts.append("Note: Keep response concise and to the point.")
        elif style == "detailed":
            prompt_parts.append("Note: Provide comprehensive and detailed explanations.")
        
        # Add current query
        prompt_parts.append(f"User: {state.processed_query}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def _calculate_max_tokens(self, state: GraphState) -> int:
        """Calculate appropriate max tokens based on context."""
        context = state.intermediate_results.get("conversation_context", {})
        style = context.get("preferred_response_style", "balanced")
        complexity = getattr(state, 'query_complexity', 0.5)
        
        base_tokens = {
            "concise": 150,
            "balanced": 300,
            "detailed": 500
        }.get(style, 300)
        
        # Adjust for complexity
        if complexity > 0.7:
            base_tokens = int(base_tokens * 1.5)
        elif complexity < 0.3:
            base_tokens = int(base_tokens * 0.7)
        
        return min(base_tokens, 800)  # Cap at 800 tokens
    
    def _calculate_temperature(self, state: GraphState) -> float:
        """Calculate appropriate temperature based on intent."""
        intent = getattr(state, 'query_intent', 'question')
        
        temperature_mapping = {
            "creative": 0.8,
            "conversation": 0.7,
            "question": 0.5,
            "analysis": 0.3,
            "code": 0.2,
            "request": 0.5
        }
        
        return temperature_mapping.get(intent, 0.6)
    
    def _post_process_response(self, response: str, state: GraphState) -> str:
        """Post-process generated response."""
        # Basic cleanup
        response = response.strip()
        
        # Remove any artifacts from prompt structure
        if response.startswith("Assistant:"):
            response = response[10].strip()
        
        # Ensure response ends properly
        if not response.endswith(('.', '!', '?', ':', '"', "'")):
            # Find last complete sentence
            sentences = response.split('.')
            if len(sentences) > 1 and sentences[-1].strip():
                response = '.'.join(sentences[:-1]) + '.'
        
        return response

class CacheUpdateNode(BaseGraphNode):
    """
    Updates conversation cache and learns from successful interactions.
    """
    
    def __init__(self, cache_manager=None):
        super().__init__("cache_update", NodeType.PROCESSING)
        self.cache_manager = cache_manager
    
    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        logger.debug(f"[CacheUpdateNode] Enter execute. state.query_id={getattr(state, 'query_id', None)}")
        try:
            updates_made = []
            
            # ENSURE final_response is set if not already set
            if not state.final_response:
                response_data = state.intermediate_results.get("response_generator", {})
                generated_response = response_data.get("response", "")
                if generated_response:
                    state.final_response = generated_response
            
            # Update conversation history
            if state.session_id and self.cache_manager:
                # Add current exchange to history
                new_exchange = [
                    {"role": "user", "content": state.original_query, "timestamp": time.time()},
                    {"role": "assistant", "content": state.final_response, "timestamp": time.time()}
                ]
                
                updated_history = state.conversation_history + new_exchange
                
                # Keep only recent history (last 20 exchanges)
                if len(updated_history) > 40:  # 20 exchanges = 40 messages
                    updated_history = updated_history[-40:]
                
                await self.cache_manager.update_conversation_history(
                    state.session_id, 
                    updated_history
                )
                updates_made.append("conversation_history")
            
            # Cache successful routing patterns
            if not state.errors and self.cache_manager:
                await self.cache_manager.cache_successful_route(
                    state.original_query,
                    state.execution_path,
                    state.calculate_total_cost()
                )
                updates_made.append("routing_pattern")
            
            # Update user preferences
            if state.user_id and self.cache_manager:
                context = state.intermediate_results.get("conversation_context", {})
                if context:
                    await self.cache_manager.update_user_pattern(
                        state.user_id,
                        {
                            "expertise_level": context.get("user_expertise_level"),
                            "response_style": context.get("preferred_response_style"),
                            "conversation_mood": context.get("conversation_mood"),
                            "last_interaction": time.time()
                        }
                    )
                    updates_made.append("user_preferences")
            
            logger.debug(f"[CacheUpdateNode] Success. state.query_id={getattr(state, 'query_id', None)}")
            return NodeResult(
                success=True,
                data={
                    "updates_made": updates_made,
                    "cache_operations": len(updates_made)
                },
                confidence=1.0,
                execution_time=0.05  # Cache operations are fast
            )
            
        except Exception as e:
            logger.error(f"[CacheUpdateNode] Error: {e}")
            return NodeResult(
                success=True,  # Non-critical failure
                data={"updates_made": []},
                confidence=0.5,
                error=f"Cache update failed: {str(e)}"
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
        return {
            "context_manager": ContextManagerNode(self.cache_manager),
            "intent_classifier": IntentClassifierNode(self.model_manager),
            "response_generator": ResponseGeneratorNode(self.model_manager),
            "cache_update": CacheUpdateNode(self.cache_manager),
            "error_handler": ErrorHandlerNode()  # Include error handler in initial nodes
        }
    
    def define_edges(self) -> List[tuple]:
        """
        Define the flow between nodes using proper LangGraph constants.
        
        Fixed to use START and END constants and include conditional edges.
        """
        return [
            # Use START constant for entry point
            (START, "context_manager"),
            ("context_manager", "intent_classifier"),
            ("intent_classifier", "response_generator"),
            ("response_generator", "cache_update"),
            # Add conditional edge for error handling
            ("cache_update", self._check_for_errors, {
                "error_handler": "error_handler",
                "continue": END
            }),
            # Use END constant for exit point
            ("error_handler", END)
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
