from typing import Dict, Any
import structlog

from app.graphs.base import (
    BaseGraph, BaseGraphNode, GraphState, NodeResult, 
    GraphType, StartNode, EndNode, ErrorHandlerNode
)
from app.models.manager import ModelManager
from app.cache.redis_client import CacheManager

logger = structlog.get_logger(__name__)

# --- Node Implementations ---

class ContextManagerNode(BaseGraphNode):
    """Manages conversation context and history"""
    def __init__(self, cache_manager: CacheManager):
        super().__init__("context_manager", "processing")
        self.cache_manager = cache_manager

    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        try:
            # Load conversation history from cache
            if state.session_id:
                cached_history = await self.cache_manager.get_conversation_history(state.session_id)
                if cached_history:
                    state.conversation_history = cached_history
                    self.logger.info(
                        "Loaded conversation history",
                        session_id=state.session_id,
                        history_length=len(cached_history)
                    )
            # Analyze conversation context
            context_analysis = self._analyze_context(state)
            # Update user preferences based on history
            inferred_preferences = self._infer_preferences(state)
            state.user_preferences.update(inferred_preferences)
            return NodeResult(
                success=True,
                confidence=0.9,
                data={
                    "context_loaded": True,
                    "history_length": len(state.conversation_history),
                    "context_analysis": context_analysis,
                    "inferred_preferences": inferred_preferences
                },
                cost=0.0
            )
        except Exception as e:
            return NodeResult(
                success=False,
                error=f"Context management failed: {str(e)}",
                confidence=0.0
            )

    def _analyze_context(self, state: GraphState) -> Dict[str, Any]:
        history = state.conversation_history
        if not history:
            return {"type": "new_conversation", "complexity": "simple"}
        user_messages = [msg for msg in history if msg.get("role") == "user"]
        recent_topics = []
        for msg in user_messages[-5:]:
            content = msg.get("content", "").lower()
            if any(word in content for word in ["code", "programming", "python", "javascript"]):
                recent_topics.append("programming")
            elif any(word in content for word in ["analyze", "research", "study"]):
                recent_topics.append("research")
            elif any(word in content for word in ["explain", "what is", "how does"]):
                recent_topics.append("educational")
        conversation_length = len(user_messages)
        complexity = "simple"
        if conversation_length > 10:
            complexity = "complex"
        elif conversation_length > 5:
            complexity = "moderate"
        return {
            "type": "continuing_conversation",
            "length": conversation_length,
            "complexity": complexity,
            "recent_topics": list(set(recent_topics)),
            "last_user_message": user_messages[-1].get("content", "") if user_messages else ""
        }

    def _infer_preferences(self, state: GraphState) -> Dict[str, Any]:
        preferences = {}
        history = state.conversation_history
        if len(history) >= 4:
            user_messages = [msg.get("content", "") for msg in history if msg.get("role") == "user"]
            avg_user_length = sum(len(msg) for msg in user_messages) / len(user_messages)
            if avg_user_length < 50:
                preferences["response_style"] = "concise"
            elif avg_user_length > 200:
                preferences["response_style"] = "detailed"
            else:
                preferences["response_style"] = "balanced"
            technical_keywords = ["api", "algorithm", "implementation", "technical", "code"]
            technical_mentions = sum(
                1 for msg in user_messages 
                for keyword in technical_keywords 
                if keyword in msg.lower()
            )
            if technical_mentions > len(user_messages) * 0.3:
                preferences["technical_level"] = "advanced"
            else:
                preferences["technical_level"] = "general"
        return preferences

class IntentClassifierNode(BaseGraphNode):
    """Classifies user intent to determine appropriate response strategy"""
    def __init__(self, model_manager: ModelManager):
        super().__init__("intent_classifier", "processing")
        self.model_manager = model_manager

    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        try:
            query = state.original_query
            classification_prompt = self._build_classification_prompt(query, state)
            model_result = await self.model_manager.generate(
                model_name="phi:mini",
                prompt=classification_prompt,
                max_tokens=100,
                temperature=0.1
            )
            if not model_result.success:
                intent = self._fallback_classification(query)
                confidence = 0.6
            else:
                intent, confidence = self._parse_classification_result(model_result.text)
            state.intermediate_results["intent"] = intent
            state.confidence_scores["intent_classification"] = confidence
            return NodeResult(
                success=True,
                confidence=confidence,
                data={
                    "intent": intent,
                    "classification_method": "model" if model_result.success else "fallback"
                },
                cost=model_result.cost if model_result.success else 0.0,
                model_used="phi:mini" if model_result.success else "fallback"
            )
        except Exception as e:
            return NodeResult(
                success=False,
                error=f"Intent classification failed: {str(e)}",
                confidence=0.0
            )

    def _build_classification_prompt(self, query: str, state: GraphState) -> str:
        context_info = ""
        if state.conversation_history:
            context_info = f"Previous context: {state.conversation_history[-1].get('content', '')[:100]}..."
        prompt = f"""Classify the user's intent from this query. Respond with only the category name.

Categories:
- simple_question: Basic factual questions
- complex_analysis: Requires research or deep thinking
- conversational: Casual chat, greetings, or follow-ups
- search_needed: Needs web search for current information
- code_help: Programming or technical assistance

{context_info}

User query: "{query}"

Intent category:"""
        return prompt

    def _parse_classification_result(self, result_text: str):
        text = result_text.strip().lower()
        intent_mapping = {
            "simple_question": ("simple_chat", 0.9),
            "complex_analysis": ("analysis_needed", 0.8),
            "conversational": ("simple_chat", 0.9),
            "search_needed": ("search_needed", 0.9),
            "code_help": ("code_assistance", 0.8)
        }
        for keyword, (intent, confidence) in intent_mapping.items():
            if keyword in text:
                return intent, confidence
        return "simple_chat", 0.5

    def _fallback_classification(self, query: str):
        query_lower = query.lower()
        search_keywords = ["latest", "current", "recent", "news", "today", "2024", "2025"]
        if any(keyword in query_lower for keyword in search_keywords):
            return "search_needed"
        code_keywords = ["code", "programming", "function", "api", "debug", "error"]
        if any(keyword in query_lower for keyword in code_keywords):
            return "code_assistance"
        analysis_keywords = ["analyze", "compare", "evaluate", "research", "study"]
        if any(keyword in query_lower for keyword in analysis_keywords):
            return "analysis_needed"
        return "simple_chat"

class ResponseGeneratorNode(BaseGraphNode):
    """Generates the final response using appropriate model"""
    def __init__(self, model_manager: ModelManager):
        super().__init__("response_generator", "processing")
        self.model_manager = model_manager

    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        try:
            intent = state.intermediate_results.get("intent", "simple_chat")
            model_name = self._select_model(intent, getattr(state, "quality_requirement", "balanced"))
            response_prompt = self._build_response_prompt(state)
            model_result = await self.model_manager.generate(
                model_name=model_name,
                prompt=response_prompt,
                max_tokens=self._get_max_tokens(intent),
                temperature=self._get_temperature(intent)
            )
            if not model_result.success:
                response = "I apologize, but I'm having trouble processing your request right now. Please try again."
                confidence = 0.1
            else:
                response = model_result.text.strip()
                confidence = 0.8
            state.final_response = response
            state.response_metadata = {
                "intent": intent,
                "model_used": model_name,
                "response_length": len(response),
                "generation_successful": model_result.success
            }
            return NodeResult(
                success=True,
                confidence=confidence,
                data={
                    "response": response,
                    "intent": intent,
                    "model_used": model_name
                },
                cost=model_result.cost if model_result.success else 0.0,
                model_used=model_name
            )
        except Exception as e:
            return NodeResult(
                success=False,
                error=f"Response generation failed: {str(e)}",
                confidence=0.0
            )

    def _select_model(self, intent: str, quality_requirement: str) -> str:
        if quality_requirement == "premium":
            if intent in ["analysis_needed", "code_assistance"]:
                return "llama2:13b"
            return "llama2:7b"
        elif quality_requirement == "high":
            if intent == "code_assistance":
                return "codellama"
            elif intent == "analysis_needed":
                return "mistral:7b"
            return "llama2:7b"
        else:
            if intent == "simple_chat":
                return "phi:mini"
            return "llama2:7b"

    def _build_response_prompt(self, state: GraphState) -> str:
        intent = state.intermediate_results.get("intent", "simple_chat")
        query = state.original_query
        context = ""
        if state.conversation_history:
            recent_context = state.conversation_history[-3:]
            context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_context])
        style_instruction = ""
        if state.user_preferences.get("response_style") == "concise":
            style_instruction = "Keep your response concise and to the point."
        elif state.user_preferences.get("response_style") == "detailed":
            style_instruction = "Provide a comprehensive and detailed response."
        if intent == "simple_chat":
            prompt = f"""You are a helpful AI assistant. Respond naturally and conversationally to the user's query.

{style_instruction}

{f"Previous conversation:\n{context}\n" if context else ""}

User: {query}
"""
            return prompt
        elif intent == "analysis_needed":
            prompt = f"""You are an expert AI assistant. Analyze the user's query in depth and provide a detailed, structured answer.

{style_instruction}

{f"Conversation context:\n{context}\n" if context else ""}

User: {query}
"""
            return prompt
        elif intent == "code_assistance":
            prompt = f"""You are a coding assistant. Help the user with programming queries by providing clear code examples and explanations.

{style_instruction}

{f"Relevant chat history:\n{context}\n" if context else ""}

User: {query}
"""
            return prompt
        elif intent == "search_needed":
            prompt = f"""The user has asked for up-to-date or recent information. Summarize or answer based on the latest available data.

{style_instruction}

{f"Context:\n{context}\n" if context else ""}

User: {query}
"""
            return prompt
        else:
            prompt = f"""You are a helpful AI assistant. Respond naturally and conversationally to the user's query.

{style_instruction}

{f"Previous conversation:\n{context}\n" if context else ""}

User: {query}
"""
            return prompt

    def _get_max_tokens(self, intent: str) -> int:
        if intent == "analysis_needed":
            return 512
        elif intent == "code_assistance":
            return 384
        elif intent == "simple_chat":
            return 256
        else:
            return 256

    def _get_temperature(self, intent: str) -> float:
        if intent in ["simple_chat", "conversational"]:
            return 0.7
        elif intent == "analysis_needed":
            return 0.5
        elif intent == "code_assistance":
            return 0.2
        elif intent == "search_needed":
            return 0.3
        else:
            return 0.5

# --- Orchestration Graph ---

class ChatGraph(BaseGraph):
    """
    Orchestrates the conversational flow for chat, managing context, intent, and response.
    """
    def __init__(self, model_manager: ModelManager, cache_manager: CacheManager):
        super().__init__(name="chat_graph", graph_type=GraphType.Chat)
        self.start_node = StartNode()
        self.context_manager_node = ContextManagerNode(cache_manager)
        self.intent_classifier_node = IntentClassifierNode(model_manager)
        self.response_generator_node = ResponseGeneratorNode(model_manager)
        self.end_node = EndNode()
        self.error_handler_node = ErrorHandlerNode()
        self.nodes = [
            self.start_node,
            self.context_manager_node,
            self.intent_classifier_node,
            self.response_generator_node,
            self.end_node
        ]
        self.error_node = self.error_handler_node

    async def run(self, state: GraphState) -> NodeResult:
        current_state = state
        for node in self.nodes:
            try:
                result = await node.execute(current_state)
                if not result.success:
                    logger.error(f"Node {node.name} failed", error=result.error)
                    err_result = await self.error_node.execute(current_state, error=result.error)
                    return err_result
            except Exception as exc:
                logger.exception(f"Exception in node {node.name}", exc_info=exc)
                err_result = await self.error_node.execute(current_state, error=str(exc))
                return err_result
        return NodeResult(
            success=True,
            data={
                "response": current_state.final_response,
                "metadata": getattr(current_state, "response_metadata", {})
            }
        )
