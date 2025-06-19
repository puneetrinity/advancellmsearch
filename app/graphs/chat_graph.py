"""
ChatGraph - Intelligent conversation management with context awareness.
Implements sophisticated chat workflows with model selection and optimization.
"""
import re
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from app.graphs.base import (
    BaseGraph, BaseGraphNode, GraphState, NodeResult, 
    GraphType, NodeType, ConditionalNode
)
from app.models.manager import ModelManager, TaskType, QualityLevel
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
        """Analyze and update conversation context."""
        try:
            # Load conversation history if session exists
            if state.session_id and self.cache_manager:
                cached_history = await self.cache_manager.get_conversation_history(state.session_id)
                if cached_history:
                    state.conversation_history = cached_history
            
            # Analyze conversation context
            context = self._analyze_conversation_context(state)
            
            # Update processed query with context
            state.processed_query = self._enhance_query_with_context(
                state.original_query, 
                context, 
                state.conversation_history
            )
            
            # Store context in state
            state.intermediate_results["conversation_context"] = context.__dict__
            
            # Determine conversation complexity
            complexity = self._assess_query_complexity(state.original_query, context)
            state.query_complexity = complexity
            
            return NodeResult(
                success=True,
                data={
                    "context": context.__dict__,
                    "processed_query": state.processed_query,
                    "complexity": complexity,
                    "history_length": len(state.conversation_history)
                },
                confidence=0.9,
                metadata={
                    "context_enhancement": state.processed_query != state.original_query,
                    "history_available": len(state.conversation_history) > 0
                }
            )
            
        except Exception as e:
            return NodeResult(
                success=False,
                error=f"Context management failed: {str(e)}"
            )
    
    def _analyze_conversation_context(self, state: GraphState) -> ConversationContext:
        """Analyze conversation history to build context."""
        context = ConversationContext()
        
        if not state.conversation_history:
            return context
        
        # Extract user information from history
        user_messages = [
            msg for msg in state.conversation_history 
            if msg.get("role") == "user"
        ]
        
        # Detect user expertise level
        context.user_expertise_level = self._detect_expertise_level(user_messages)
        
        # Extract key entities and topics
        all_text = " ".join([msg.get("content", "") for msg in user_messages])
        context.key_entities = self._extract_entities(all_text)
        context.previous_topics = self._extract_topics(user_messages)
        
        # Detect conversation mood/style
        context.conversation_mood = self._detect_conversation_mood(user_messages)
        context.preferred_response_style = self._detect_response_style_preference(user_messages)
        
        # Identify main conversation topic
        if context.previous_topics:
            context.conversation_topic = context.previous_topics[-1]  # Most recent topic
        
        return context
    
    def _detect_expertise_level(self, user_messages: List[Dict[str, Any]]) -> str:
        """Detect user's expertise level from their messages."""
        if not user_messages:
            return "intermediate"
        
        technical_indicators = 0
        total_words = 0
        
        for message in user_messages[-3:]:  # Analyze last 3 messages
            content = message.get("content", "").lower()
            words = content.split()
            total_words += len(words)
            
            # Technical terms indicate higher expertise
            technical_terms = [
                "algorithm", "api", "framework", "implementation", "optimization",
                "architecture", "database", "schema", "deployment", "scalability",
                "performance", "latency", "throughput", "concurrent", "async"
            ]
            
            for term in technical_terms:
                if term in content:
                    technical_indicators += 1
        
        if total_words == 0:
            return "intermediate"
        
        technical_ratio = technical_indicators / total_words
        
        if technical_ratio > 0.05:
            return "expert"
        elif technical_ratio > 0.02:
            return "intermediate"
        else:
            return "beginner"
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text using simple patterns."""
        entities = []
        
        # Simple entity patterns (in production, use NER models)
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b\w+(?:\.\w+)+\b',  # Domain names/URLs
            r'\b[A-Z]{2,}\b',  # Acronyms
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches[:10])  # Limit to prevent noise
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_topics(self, user_messages: List[Dict[str, Any]]) -> List[str]:
        """Extract main topics from conversation."""
        topics = []
        
        # Topic keywords mapping
        topic_keywords = {
            "programming": ["code", "python", "javascript", "programming", "development"],
            "ai_ml": ["ai", "machine learning", "neural network", "model", "training"],
            "web_development": ["website", "html", "css", "react", "frontend", "backend"],
            "data_science": ["data", "analysis", "visualization", "pandas", "statistics"],
            "business": ["strategy", "market", "revenue", "growth", "business"],
            "general_help": ["help", "how to", "what is", "explain", "guide"]
        }
        
        for message in user_messages:
            content = message.get("content", "").lower()
            
            for topic, keywords in topic_keywords.items():
                if any(keyword in content for keyword in keywords):
                    if topic not in topics:
                        topics.append(topic)
                    break
        
        return topics
    
    def _detect_conversation_mood(self, user_messages: List[Dict[str, Any]]) -> str:
        """Detect conversation mood/formality."""
        if not user_messages:
            return "neutral"
        
        recent_content = " ".join([
            msg.get("content", "") for msg in user_messages[-2:]
        ]).lower()
        
        # Casual indicators
        casual_indicators = ["hey", "hi", "thanks", "cool", "awesome", "yeah"]
        professional_indicators = ["please", "could you", "would you", "thank you"]
        
        casual_count = sum(1 for indicator in casual_indicators if indicator in recent_content)
        professional_count = sum(1 for indicator in professional_indicators if indicator in recent_content)
        
        if casual_count > professional_count:
            return "casual"
        elif professional_count > casual_count:
            return "professional"
        else:
            return "neutral"
    
    def _detect_response_style_preference(self, user_messages: List[Dict[str, Any]]) -> str:
        """Detect user's preferred response style."""
        if not user_messages:
            return "balanced"
        
        # Analyze message length and complexity
        avg_length = sum(len(msg.get("content", "").split()) for msg in user_messages) / len(user_messages)
        
        if avg_length < 10:
            return "concise"
        elif avg_length > 30:
            return "detailed"
        else:
            return "balanced"
    
    def _enhance_query_with_context(
        self, 
        original_query: str, 
        context: ConversationContext, 
        history: List[Dict[str, Any]]
    ) -> str:
        """Enhance query with conversation context."""
        enhanced_query = original_query
        
        # Add context if relevant
        if context.conversation_topic and len(history) > 0:
            # Check if query refers to previous context
            context_indicators = ["this", "that", "it", "the above", "previous"]
            if any(indicator in original_query.lower() for indicator in context_indicators):
                enhanced_query = f"In the context of {context.conversation_topic}: {original_query}"
        
        # Add user expertise context
        if context.user_expertise_level == "beginner":
            enhanced_query += " (Please explain in simple terms)"
        elif context.user_expertise_level == "expert":
            enhanced_query += " (Technical details are welcome)"
        
        return enhanced_query
    
    def _assess_query_complexity(self, query: str, context: ConversationContext) -> float:
        """Assess query complexity (0.0 = simple, 1.0 = very complex)."""
        complexity_score = 0.0
        
        # Length-based complexity
        word_count = len(query.split())
        if word_count > 50:
            complexity_score += 0.3
        elif word_count > 20:
            complexity_score += 0.2
        elif word_count > 10:
            complexity_score += 0.1
        
        # Question complexity indicators
        complex_indicators = [
            "analyze", "compare", "evaluate", "explain in detail", "comprehensive",
            "strategy", "approach", "methodology", "framework", "architecture"
        ]
        
        for indicator in complex_indicators:
            if indicator in query.lower():
                complexity_score += 0.2
                break
        
        # Multiple questions
        question_count = query.count("?")
        if question_count > 1:
            complexity_score += 0.15
        
        # Technical terms increase complexity
        if context.user_expertise_level == "expert":
            complexity_score += 0.1
        
        return min(1.0, complexity_score)


class IntentClassifierNode(BaseGraphNode):
    """
    Classifies user intent to route to appropriate processing.
    """
    
    def __init__(self, model_manager: ModelManager):
        super().__init__("intent_classifier", NodeType.PROCESSING)
        self.model_manager = model_manager
    
    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        """Classify user intent."""
        try:
            # Use fast model for classification
            classifier_model = self.model_manager.select_optimal_model(
                TaskType.SIMPLE_CLASSIFICATION,
                QualityLevel.MINIMAL
            )
            
            # Build classification prompt
            prompt = self._build_classification_prompt(state.processed_query)
            
            # Get classification
            result = await self.model_manager.generate(
                model_name=classifier_model,
                prompt=prompt,
                max_tokens=20,
                temperature=0.1  # Low temperature for consistent classification
            )
            
            if not result.success:
                # Fallback to rule-based classification
                intent = self._rule_based_classification(state.processed_query)
            else:
                intent = self._parse_intent_from_response(result.text)
            
            state.query_intent = intent
            
            return NodeResult(
                success=True,
                data={
                    "intent": intent,
                    "classification_method": "model" if result.success else "rule_based"
                },
                confidence=0.8 if result.success else 0.6,
                cost=result.cost if result.success else 0.0,
                model_used=result.model_used if result.success else None,
                execution_time=result.execution_time if result.success else 0.1
            )
            
        except Exception as e:
            # Fallback to rule-based
            intent = self._rule_based_classification(state.processed_query)
            state.query_intent = intent
            
            return NodeResult(
                success=True,
                data={"intent": intent, "classification_method": "fallback"},
                confidence=0.5,
                warnings=[f"Classification model failed: {str(e)}"]
            )
    
    def _build_classification_prompt(self, query: str) -> str:
        """Build prompt for intent classification."""
        return f"""Classify the following user query into one of these categories:
- question: Asking for information or explanation
- request: Asking for help with a task
- conversation: General chat or greeting
- creative: Creative writing or brainstorming
- analysis: Asking for analysis or comparison
- code: Programming or technical help

Query: "{query}"

Category:"""
    
    def _parse_intent_from_response(self, response: str) -> str:
        """Parse intent from model response."""
        response = response.strip().lower()
        
        # Map to standard intents
        intent_mapping = {
            "question": "question",
            "request": "request", 
            "conversation": "conversation",
            "creative": "creative",
            "analysis": "analysis",
            "code": "code"
        }
        
        for key, intent in intent_mapping.items():
            if key in response:
                return intent
        
        return "question"  # Default fallback
    
    def _rule_based_classification(self, query: str) -> str:
        """Rule-based intent classification as fallback."""
        query_lower = query.lower()
        
        # Conversation patterns
        if any(word in query_lower for word in ["hello", "hi", "hey", "how are you"]):
            return "conversation"
        
        # Code/technical patterns
        if any(word in query_lower for word in ["code", "function", "programming", "debug", "error"]):
            return "code"
        
        # Analysis patterns
        if any(word in query_lower for word in ["compare", "analyze", "difference", "pros and cons"]):
            return "analysis"
        
        # Creative patterns
        if any(word in query_lower for word in ["write", "create", "story", "poem", "creative"]):
            return "creative"
        
        # Request patterns
        if any(word in query_lower for word in ["help me", "can you", "please", "how to"]):
            return "request"
        
        # Default to question
        return "question"


class ResponseGeneratorNode(BaseGraphNode):
    """
    Generates final response using appropriate model and context.
    """
    
    def __init__(self, model_manager: ModelManager):
        super().__init__("response_generator", NodeType.PROCESSING)
        self.model_manager = model_manager
    
    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        """Generate response based on query, intent, and context."""
        try:
            # Select appropriate model based on intent and complexity
            selected_model = self._select_model_for_intent(state)
            
            # Build generation prompt
            prompt = self._build_generation_prompt(state)
            
            # Generate response
            result = await self.model_manager.generate(
                model_name=selected_model,
                prompt=prompt,
                max_tokens=self._calculate_max_tokens(state),
                temperature=self._calculate_temperature(state)
            )
            
            if result.success:
                # Post-process response
                final_response = self._post_process_response(result.text, state)
                state.final_response = final_response
                
                return NodeResult(
                    success=True,
                    data={
                        "response": final_response,
                        "model_selected": selected_model,
                        "prompt_length": len(prompt),
                        "response_length": len(final_response)
                    },
                    confidence=0.85,
                    cost=result.cost,
                    model_used=result.model_used,
                    execution_time=result.execution_time
                )
            else:
                return NodeResult(
                    success=False,
                    error=f"Response generation failed: {result.error}"
                )
                
        except Exception as e:
            return NodeResult(
                success=False,
                error=f"Response generator failed: {str(e)}"
            )
    
    def _select_model_for_intent(self, state: GraphState) -> str:
        """Select optimal model based on intent and complexity."""
        intent = state.query_intent or "question"
        complexity = state.query_complexity
        quality = state.quality_requirement
        
        # Intent-based task mapping
        intent_task_mapping = {
            "code": TaskType.CODE_TASKS,
            "analysis": TaskType.ANALYTICAL_REASONING,
            "creative": TaskType.CREATIVE_WRITING,
            "conversation": TaskType.CONVERSATION,
            "question": TaskType.QA_AND_SUMMARY,
            "request": TaskType.QA_AND_SUMMARY
        }
        
        task_type = intent_task_mapping.get(intent, TaskType.CONVERSATION)
        
        # Upgrade quality for complex queries
        if complexity > 0.7 and quality == QualityLevel.BALANCED:
            quality = QualityLevel.HIGH
        elif complexity > 0.5 and quality == QualityLevel.MINIMAL:
            quality = QualityLevel.BALANCED
        
        return self.model_manager.select_optimal_model(task_type, quality)
    
    def _build_generation_prompt(self, state: GraphState) -> str:
        """Build comprehensive prompt for response generation."""
        context = state.intermediate_results.get("conversation_context", {})
        
        # Base prompt
        prompt_parts = []
        
        # Add conversation context if available
        if state.conversation_history:
            recent_history = state.conversation_history[-3:]  # Last 3 exchanges
            prompt_parts.append("Previous conversation:")
            for msg in recent_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")[:200]  # Limit length
                prompt_parts.append(f"{role.title()}: {content}")
            prompt_parts.append("")
        
        # Add user context
        expertise = context.get("user_expertise_level", "intermediate")
        style = context.get("preferred_response_style", "balanced")
        mood = context.get("conversation_mood", "neutral")
        
        if expertise == "beginner":
            prompt_parts.append("Note: Explain concepts clearly and avoid jargon.")
        elif expertise == "expert":
            prompt_parts.append("Note: Technical details and precise explanations are welcomed.")
        
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
        complexity = state.query_complexity
        
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
        intent = state.query_intent or "question"
        
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
            response = response[10:].strip()
        
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
        """Update cache with conversation and performance data."""
        try:
            updates_made = []
            
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
            return NodeResult(
                success=True,  # Non-critical failure
                data={"updates_made": []},
                confidence=0.5,
                warnings=[f"Cache update failed: {str(e)}"]
            )


class ChatGraph(BaseGraph):
    """
    Main chat graph implementation for intelligent conversation management.
    """
    
    def __init__(self, model_manager: ModelManager, cache_manager=None):
        super().__init__(GraphType.CHAT, "chat_graph")
        self.model_manager = model_manager
        self.cache_manager = cache_manager
    
    def define_nodes(self) -> Dict[str, BaseGraphNode]:
        """Define all nodes for the chat graph."""
        return {
            "context_manager": ContextManagerNode(self.cache_manager),
            "intent_classifier": IntentClassifierNode(self.model_manager),
            "response_generator": ResponseGeneratorNode(self.model_manager),
            "cache_update": CacheUpdateNode(self.cache_manager)
        }
    
    def define_edges(self) -> List[tuple]:
        """Define the flow between nodes."""
        return [
            ("start", "context_manager"),
            ("context_manager", "intent_classifier"),
            ("intent_classifier", "response_generator"),
            ("response_generator", "cache_update"),
            ("cache_update", "end")
        ]
    
    def build(self) -> None:
        """Build the chat graph with conditional routing."""
        super().build()
        
        # Add conditional edge for error handling
        def check_for_errors(state: GraphState) -> str:
            if state.errors:
                return "error_handler"
            return "end"
        
        self.add_conditional_edge("cache_update", check_for_errors)


# Export main classes
__all__ = [
    'ChatGraph',
    'ContextManagerNode',
    'IntentClassifierNode', 
    'ResponseGeneratorNode',
    'CacheUpdateNode',
    'ConversationContext'
]
