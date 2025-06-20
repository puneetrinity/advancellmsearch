"""
Search Graph Nodes
Individual node implementations for the search workflow
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

from app.providers.router import SmartSearchRouter
from app.providers.search_providers import SearchResult, SearchResponse

logger = logging.getLogger(__name__)

class NodeType(Enum):
    SEARCH = "search"
    ANALYZE = "analyze"
    SYNTHESIZE = "synthesize"
    ROUTE = "route"
    ENHANCE = "enhance"
    VALIDATE = "validate"

class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class GraphContext:
    """Shared context across all graph nodes"""
    query: str
    user_budget: float = 2.0
    quality_requirement: str = "standard"
    max_results: int = 10
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    search_results: List[Any] = field(default_factory=list)
    enhanced_results: List[Any] = field(default_factory=list)
    analysis: Dict[str, Any] = field(default_factory=dict)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    total_cost: float = 0.0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NodeResult:
    """Result from executing a graph node"""
    node_id: str
    status: NodeStatus
    data: Dict[str, Any]
    execution_time: float
    error: Optional[str] = None
    next_nodes: List[str] = field(default_factory=list)

class GraphNode(ABC):
    """Abstract base class for graph nodes"""
    
    def __init__(self, node_id: str, node_type: NodeType):
        self.node_id = node_id
        self.node_type = node_type
        self.dependencies = []
        self.conditions = []
        
    @abstractmethod
    async def execute(self, context: GraphContext) -> NodeResult:
        """Execute the node logic"""
        pass
    
    def add_dependency(self, node_id: str):
        """Add a dependency on another node"""
        self.dependencies.append(node_id)
    
    def add_condition(self, condition_func: Callable[[GraphContext], bool]):
        """Add a condition that must be met to execute this node"""
        self.conditions.append(condition_func)
    
    def should_execute(self, context: GraphContext) -> bool:
        """Check if this node should execute based on conditions"""
        return all(condition(context) for condition in self.conditions)

class SearchRouterNode(GraphNode):
    """Intelligent search routing node"""
    
    def __init__(self, node_id: str = "search_router"):
        super().__init__(node_id, NodeType.ROUTE)
        self.router = None
        
    async def execute(self, context: GraphContext) -> NodeResult:
        start_time = time.time()
        
        try:
            # Determine if search is needed
            search_needed = self._should_search(context)
            
            result_data = {
                "search_needed": search_needed,
                "routing_decision": self._get_routing_decision(context),
                "estimated_cost": self._estimate_cost(context)
            }
            
            next_nodes = ["primary_search"] if search_needed else ["direct_response"]
            
            return NodeResult(
                node_id=self.node_id,
                status=NodeStatus.COMPLETED,
                data=result_data,
                execution_time=time.time() - start_time,
                next_nodes=next_nodes
            )
            
        except Exception as e:
            return NodeResult(
                node_id=self.node_id,
                status=NodeStatus.FAILED,
                data={},
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    def _should_search(self, context: GraphContext) -> bool:
        """Determine if search is needed for this query"""
        query_lower = context.query.lower()
        
        # Search indicators
        search_keywords = [
            "latest", "recent", "current", "news", "today", "2024", "2025",
            "what is", "who is", "how to", "where is", "when did",
            "compare", "review", "price", "cost", "buy"
        ]
        
        # Don't search for these
        no_search_keywords = [
            "hello", "hi", "thanks", "thank you", "help me code",
            "explain", "define", "calculate", "solve"
        ]
        
        if any(keyword in query_lower for keyword in no_search_keywords):
            return False
            
        if any(keyword in query_lower for keyword in search_keywords):
            return True
            
        # Default: search for questions and complex queries
        return len(context.query.split()) > 3 or "?" in context.query
    
    def _get_routing_decision(self, context: GraphContext) -> Dict[str, Any]:
        """Get routing decision based on context"""
        if context.user_budget >= 1.50 and context.quality_requirement == "premium":
            return {
                "provider": "brave",
                "enhance_content": True,
                "max_results": context.max_results
            }
        elif context.user_budget >= 0.50:
            return {
                "provider": "brave", 
                "enhance_content": False,
                "max_results": context.max_results
            }
        else:
            return {
                "provider": "duckduckgo",
                "enhance_content": False,
                "max_results": min(context.max_results, 5)
            }
    
    def _estimate_cost(self, context: GraphContext) -> float:
        """Estimate search cost"""
        decision = self._get_routing_decision(context)
        
        if decision["provider"] == "brave":
            base_cost = 0.42
            if decision["enhance_content"]:
                base_cost += 0.84 * min(3, context.max_results)  # Enhance top 3
            return base_cost
        else:
            return 0.0  # DuckDuckGo is free

class PrimarySearchNode(GraphNode):
    """Primary search execution node"""
    
    def __init__(self, node_id: str = "primary_search", search_router: SmartSearchRouter = None):
        super().__init__(node_id, NodeType.SEARCH)
        self.search_router = search_router
        
    async def execute(self, context: GraphContext) -> NodeResult:
        start_time = time.time()
        
        try:
            if not self.search_router:
                raise Exception("Search router not initialized")
            
            # Execute search
            search_response = await self.search_router.search(
                query=context.query,
                budget=context.user_budget,
                quality_requirement=context.quality_requirement,
                max_results=context.max_results
            )
            
            # Update context
            context.search_results = search_response.results
            context.total_cost += search_response.total_cost
            
            result_data = {
                "search_response": search_response.to_dict(),
                "results_count": len(search_response.results),
                "search_time": search_response.search_time,
                "cost": search_response.total_cost,
                "provider": search_response.provider_used,
                "enhanced": search_response.enhanced
            }
            
            next_nodes = ["analyze_results"] if search_response.results else ["no_results_handler"]
            
            return NodeResult(
                node_id=self.node_id,
                status=NodeStatus.COMPLETED,
                data=result_data,
                execution_time=time.time() - start_time,
                next_nodes=next_nodes
            )
            
        except Exception as e:
            logger.error(f"Primary search failed: {str(e)}")
            return NodeResult(
                node_id=self.node_id,
                status=NodeStatus.FAILED,
                data={},
                execution_time=time.time() - start_time,
                error=str(e),
                next_nodes=["fallback_search"]
            )

class AnalyzeResultsNode(GraphNode):
    """Analyze and score search results"""
    
    def __init__(self, node_id: str = "analyze_results"):
        super().__init__(node_id, NodeType.ANALYZE)
        
    async def execute(self, context: GraphContext) -> NodeResult:
        start_time = time.time()
        
        try:
            analysis = {
                "total_results": len(context.search_results),
                "avg_confidence": self._calculate_avg_confidence(context.search_results),
                "source_diversity": self._analyze_source_diversity(context.search_results),
                "content_quality": self._assess_content_quality(context.search_results),
                "relevance_scores": self._calculate_relevance_scores(context.query, context.search_results),
                "credibility_assessment": self._assess_source_credibility(context.search_results)
            }
            
            # Update context
            context.analysis = analysis
            
            # Determine next steps
            next_nodes = []
            if analysis["avg_confidence"] < 0.6:
                next_nodes.append("enhancement_needed")
            
            if analysis["total_results"] > 0:
                next_nodes.append("synthesize_response")
            else:
                next_nodes.append("no_results_handler")
            
            return NodeResult(
                node_id=self.node_id,
                status=NodeStatus.COMPLETED,
                data=analysis,
                execution_time=time.time() - start_time,
                next_nodes=next_nodes
            )
            
        except Exception as e:
            return NodeResult(
                node_id=self.node_id,
                status=NodeStatus.FAILED,
                data={},
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    def _calculate_avg_confidence(self, results: List[SearchResult]) -> float:
        """Calculate average confidence score"""
        if not results:
            return 0.0
        return sum(r.confidence_score for r in results) / len(results)
    
    def _analyze_source_diversity(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Analyze diversity of sources"""
        domains = set()
        for result in results:
            try:
                from urllib.parse import urlparse
                domain = urlparse(result.url).netloc
                domains.add(domain)
            except:
                continue
                
        return {
            "unique_domains": len(domains),
            "diversity_score": len(domains) / len(results) if results else 0,
            "domains": list(domains)
        }
    
    def _assess_content_quality(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Assess overall content quality"""
        enhanced_count = sum(1 for r in results if r.enhanced_content)
        snippet_quality = sum(1 for r in results if len(r.snippet) > 100)
        
        return {
            "enhanced_results": enhanced_count,
            "quality_snippets": snippet_quality,
            "avg_snippet_length": sum(len(r.snippet) for r in results) / len(results) if results else 0
        }
    
    def _calculate_relevance_scores(self, query: str, results: List[SearchResult]) -> List[float]:
        """Calculate relevance scores for results"""
        query_terms = set(query.lower().split())
        scores = []
        
        for result in results:
            # Simple relevance scoring
            title_terms = set(result.title.lower().split())
            snippet_terms = set(result.snippet.lower().split())
            
            title_overlap = len(query_terms.intersection(title_terms)) / len(query_terms)
            snippet_overlap = len(query_terms.intersection(snippet_terms)) / len(query_terms)
            
            relevance_score = (title_overlap * 0.7 + snippet_overlap * 0.3)
            scores.append(relevance_score)
            
        return scores
    
    def _assess_source_credibility(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Assess credibility of sources"""
        credible_domains = ['.edu', '.gov', '.org']
        news_domains = ['bbc.com', 'reuters.com', 'ap.org', 'cnn.com']
        
        credible_count = 0
        news_count = 0
        
        for result in results:
            url_lower = result.url.lower()
            if any(domain in url_lower for domain in credible_domains):
                credible_count += 1
            if any(domain in url_lower for domain in news_domains):
                news_count += 1
        
        return {
            "credible_sources": credible_count,
            "news_sources": news_count,
            "credibility_ratio": credible_count / len(results) if results else 0,
            "news_ratio": news_count / len(results) if results else 0
        }

class SynthesizeResponseNode(GraphNode):
    """Synthesize final response with citations"""
    
    def __init__(self, node_id: str = "synthesize_response"):
        super().__init__(node_id, NodeType.SYNTHESIZE)
        
    async def execute(self, context: GraphContext) -> NodeResult:
        start_time = time.time()
        
        try:
            # Create comprehensive response with citations
            response_data = {
                "synthesized_response": self._create_response(context),
                "citations": self._generate_citations(context),
                "confidence_score": self._calculate_overall_confidence(context),
                "source_summary": self._summarize_sources(context),
                "search_metadata": {
                    "total_cost": context.total_cost,
                    "search_time": context.execution_time,
                    "results_analyzed": len(context.search_results),
                    "quality_score": context.analysis.get("avg_confidence", 0.0)
                }
            }
            
            # Update context with final data
            context.citations = response_data["citations"]
            
            return NodeResult(
                node_id=self.node_id,
                status=NodeStatus.COMPLETED,
                data=response_data,
                execution_time=time.time() - start_time,
                next_nodes=["finalize_response"]
            )
            
        except Exception as e:
            return NodeResult(
                node_id=self.node_id,
                status=NodeStatus.FAILED,
                data={},
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    def _create_response(self, context: GraphContext) -> str:
        """Create synthesized response from search results"""
        if not context.search_results:
            return "I couldn't find any relevant information for your query."
        
        # Get top results
        top_results = sorted(
            context.search_results, 
            key=lambda x: x.confidence_score, 
            reverse=True
        )[:5]
        
        # Create response
        response_parts = []
        response_parts.append(f"Based on my search, here's what I found about '{context.query}':")
        
        # Add key findings
        for i, result in enumerate(top_results[:3]):
            if result.enhanced_content:
                # Use enhanced content if available
                content = result.enhanced_content[:300] + "..."
            else:
                content = result.snippet
            
            response_parts.append(f"\n{i+1}. **{result.title}**")
            response_parts.append(f"   {content}")
            response_parts.append(f"   *Source: {result.url}*")
        
        # Add summary
        if len(context.search_results) > 3:
            response_parts.append(f"\nI found {len(context.search_results)} total results from {context.analysis.get('source_diversity', {}).get('unique_domains', 'multiple')} different sources.")
        
        return "\n".join(response_parts)
    
    def _generate_citations(self, context: GraphContext) -> List[Dict[str, Any]]:
        """Generate proper citations for all sources"""
        citations = []
        
        for i, result in enumerate(context.search_results):
            citation = {
                "id": i + 1,
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "confidence": result.confidence_score,
                "provider": result.provider,
                "accessed_at": time.time(),
                "enhanced": bool(result.enhanced_content)
            }
            citations.append(citation)
        
        return citations
    
    def _calculate_overall_confidence(self, context: GraphContext) -> float:
        """Calculate overall confidence in the response"""
        if not context.search_results:
            return 0.0
        
        # Weight by result position and confidence
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for i, result in enumerate(context.search_results):
            weight = 1.0 / (i + 1)  # Diminishing weight by position
            weighted_confidence += result.confidence_score * weight
            total_weight += weight
        
        base_confidence = weighted_confidence / total_weight
        
        # Adjust based on source diversity and credibility
        diversity_bonus = min(0.1, context.analysis.get("source_diversity", {}).get("diversity_score", 0) * 0.1)
        credibility_bonus = min(0.1, context.analysis.get("credibility_assessment", {}).get("credibility_ratio", 0) * 0.1)
        
        return min(1.0, base_confidence + diversity_bonus + credibility_bonus)
    
    def _summarize_sources(self, context: GraphContext) -> Dict[str, Any]:
        """Summarize the sources used"""
        source_summary = {
            "total_sources": len(context.search_results),
            "enhanced_sources": sum(1 for r in context.search_results if r.enhanced_content),
            "providers_used": list(set(r.provider for r in context.search_results)),
            "domains": context.analysis.get("source_diversity", {}).get("domains", []),
            "avg_confidence": context.analysis.get("avg_confidence", 0.0)
        }
        
        return source_summary

class FallbackSearchNode(GraphNode):
    """Fallback search when primary search fails"""
    
    def __init__(self, node_id: str = "fallback_search"):
        super().__init__(node_id, NodeType.SEARCH)
        
    async def execute(self, context: GraphContext) -> NodeResult:
        start_time = time.time()
        
        try:
            # Simplified fallback search logic
            # In practice, this might use DuckDuckGo or cached results
            
            fallback_result = SearchResult(
                title=f"Fallback result for: {context.query}",
                url="https://example.com/fallback",
                snippet="This is a fallback response when primary search fails.",
                provider="fallback",
                confidence_score=0.3,
                cost=0.0
            )
            
            context.search_results = [fallback_result]
            
            return NodeResult(
                node_id=self.node_id,
                status=NodeStatus.COMPLETED,
                data={"fallback_used": True},
                execution_time=time.time() - start_time,
                next_nodes=["synthesize_response"]
            )
            
        except Exception as e:
            return NodeResult(
                node_id=self.node_id,
                status=NodeStatus.FAILED,
                data={},
                execution_time=time.time() - start_time,
                error=str(e)
            )

class DirectResponseNode(GraphNode):
    """Direct response node for queries that don't need search"""
    
    def __init__(self, node_id: str = "direct_response"):
        super().__init__(node_id, NodeType.SYNTHESIZE)
        
    async def execute(self, context: GraphContext) -> NodeResult:
        start_time = time.time()
        
        try:
            # Generate direct response without search
            response_data = {
                "synthesized_response": self._create_direct_response(context),
                "citations": [],
                "confidence_score": 0.8,  # High confidence for direct responses
                "source_summary": {"total_sources": 0, "search_bypassed": True},
                "search_metadata": {
                    "total_cost": 0.0,
                    "search_time": 0.0,
                    "results_analyzed": 0,
                    "search_bypassed": True
                }
            }
            
            return NodeResult(
                node_id=self.node_id,
                status=NodeStatus.COMPLETED,
                data=response_data,
                execution_time=time.time() - start_time,
                next_nodes=["finalize_response"]
            )
            
        except Exception as e:
            return NodeResult(
                node_id=self.node_id,
                status=NodeStatus.FAILED,
                data={},
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    def _create_direct_response(self, context: GraphContext) -> str:
        """Create direct response without search"""
        query_lower = context.query.lower()
        
        # Handle greetings
        if any(greeting in query_lower for greeting in ["hello", "hi", "hey"]):
            return "Hello! I'm your AI search assistant. I can help you find information on any topic. What would you like to know about?"
        
        # Handle thanks
        if any(thanks in query_lower for thanks in ["thank", "thanks"]):
            return "You're welcome! Is there anything else I can help you search for?"
        
        # Handle coding help requests
        if any(code in query_lower for code in ["help me code", "programming", "code"]):
            return "I'd be happy to help with programming! What specific coding question or concept would you like assistance with?"
        
        # Default response for other direct queries
        return f"I understand you're asking about '{context.query}'. For the most accurate and up-to-date information, I can search for current resources. Would you like me to search for this topic?"

class NoResultsHandlerNode(GraphNode):
    """Handle cases where search returns no results"""
    
    def __init__(self, node_id: str = "no_results_handler"):
        super().__init__(node_id, NodeType.SYNTHESIZE)
        
    async def execute(self, context: GraphContext) -> NodeResult:
        start_time = time.time()
        
        try:
            response_data = {
                "synthesized_response": self._create_no_results_response(context),
                "citations": [],
                "confidence_score": 0.1,
                "source_summary": {
                    "total_sources": 0,
                    "no_results_found": True,
                    "search_attempted": True
                },
                "search_metadata": {
                    "total_cost": context.total_cost,
                    "search_time": context.execution_time,
                    "results_analyzed": 0,
                    "no_results": True
                }
            }
            
            return NodeResult(
                node_id=self.node_id,
                status=NodeStatus.COMPLETED,
                data=response_data,
                execution_time=time.time() - start_time,
                next_nodes=["finalize_response"]
            )
            
        except Exception as e:
            return NodeResult(
                node_id=self.node_id,
                status=NodeStatus.FAILED,
                data={},
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    def _create_no_results_response(self, context: GraphContext) -> str:
        """Create response when no search results are found"""
        suggestions = [
            "Try using different keywords or phrases",
            "Check for spelling errors in your query",
            "Make your query more specific or more general",
            "Try searching for related topics"
        ]
        
        response = f"I couldn't find any relevant results for '{context.query}'. "
        response += "Here are some suggestions to improve your search:\n\n"
        
        for i, suggestion in enumerate(suggestions, 1):
            response += f"{i}. {suggestion}\n"
        
        response += "\nWould you like to try a different search query?"
        
        return response