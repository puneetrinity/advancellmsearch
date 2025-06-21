# app/graphs/search_graph.py
"""
SearchGraph Implementation - Clean Web Search with Brave + ScrapingBee
Handles intelligent search routing, content enhancement, and response synthesis
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import structlog

from app.graphs.base import (
    BaseGraph, BaseGraphNode, GraphState, NodeResult, 
    GraphType, StartNode, EndNode, ErrorHandlerNode
)
from app.models.manager import ModelManager
from app.cache.redis_client import CacheManager
from app.core.config import get_settings

# Import standardized providers
from app.providers.brave_search_provider import (
    BraveSearchProvider, 
    ProviderConfig as BraveConfig, 
    SearchQuery as BraveSearchQuery,
    SearchResult as BraveSearchResult
)
from app.providers.scrapingbee_provider import (
    ScrapingBeeProvider,
    ProviderConfig as ScrapingBeeConfig, 
    ScrapingQuery
)

logger = structlog.get_logger(__name__)


@dataclass
class EnhancedSearchResult:
    """Enhanced search result with content and analysis"""
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float = 0.0
    content: str = ""
    content_quality: str = "basic"  # basic, enhanced, analyzed
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @classmethod
    def from_brave_result(cls, brave_result: BraveSearchResult) -> 'EnhancedSearchResult':
        """Convert BraveSearchResult to EnhancedSearchResult"""
        return cls(
            title=brave_result.title,
            url=brave_result.url,
            snippet=brave_result.snippet,
            source=brave_result.source,
            relevance_score=brave_result.relevance_score,
            metadata=brave_result.metadata.copy()
        )


class SmartSearchRouterNode(BaseGraphNode):
    """Intelligent search routing based on query type and budget"""
    
    def __init__(self):
        super().__init__("smart_router", "decision")
    
    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        """Route search based on query analysis and budget"""
        try:
            query = state.original_query
            budget = state.cost_budget_remaining
            quality = state.quality_requirement
            
            # Analyze query to determine search strategy
            search_strategy = self._determine_search_strategy(query, budget, quality)
            
            # Store strategy in state
            state.intermediate_results["search_strategy"] = search_strategy
            
            # Determine next nodes based on strategy
            if search_strategy["skip_search"]:
                next_nodes = ["direct_response"]
            else:
                next_nodes = ["brave_search"]
            
            return NodeResult(
                success=True,
                confidence=0.9,
                data={
                    "strategy": search_strategy,
                    "estimated_cost": search_strategy["estimated_cost"],
                    "route_decision": search_strategy["route"]
                },
                cost=0.0,
                next_nodes=next_nodes
            )
            
        except Exception as e:
            return NodeResult(
                success=False,
                error=f"Router failed: {str(e)}",
                confidence=0.0
            )
    
    def _determine_search_strategy(self, query: str, budget: float, quality: str) -> Dict[str, Any]:
        """Determine optimal search strategy"""
        query_lower = query.lower()
        
        # Check if search is needed
        no_search_patterns = [
            "hello", "hi", "thanks", "thank you", 
            "help me code", "explain", "define", "calculate"
        ]
        
        if any(pattern in query_lower for pattern in no_search_patterns):
            return {
                "route": "direct",
                "skip_search": True,
                "use_brave": False,
                "use_scraping": False,
                "estimated_cost": 0.0,
                "reason": "Direct response sufficient"
            }
        
        # Determine search route based on budget and quality
        if quality == "premium" and budget >= 1.50:
            return {
                "route": "premium",
                "skip_search": False,
                "use_brave": True,
                "use_scraping": True,
                "max_scrape": 3,
                "estimated_cost": 0.426,  # 0.42 + (3 * 0.002)
                "reason": "Premium quality with content enhancement"
            }
        elif budget >= 0.50:
            return {
                "route": "standard",
                "skip_search": False,
                "use_brave": True,
                "use_scraping": False,
                "max_scrape": 0,
                "estimated_cost": 0.42,
                "reason": "Standard Brave search only"
            }
        else:
            return {
                "route": "fallback",
                "skip_search": False,
                "use_brave": False,
                "use_scraping": False,
                "estimated_cost": 0.0,
                "reason": "Budget-constrained fallback"
            }


class BraveSearchNode(BaseGraphNode):
    """Brave Search execution with standardized provider"""
    
    def __init__(self, cache_manager: CacheManager):
        super().__init__("brave_search", "processing")
        self.cache_manager = cache_manager
        self.settings = get_settings()
        
        # Initialize Brave provider
        self.provider = None
        self._initialized = False
    
    async def _ensure_provider_initialized(self):
        """Lazy initialization of provider"""
        if not self._initialized:
            config = BraveConfig(
                api_key=getattr(self.settings, 'brave_search_api_key', None) or 
                       getattr(self.settings, 'BRAVE_API_KEY', None),
                cost_per_request=0.008,
                timeout=30,
                max_retries=2
            )
            
            self.provider = BraveSearchProvider(config, logger)
            await self.provider.initialize()
            self._initialized = True
    
    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        """Execute Brave search with caching"""
        try:
            await self._ensure_provider_initialized()
            
            if not self.provider.is_available():
                return NodeResult(
                    success=False,
                    error="Brave Search provider not available",
                    confidence=0.0
                )
            
            query = state.original_query
            max_results = kwargs.get("max_results", 10)
            
            # Check cache first
            cache_key = f"brave_search:{hash(query)}:{max_results}"
            cached_results = await self.cache_manager.get(cache_key)
            
            if cached_results:
                # Convert cached results to EnhancedSearchResult
                enhanced_results = [
                    EnhancedSearchResult(**result) for result in cached_results
                ]
                state.search_results = enhanced_results
                
                return NodeResult(
                    success=True,
                    confidence=0.9,
                    data={
                        "results_count": len(enhanced_results),
                        "cached": True,
                        "provider": "brave_search"
                    },
                    cost=0.0
                )
            
            # Execute search
            brave_query = BraveSearchQuery(
                text=query,
                max_results=max_results,
                language="en",
                search_type="web"
            )
            
            provider_result = await self.provider.search(brave_query)
            
            if not provider_result.success:
                return NodeResult(
                    success=False,
                    error=provider_result.error,
                    confidence=0.0
                )
            
            # Convert to EnhancedSearchResult
            enhanced_results = [
                EnhancedSearchResult.from_brave_result(result) 
                for result in provider_result.data
            ]
            
            # Cache results
            cache_data = [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "source": r.source,
                    "relevance_score": r.relevance_score,
                    "content": r.content,
                    "content_quality": r.content_quality,
                    "metadata": r.metadata
                }
                for r in enhanced_results
            ]
            await self.cache_manager.set(cache_key, cache_data, ttl=1800)  # 30 min
            
            # Store results in state
            state.search_results = enhanced_results
            
            # Get provider stats
            stats = self.provider.get_stats()
            
            return NodeResult(
                success=True,
                confidence=0.85,
                data={
                    "results_count": len(enhanced_results),
                    "cached": False,
                    "provider": "brave_search",
                    "provider_stats": stats
                },
                cost=provider_result.cost
            )
            
        except Exception as e:
            logger.error("Brave search failed", error=str(e))
            return NodeResult(
                success=False,
                error=f"Brave search failed: {str(e)}",
                confidence=0.0
            )
    
    async def cleanup(self):
        """Cleanup provider resources"""
        if self.provider:
            await self.provider.cleanup()


class ContentEnhancementNode(BaseGraphNode):
    """Content enhancement using ScrapingBee for premium results"""
    
    def __init__(self, cache_manager: CacheManager):
        super().__init__("content_enhancement", "processing")
        self.cache_manager = cache_manager
        self.settings = get_settings()
        
        # Initialize ScrapingBee provider
        self.provider = None
        self._initialized = False
    
    async def _ensure_provider_initialized(self):
        """Lazy initialization of provider"""
        if not self._initialized:
            config = ScrapingBeeConfig(
                api_key=getattr(self.settings, 'scrapingbee_api_key', None) or 
                       getattr(self.settings, 'SCRAPINGBEE_API_KEY', None),
                cost_per_request=0.002,
                timeout=60,
                max_retries=2
            )
            
            self.provider = ScrapingBeeProvider(config, logger)
            await self.provider.initialize()
            self._initialized = True
    
    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        """Enhance search results with full content"""
        try:
            # Check if enhancement is enabled
            strategy = state.intermediate_results.get("search_strategy", {})
            if not strategy.get("use_scraping", False):
                return NodeResult(
                    success=True,
                    confidence=1.0,
                    data={"enhancement_skipped": True, "reason": "Not enabled in strategy"},
                    cost=0.0
                )
            
            if not state.search_results:
                return NodeResult(
                    success=False,
                    error="No search results available for enhancement",
                    confidence=0.0
                )
            
            await self._ensure_provider_initialized()
            
            if not self.provider.is_available():
                logger.warning("ScrapingBee not available, skipping enhancement")
                return NodeResult(
                    success=True,
                    confidence=0.7,
                    data={"enhancement_skipped": True, "reason": "Provider unavailable"},
                    cost=0.0
                )
            
            # Select top results for enhancement
            max_enhance = strategy.get("max_scrape", 3)
            top_results = state.search_results[:max_enhance]
            
            # Create scraping queries
            scraping_queries = []
            for result in top_results:
                scraping_queries.append(ScrapingQuery(
                    url=result.url,
                    render_js=True,
                    extract_rules={
                        "headings": "h1, h2, h3",
                        "paragraphs": "p",
                        "main_content": "main, article, .content"
                    }
                ))
            
            # Execute scraping
            scraping_results = await self.provider.scrape_multiple(
                scraping_queries, 
                max_concurrent=3
            )
            
            # Process results
            enhanced_count = 0
            total_content_length = 0
            
            for i, scraping_result in enumerate(scraping_results):
                if scraping_result.success and i < len(top_results):
                    # Extract clean content
                    content = scraping_result.data.text[:5000]  # Limit content size
                    top_results[i].content = content
                    top_results[i].content_quality = "enhanced"
                    top_results[i].metadata.update({
                        "enhanced": True,
                        "content_length": len(content),
                        "extraction_rules_used": True
                    })
                    
                    enhanced_count += 1
                    total_content_length += len(content)
                else:
                    logger.warning(f"Failed to enhance content for {top_results[i].url}")
            
            # Get provider stats
            stats = self.provider.get_stats()
            
            return NodeResult(
                success=enhanced_count > 0,
                confidence=enhanced_count / len(top_results) if top_results else 0.0,
                data={
                    "enhanced_count": enhanced_count,
                    "total_results": len(top_results),
                    "total_content_length": total_content_length,
                    "provider_stats": stats
                },
                cost=stats.get("total_cost", 0.0)
            )
            
        except Exception as e:
            logger.error("Content enhancement failed", error=str(e))
            return NodeResult(
                success=False,
                error=f"Content enhancement failed: {str(e)}",
                confidence=0.0
            )
    
    async def cleanup(self):
        """Cleanup provider resources"""
        if self.provider:
            await self.provider.cleanup()


class ResponseSynthesisNode(BaseGraphNode):
    """Synthesize final response with citations and analysis"""
    
    def __init__(self, model_manager: ModelManager):
        super().__init__("response_synthesis", "processing")
        self.model_manager = model_manager
    
    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        """Synthesize comprehensive response from search results"""
        try:
            if not state.search_results:
                return NodeResult(
                    success=False,
                    error="No search results available for synthesis",
                    confidence=0.0
                )
            
            query = state.original_query
            results = state.search_results
            quality = state.quality_requirement
            
            # Build synthesis prompt
            synthesis_prompt = self._build_synthesis_prompt(query, results, quality)
            
            # Select model based on quality requirement
            model_name = self._select_model(quality)
            
            # Generate response
            model_result = await self.model_manager.generate(
                model_name=model_name,
                prompt=synthesis_prompt,
                max_tokens=800,
                temperature=0.4
            )
            
            if model_result.success:
                response_text = model_result.text.strip()
                confidence = 0.85
                cost = model_result.cost
            else:
                # Fallback response
                response_text = self._generate_fallback_response(query, results)
                confidence = 0.6
                cost = 0.0
            
            # Add citations
            response_with_citations = self._add_citations(response_text, results)
            
            # Calculate response quality metrics
            quality_metrics = self._calculate_quality_metrics(results)
            
            # Store final response
            state.final_response = response_with_citations
            state.response_metadata = {
                "search_results_count": len(results),
                "enhanced_results_count": sum(1 for r in results if r.content_quality == "enhanced"),
                "sources_used": [r.url for r in results[:5]],
                "synthesis_model": model_name,
                "quality_metrics": quality_metrics
            }
            
            return NodeResult(
                success=True,
                confidence=confidence,
                data={
                    "response": response_with_citations,
                    "sources_count": len(results),
                    "enhanced_sources": sum(1 for r in results if r.content),
                    "synthesis_model": model_name,
                    "quality_metrics": quality_metrics
                },
                cost=cost,
                model_used=model_name
            )
            
        except Exception as e:
            logger.error("Response synthesis failed", error=str(e))
            return NodeResult(
                success=False,
                error=f"Response synthesis failed: {str(e)}",
                confidence=0.0
            )
    
    def _build_synthesis_prompt(self, query: str, results: List[EnhancedSearchResult], quality: str) -> str:
        """Build comprehensive synthesis prompt"""
        # Prepare source information
        sources_info = []
        for i, result in enumerate(results[:5]):  # Top 5 sources
            source_content = result.content if result.content else result.snippet
            content_preview = source_content[:800] if len(source_content) > 800 else source_content
            
            sources_info.append(f"""Source {i+1}: {result.title}
URL: {result.url}
Content: {content_preview}
Quality: {result.content_quality}
""")
        
        sources_text = "\n---\n".join(sources_info)
        
        # Quality-based instructions
        if quality == "premium":
            style_instruction = "Provide a comprehensive, detailed, and well-structured response with in-depth analysis."
        elif quality == "high":
            style_instruction = "Provide a thorough and informative response with good detail."
        else:
            style_instruction = "Provide a clear and informative response."
        
        prompt = f"""You are an AI assistant providing information based on web search results. Synthesize the following sources to answer the user's question accurately and comprehensively.

User Question: "{query}"

Instructions: {style_instruction}

Search Results:
{sources_text}

Guidelines:
1. Provide a clear, accurate answer based on the search results
2. Integrate information from multiple sources when possible
3. Be objective and factual
4. If sources conflict, mention the different perspectives
5. Use the enhanced content when available for deeper insights
6. Structure your response logically with clear sections if appropriate
7. Do not include URLs in the response (citations will be added separately)

Response:"""
        
        return prompt
    
    def _select_model(self, quality: str) -> str:
        """Select appropriate model based on quality requirement"""
        model_mapping = {
            "premium": "llama2:13b",
            "high": "mistral:7b", 
            "balanced": "llama2:7b",
            "minimal": "phi:mini"
        }
        return model_mapping.get(quality, "llama2:7b")
    
    def _generate_fallback_response(self, query: str, results: List[EnhancedSearchResult]) -> str:
        """Generate simple fallback response"""
        if not results:
            return f"I searched for information about '{query}' but couldn't generate a comprehensive response."
        
        top_result = results[0]
        response = f"Based on my search about '{query}', here's what I found:\n\n"
        response += top_result.snippet
        
        if len(results) > 1:
            response += f"\n\nI found {len(results)} relevant sources with additional information on this topic."
        
        return response
    
    def _add_citations(self, response: str, results: List[EnhancedSearchResult]) -> str:
        """Add formatted citations to response"""
        if not results:
            return response
        
        citations = "\n\n**Sources:**\n"
        for i, result in enumerate(results[:5], 1):
            enhanced_indicator = " ⭐" if result.content_quality == "enhanced" else ""
            citations += f"{i}. {result.title}{enhanced_indicator}\n   {result.url}\n"
        
        return response + citations
    
    def _calculate_quality_metrics(self, results: List[EnhancedSearchResult]) -> Dict[str, Any]:
        """Calculate response quality metrics"""
        if not results:
            return {"overall_quality": "poor", "score": 0.0}
        
        enhanced_count = sum(1 for r in results if r.content_quality == "enhanced")
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        
        # Calculate overall quality score
        enhancement_score = enhanced_count / len(results)
        relevance_score = avg_relevance
        source_diversity_score = len(set(r.url.split('/')[2] for r in results)) / len(results)
        
        overall_score = (enhancement_score * 0.4 + relevance_score * 0.4 + source_diversity_score * 0.2)
        
        if overall_score >= 0.8:
            quality_label = "excellent"
        elif overall_score >= 0.6:
            quality_label = "good"
        elif overall_score >= 0.4:
            quality_label = "fair"
        else:
            quality_label = "poor"
        
        return {
            "overall_quality": quality_label,
            "score": overall_score,
            "enhanced_ratio": enhancement_score,
            "avg_relevance": relevance_score,
            "source_diversity": source_diversity_score,
            "total_sources": len(results)
        }


class DirectResponseNode(BaseGraphNode):
    """Direct response for queries that don't need search"""
    
    def __init__(self, model_manager: ModelManager):
        super().__init__("direct_response", "processing")
        self.model_manager = model_manager
    
    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        """Generate direct response without search"""
        try:
            query = state.original_query
            
            # Check for simple patterns first
            simple_response = self._get_simple_response(query)
            if simple_response:
                state.final_response = simple_response
                return NodeResult(
                    success=True,
                    confidence=0.9,
                    data={"response": simple_response, "type": "simple"},
                    cost=0.0
                )
            
            # Use model for more complex direct responses
            response_prompt = f"""Provide a helpful direct response to this query: "{query}"

Since this doesn't require web search, give a conversational and informative response based on your knowledge.

Response:"""
            
            model_result = await self.model_manager.generate(
                model_name="phi:mini",  # Fast model for direct responses
                prompt=response_prompt,
                max_tokens=300,
                temperature=0.6
            )
            
            if model_result.success:
                response = model_result.text.strip()
                confidence = 0.8
                cost = model_result.cost
            else:
                response = "I'd be happy to help! Could you provide more details about what you're looking for?"
                confidence = 0.5
                cost = 0.0
            
            state.final_response = response
            
            return NodeResult(
                success=True,
                confidence=confidence,
                data={"response": response, "type": "model_generated"},
                cost=cost,
                model_used="phi:mini" if model_result.success else None
            )
            
        except Exception as e:
            return NodeResult(
                success=False,
                error=f"Direct response failed: {str(e)}",
                confidence=0.0
            )
    
    def _get_simple_response(self, query: str) -> Optional[str]:
        """Get simple response for common patterns"""
        query_lower = query.lower()
        
        if any(greeting in query_lower for greeting in ["hello", "hi", "hey"]):
            return "Hello! I'm your AI search assistant. I can help you find information on any topic or answer questions directly. What would you like to know about?"
        
        if any(thanks in query_lower for thanks in ["thank", "thanks"]):
            return "You're welcome! Is there anything else I can help you search for or answer?"
        
        if "help me code" in query_lower or "programming help" in query_lower:
            return "I'd be happy to help with programming! What specific coding question, language, or concept would you like assistance with?"
        
        return None


class SearchGraph(BaseGraph):
    """Main search graph implementation with standardized providers"""
    
    def __init__(self, model_manager: ModelManager, cache_manager: CacheManager):
        super().__init__(GraphType.SEARCH, "enhanced_search_graph")
        self.model_manager = model_manager
        self.cache_manager = cache_manager
        self._node_instances = {}
    
    def define_nodes(self) -> Dict[str, BaseGraphNode]:
        """Define search graph nodes"""
        if not self._node_instances:
            self._node_instances = {
                "start": StartNode(),
                "smart_router": SmartSearchRouterNode(),
                "brave_search": BraveSearchNode(self.cache_manager),
                "content_enhancement": ContentEnhancementNode(self.cache_manager),
                "response_synthesis": ResponseSynthesisNode(self.model_manager),
                "direct_response": DirectResponseNode(self.model_manager),
                "end": EndNode(),
                "error_handler": ErrorHandlerNode()
            }
        return self._node_instances
    
    def define_edges(self) -> List[tuple]:
        """Define search graph edges with smart routing"""
        return [
            # Start with routing decision
            ("start", "smart_router"),
            
            # Search path
            ("smart_router", "brave_search"),
            ("brave_search", "content_enhancement"),
            ("content_enhancement", "response_synthesis"),
            ("response_synthesis", "end"),
            
            # Direct response path
            ("smart_router", "direct_response"),
            ("direct_response", "end"),
            
            # Error handling
            ("smart_router", self._check_routing_errors, {
                "error": "error_handler",
                "search": "brave_search",
                "direct": "direct_response"
            }),
            ("brave_search", self._check_search_results, {
                "no_results": "direct_response",
                "has_results": "content_enhancement"
            }),
            ("content_enhancement", self._check_enhancement_results, {
                "continue": "response_synthesis",
                "error": "response_synthesis"  # Continue even if enhancement fails
            }),
            ("error_handler", "end")
        ]
    
    def _check_routing_errors(self, state: GraphState) -> str:
        """Check routing decision"""
        if state.errors:
            return "error"
        
        strategy = state.intermediate_results.get("search_strategy", {})
        if strategy.get("skip_search", False):
            return "direct"
        else:
            return "search"
    
    def _check_search_results(self, state: GraphState) -> str:
        """Check search results quality"""
        if not state.search_results or len(state.search_results) == 0:
            return "no_results"
        return "has_results"
    
    def _check_enhancement_results(self, state: GraphState) -> str:
        """Always continue to synthesis regardless of enhancement results"""
        return "continue"
    
    async def cleanup(self):
        """Cleanup all node resources"""
        for node in self._node_instances.values():
            if hasattr(node, 'cleanup'):
                try:
                    await node.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up node {node.node_id}: {e}")


# Convenience function for simple search workflow
async def execute_search(
    query: str,
    model_manager: ModelManager,
    cache_manager: CacheManager,
    budget: float = 2.0,
    quality: str = "balanced",
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Execute search workflow with simple interface
    
    Args:
        query: Search query
        model_manager: Model manager instance
        cache_manager: Cache manager instance  
        budget: Cost budget for the search
        quality: Quality requirement (minimal, balanced, high, premium)
        max_results: Maximum number of results
    
    Returns:
        Dict containing response, citations, and metadata
    """
    
    # Create and execute search graph
    search_graph = SearchGraph(model_manager, cache_manager)
    
    try:
        # Create initial state
        state = GraphState(
            original_query=query,
            cost_budget_remaining=budget,
            quality_requirement=quality,
            max_execution_time=30.0
        )
        
        # Execute graph
        start_time = time.time()
        result = await search_graph.execute(state, max_results=max_results)
        execution_time = time.time() - start_time
        
        return {
            "query": query,
            "response": state.final_response or "No response generated",
            "citations": [],  # Will be included in response
            "metadata": {
                "execution_time": execution_time,
                "total_cost": state.calculate_total_cost(),
                "search_results_count": len(state.search_results) if state.search_results else 0,
                "quality_used": quality,
                "budget_used": budget - state.cost_budget_remaining,
                **state.response_metadata
            },
            "success": result.success if result else False
        }
        
    finally:
        # Cleanup resources
        await search_graph.cleanup()