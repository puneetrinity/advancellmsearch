# app/graphs/search_graph.py
"""
SearchGraph Implementation - Web Search and Analysis
Handles information retrieval, web scraping, and content analysis
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import aiohttp
import structlog
from bs4 import BeautifulSoup

from app.graphs.base import (
    BaseGraph, BaseGraphNode, GraphState, NodeResult, 
    GraphType, RoutingCondition, StartNode, EndNode, ErrorHandlerNode
)
from app.models.manager import ModelManager
from app.cache.redis_client import CacheManager
from app.core.config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """Individual search result"""
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float = 0.0
    content: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchQuery:
    """Structured search query"""
    original_query: str
    expanded_queries: List[str]
    search_type: str  # web, academic, news, technical
    max_results: int = 10
    language: str = "en"
    time_filter: Optional[str] = None  # day, week, month, year


class QueryExpanderNode(BaseGraphNode):
    """Expands and optimizes search queries for better results"""
    
    def __init__(self, model_manager: ModelManager):
        super().__init__("query_expander", "processing")
        self.model_manager = model_manager
    
    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        """Expand query for comprehensive search"""
        try:
            original_query = state.original_query
            
            # Use fast model for query expansion
            expansion_prompt = self._build_expansion_prompt(original_query, state)
            
            model_result = await self.model_manager.generate(
                model_name="phi:mini",
                prompt=expansion_prompt,
                max_tokens=150,
                temperature=0.3
            )
            
            if model_result.success:
                expanded_queries = self._parse_expanded_queries(model_result.text, original_query)
                confidence = 0.8
            else:
                # Fallback to rule-based expansion
                expanded_queries = self._fallback_expansion(original_query)
                confidence = 0.6
            
            # Determine search type
            search_type = self._classify_search_type(original_query, state)
            
            # Create search query object
            search_query = SearchQuery(
                original_query=original_query,
                expanded_queries=expanded_queries,
                search_type=search_type,
                max_results=kwargs.get("max_results", 10)
            )
            
            # Store in state
            state.intermediate_results["search_query"] = search_query
            
            return NodeResult(
                success=True,
                confidence=confidence,
                data={
                    "original_query": original_query,
                    "expanded_queries": expanded_queries,
                    "search_type": search_type
                },
                cost=model_result.cost if model_result.success else 0.0,
                model_used="phi:mini" if model_result.success else "fallback"
            )
            
        except Exception as e:
            return NodeResult(
                success=False,
                error=f"Query expansion failed: {str(e)}",
                confidence=0.0
            )
    
    def _build_expansion_prompt(self, query: str, state: GraphState) -> str:
        """Build prompt for query expansion"""
        return f"""Expand this search query to find comprehensive information. Generate 2-3 related search terms that would help find relevant content.

Original query: "{query}"

Provide expanded queries in this format:
1. [first related query]
2. [second related query]  
3. [third related query]

Expanded queries:"""
    
    def _parse_expanded_queries(self, response: str, original: str) -> List[str]:
        """Parse model response into query list"""
        queries = [original]  # Always include original
        
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '-', '*']):
                # Extract query text
                query = line.split('.', 1)[-1].strip() if '.' in line else line.strip('- *')
                if query and query not in queries:
                    queries.append(query)
        
        return queries[:4]  # Limit to 4 queries maximum
    
    def _fallback_expansion(self, query: str) -> List[str]:
        """Rule-based query expansion fallback"""
        queries = [query]
        
        # Add variations based on common patterns
        if "how to" in query.lower():
            queries.append(query.replace("how to", "tutorial"))
            queries.append(query.replace("how to", "guide"))
        elif "what is" in query.lower():
            queries.append(query.replace("what is", "definition"))
            queries.append(query.replace("what is", "explanation"))
        elif "best" in query.lower():
            queries.append(query.replace("best", "top"))
            queries.append(query.replace("best", "recommended"))
        
        return queries[:3]
    
    def _classify_search_type(self, query: str, state: GraphState) -> str:
        """Classify the type of search needed"""
        query_lower = query.lower()
        
        # Check for academic/research indicators
        academic_keywords = ["research", "study", "paper", "academic", "journal", "scientific"]
        if any(keyword in query_lower for keyword in academic_keywords):
            return "academic"
        
        # Check for news/current events
        news_keywords = ["news", "latest", "recent", "current", "today", "breaking"]
        if any(keyword in query_lower for keyword in news_keywords):
            return "news"
        
        # Check for technical content
        tech_keywords = ["api", "documentation", "code", "programming", "tutorial", "github"]
        if any(keyword in query_lower for keyword in tech_keywords):
            return "technical"
        
        return "web"  # Default to general web search


class WebSearchNode(BaseGraphNode):
    """Performs web search using multiple providers"""
    
    def __init__(self, cache_manager: CacheManager):
        super().__init__("web_search", "processing")
        self.cache_manager = cache_manager
        self.settings = get_settings()
        self.search_providers = {
            "brave": BraveSearchProvider(),
            "duckduckgo": DuckDuckGoProvider(),
            "google": GoogleSearchProvider()  # Requires API key
        }
    
    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        """Execute web search across multiple providers"""
        try:
            search_query: SearchQuery = state.intermediate_results.get("search_query")
            if not search_query:
                return NodeResult(
                    success=False,
                    error="No search query found in state",
                    confidence=0.0
                )
            
            # Check cache first
            cache_key = f"search:{hash(str(search_query.expanded_queries))}"
            cached_results = await self.cache_manager.get(cache_key)
            
            if cached_results:
                state.search_results = cached_results
                return NodeResult(
                    success=True,
                    confidence=0.9,
                    data={"results_count": len(cached_results), "cached": True},
                    cost=0.0
                )
            
            # Perform searches
            all_results = []
            search_costs = []
            
            # Select appropriate provider based on search type
            providers = self._select_providers(search_query.search_type)
            
            for provider_name in providers:
                provider = self.search_providers.get(provider_name)
                if provider and provider.is_available():
                    try:
                        results = await provider.search(search_query)
                        all_results.extend(results)
                        search_costs.append(provider.get_cost())
                        
                        # Log successful search
                        self.logger.info(
                            "Search completed",
                            provider=provider_name,
                            results_count=len(results),
                            query_type=search_query.search_type
                        )
                        
                    except Exception as e:
                        self.logger.warning(
                            "Search provider failed",
                            provider=provider_name,
                            error=str(e)
                        )
            
            if not all_results:
                return NodeResult(
                    success=False,
                    error="All search providers failed",
                    confidence=0.0
                )
            
            # Deduplicate and rank results
            unique_results = self._deduplicate_results(all_results)
            ranked_results = self._rank_results(unique_results, search_query.original_query)
            
            # Store results
            state.search_results = ranked_results[:search_query.max_results]
            
            # Cache results
            await self.cache_manager.set(cache_key, state.search_results, ttl=1800)  # 30 minutes
            
            total_cost = sum(search_costs)
            
            return NodeResult(
                success=True,
                confidence=0.8,
                data={
                    "results_count": len(state.search_results),
                    "providers_used": providers,
                    "cached": False
                },
                cost=total_cost
            )
            
        except Exception as e:
            return NodeResult(
                success=False,
                error=f"Web search failed: {str(e)}",
                confidence=0.0
            )
    
    def _select_providers(self, search_type: str) -> List[str]:
        """Select appropriate search providers based on type"""
        provider_map = {
            "web": ["brave", "duckduckgo"],
            "news": ["brave", "google"],
            "academic": ["google"],  # Would integrate with academic APIs
            "technical": ["brave", "duckduckgo"]
        }
        return provider_map.get(search_type, ["brave", "duckduckgo"])
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on URL"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results
    
    def _rank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Rank results by relevance"""
        # Simple ranking based on title/snippet relevance
        query_terms = query.lower().split()
        
        for result in results:
            score = 0.0
            text = f"{result.title} {result.snippet}".lower()
            
            # Count query term matches
            for term in query_terms:
                if term in text:
                    score += 1.0
                    if term in result.title.lower():
                        score += 0.5  # Bonus for title matches
            
            # Normalize by query length
            result.relevance_score = score / len(query_terms) if query_terms else 0.0
        
        # Sort by relevance score
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)


class ContentScrapingNode(BaseGraphNode):
    """Scrapes content from search results"""
    
    def __init__(self, cache_manager: CacheManager):
        super().__init__("content_scraping", "processing")
        self.cache_manager = cache_manager
        self.settings = get_settings()
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        """Scrape content from top search results"""
        try:
            if not state.search_results:
                return NodeResult(
                    success=False,
                    error="No search results to scrape",
                    confidence=0.0
                )
            
            # Initialize HTTP session
            if not self.session:
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10),
                    headers={"User-Agent": "AI-Search-Bot/1.0"}
                )
            
            # Select top results for scraping
            max_scrape = kwargs.get("max_scrape", 5)
            top_results = state.search_results[:max_scrape]
            
            scraping_tasks = []
            for result in top_results:
                task = self._scrape_content(result)
                scraping_tasks.append(task)
            
            # Execute scraping in parallel
            scraped_contents = await asyncio.gather(*scraping_tasks, return_exceptions=True)
            
            # Process results
            successful_scrapes = 0
            total_content_length = 0
            scraping_cost = 0.0
            
            for i, content in enumerate(scraped_contents):
                if isinstance(content, Exception):
                    self.logger.warning(
                        "Content scraping failed",
                        url=top_results[i].url,
                        error=str(content)
                    )
                    continue
                
                if content:
                    top_results[i].content = content
                    successful_scrapes += 1
                    total_content_length += len(content)
                    scraping_cost += 0.002  # Small cost per successful scrape
            
            # Update search results with scraped content
            state.search_results = top_results
            
            return NodeResult(
                success=successful_scrapes > 0,
                confidence=successful_scrapes / len(top_results),
                data={
                    "scraped_count": successful_scrapes,
                    "total_results": len(top_results),
                    "total_content_length": total_content_length
                },
                cost=scraping_cost
            )
            
        except Exception as e:
            return NodeResult(
                success=False,
                error=f"Content scraping failed: {str(e)}",
                confidence=0.0
            )
        finally:
            # Clean up session
            if self.session:
                await self.session.close()
                self.session = None
    
    async def _scrape_content(self, result: SearchResult) -> Optional[str]:
        """Scrape content from a single URL"""
        try:
            # Check cache first
            cache_key = f"content:{hash(result.url)}"
            cached_content = await self.cache_manager.get(cache_key)
            if cached_content:
                return cached_content
            
            # Fetch content
            async with self.session.get(result.url) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                
                # Extract text content using BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                # Limit content length
                max_length = 5000  # 5KB per page
                if len(text) > max_length:
                    text = text[:max_length] + "..."
                
                # Cache the content
                await self.cache_manager.set(cache_key, text, ttl=3600)  # 1 hour
                
                return text
                
        except Exception as e:
            self.logger.warning(f"Failed to scrape {result.url}: {e}")
            return None


class ContentAnalysisNode(BaseGraphNode):
    """Analyzes scraped content for relevance and insights"""
    
    def __init__(self, model_manager: ModelManager):
        super().__init__("content_analysis", "processing")
        self.model_manager = model_manager
    
    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        """Analyze scraped content for insights"""
        try:
            if not state.search_results:
                return NodeResult(
                    success=False,
                    error="No search results to analyze",
                    confidence=0.0
                )
            
            # Filter results with content
            results_with_content = [r for r in state.search_results if r.content]
            
            if not results_with_content:
                return NodeResult(
                    success=False,
                    error="No scraped content available for analysis",
                    confidence=0.0
                )
            
            # Analyze content in batches
            analysis_tasks = []
            for result in results_with_content[:3]:  # Analyze top 3 results
                task = self._analyze_single_content(result, state.original_query)
                analysis_tasks.append(task)
            
            analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process analysis results
            successful_analyses = []
            total_cost = 0.0
            
            for i, analysis in enumerate(analyses):
                if isinstance(analysis, Exception):
                    self.logger.warning(f"Content analysis failed: {analysis}")
                    continue
                
                if analysis and analysis.success:
                    successful_analyses.append(analysis)
                    total_cost += analysis.cost
                    
                    # Store analysis in result metadata
                    results_with_content[i].metadata["analysis"] = analysis.data
            
            if not successful_analyses:
                return NodeResult(
                    success=False,
                    error="All content analyses failed",
                    confidence=0.0
                )
            
            # Generate overall insights
            insights = self._synthesize_insights(successful_analyses, state.original_query)
            
            # Store insights in state
            state.intermediate_results["content_insights"] = insights
            
            return NodeResult(
                success=True,
                confidence=len(successful_analyses) / len(results_with_content),
                data={
                    "analyzed_count": len(successful_analyses),
                    "insights": insights
                },
                cost=total_cost
            )
            
        except Exception as e:
            return NodeResult(
                success=False,
                error=f"Content analysis failed: {str(e)}",
                confidence=0.0
            )
    
    async def _analyze_single_content(self, result: SearchResult, query: str) -> NodeResult:
        """Analyze a single piece of content"""
        try:
            analysis_prompt = f"""Analyze this content for relevance to the query: "{query}"

Content from {result.title}:
{result.content[:2000]}...

Provide analysis in this format:
Relevance: [High/Medium/Low]
Key Points: [2-3 main points]
Summary: [Brief summary in 1-2 sentences]

Analysis:"""
            
            model_result = await self.model_manager.generate(
                model_name="mistral:7b",  # Use reasoning model for analysis
                prompt=analysis_prompt,
                max_tokens=200,
                temperature=0.3
            )
            
            if model_result.success:
                analysis_data = self._parse_analysis(model_result.text)
                return NodeResult(
                    success=True,
                    confidence=0.8,
                    data=analysis_data,
                    cost=model_result.cost
                )
            else:
                return NodeResult(
                    success=False,
                    error="Model analysis failed",
                    confidence=0.0
                )
                
        except Exception as e:
            return NodeResult(
                success=False,
                error=f"Single content analysis failed: {str(e)}",
                confidence=0.0
            )
    
    def _parse_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse model analysis response"""
        lines = analysis_text.strip().split('\n')
        analysis_data = {
            "relevance": "Medium",
            "key_points": [],
            "summary": ""
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith("Relevance:"):
                analysis_data["relevance"] = line.split(":", 1)[1].strip()
            elif line.startswith("Key Points:"):
                points = line.split(":", 1)[1].strip()
                analysis_data["key_points"] = [p.strip() for p in points.split(";") if p.strip()]
            elif line.startswith("Summary:"):
                analysis_data["summary"] = line.split(":", 1)[1].strip()
        
        return analysis_data
    
    def _synthesize_insights(self, analyses: List[NodeResult], query: str) -> Dict[str, Any]:
        """Synthesize insights from multiple content analyses"""
        high_relevance_count = 0
        all_key_points = []
        all_summaries = []
        
        for analysis in analyses:
            data = analysis.data
            if data.get("relevance", "").lower() == "high":
                high_relevance_count += 1
            
            all_key_points.extend(data.get("key_points", []))
            if data.get("summary"):
                all_summaries.append(data["summary"])
        
        return {
            "high_relevance_sources": high_relevance_count,
            "total_sources": len(analyses),
            "key_themes": list(set(all_key_points)),
            "source_summaries": all_summaries,
            "overall_quality": "high" if high_relevance_count >= len(analyses) * 0.7 else "medium"
        }


class ResponseSynthesisNode(BaseGraphNode):
    """Synthesizes final response from search results and analysis"""
    
    def __init__(self, model_manager: ModelManager):
        super().__init__("response_synthesis", "processing")
        self.model_manager = model_manager
    
    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        """Synthesize final response from all gathered information"""
        try:
            # Gather all available information
            search_results = state.search_results or []
            insights = state.intermediate_results.get("content_insights", {})
            original_query = state.original_query
            
            if not search_results:
                return NodeResult(
                    success=False,
                    error="No search results available for synthesis",
                    confidence=0.0
                )
            
            # Build synthesis prompt
            synthesis_prompt = self._build_synthesis_prompt(
                original_query, search_results, insights, state
            )
            
            # Select appropriate model based on quality requirement
            model_name = self._select_synthesis_model(state.quality_requirement)
            
            # Generate response
            model_result = await self.model_manager.generate(
                model_name=model_name,
                prompt=synthesis_prompt,
                max_tokens=800,
                temperature=0.4
            )
            
            if not model_result.success:
                # Fallback to simpler response
                response = self._generate_fallback_response(search_results, original_query)
                confidence = 0.4
                cost = 0.0
            else:
                response = model_result.text.strip()
                confidence = 0.8
                cost = model_result.cost
            
            # Add source citations
            response_with_citations = self._add_citations(response, search_results)
            
            # Store final response
            state.final_response = response_with_citations
            state.response_metadata = {
                "search_results_count": len(search_results),
                "sources_used": [r.url for r in search_results[:3]],
                "content_quality": insights.get("overall_quality", "medium"),
                "synthesis_model": model_name
            }
            
            return NodeResult(
                success=True,
                confidence=confidence,
                data={
                    "response": response_with_citations,
                    "sources_count": len(search_results),
                    "synthesis_model": model_name
                },
                cost=cost,
                model_used=model_name
            )
            
        except Exception as e:
            return NodeResult(
                success=False,
                error=f"Response synthesis failed: {str(e)}",
                confidence=0.0
            )
    
    def _build_synthesis_prompt(
        self, 
        query: str, 
        results: List[SearchResult], 
        insights: Dict[str, Any],
        state: GraphState
    ) -> str:
        """Build prompt for response synthesis"""
        # Compile source information
        sources_info = []
        for i, result in enumerate(results[:3]):  # Top 3 sources
            content_preview = result.content[:500] if result.content else result.snippet
            sources_info.append(f"Source {i+1} ({result.title}):\n{content_preview}")
        
        sources_text = "\n\n".join(sources_info)
        
        # User preference considerations
        style_instruction = ""
        if state.user_preferences.get("response_style") == "detailed":
            style_instruction = "Provide a comprehensive and detailed response."
        elif state.user_preferences.get("response_style") == "concise":
            style_instruction = "Keep the response concise and to the point."
        
        prompt = f"""You are an AI assistant providing information based on web search results. Synthesize the following sources to answer the user's question comprehensively and accurately.

User Question: "{query}"

{style_instruction}

Search Results:
{sources_text}

Key Insights:
- {insights.get('high_relevance_sources', 0)} out of {insights.get('total_sources', 0)} sources are highly relevant
- Overall content quality: {insights.get('overall_quality', 'medium')}

Instructions:
1. Provide a clear, accurate answer based on the search results
2. Integrate information from multiple sources when possible
3. Be objective and factual
4. If sources conflict, mention the different perspectives
5. Don't include URLs in the response (citations will be added separately)

Response:"""
        
        return prompt
    
    def _select_synthesis_model(self, quality_requirement: str) -> str:
        """Select appropriate model for synthesis based on quality requirement"""
        if quality_requirement == "premium":
            return "llama2:13b"
        elif quality_requirement == "high":
            return "mistral:7b"
        else:
            return "llama2:7b"
    
    def _generate_fallback_response(self, results: List[SearchResult], query: str) -> str:
        """Generate simple fallback response from search results"""
        if not results:
            return f"I found some information about '{query}', but couldn't analyze it thoroughly."
        
        # Use the top result's snippet
        top_result = results[0]
        return f"Based on my search, {top_result.snippet} You can find more information from the sources I've found."
    
    def _add_citations(self, response: str, results: List[SearchResult]) -> str:
        """Add source citations to the response"""
        if not results:
            return response
        
        citations = "\n\nSources:\n"
        for i, result in enumerate(results[:3]):  # Cite top 3 sources
            citations += f"{i+1}. {result.title} - {result.url}\n"
        
        return response + citations


class SearchGraph(BaseGraph):
    """Main search graph implementation"""
    
    def __init__(self, model_manager: ModelManager, cache_manager: CacheManager):
        super().__init__(GraphType.SEARCH, "search_graph")
        self.model_manager = model_manager
        self.cache_manager = cache_manager
    
    def define_nodes(self) -> Dict[str, BaseGraphNode]:
        """Define search graph nodes"""
        return {
            "start": StartNode(),
            "query_expander": QueryExpanderNode(self.model_manager),
            "web_search": WebSearchNode(self.cache_manager),
            "content_scraping": ContentScrapingNode(self.cache_manager),
            "content_analysis": ContentAnalysisNode(self.model_manager),
            "response_synthesis": ResponseSynthesisNode(self.model_manager),
            "end": EndNode(),
            "error_handler": ErrorHandlerNode()
        }
    
    def define_edges(self) -> List[tuple]:
        """Define search graph edges"""
        return [
            # Main flow
            ("start", "query_expander"),
            ("query_expander", "web_search"),
            ("web_search", "content_scraping"),
            ("content_scraping", "content_analysis"),
            ("content_analysis", "response_synthesis"),
            ("response_synthesis", "end"),
            
            # Error handling and conditional paths
            ("query_expander", self._check_errors, {
                "error": "error_handler",
                "continue": "web_search"
            }),
            ("web_search", self._check_search_results, {
                "no_results": "error_handler",
                "has_results": "content_scraping"
            }),
            ("content_scraping", self._check_content, {
                "no_content": "response_synthesis",  # Skip analysis if no content
                "has_content": "content_analysis"
            }),
            ("content_analysis", self._check_errors, {
                "error": "response_synthesis",  # Continue to synthesis even if analysis fails
                "continue": "response_synthesis"
            }),
            ("error_handler", "end")
        ]
    
    def _check_errors(self, state: GraphState) -> str:
        """Check for errors in execution"""
        if state.errors:
            return "error"
        return "continue"
    
    def _check_search_results(self, state: GraphState) -> str:
        """Check if search results were found"""
        if not state.search_results:
            return "no_results"
        return "has_results"
    
    def _check_content(self, state: GraphState) -> str:
        """Check if content was scraped"""
        if not state.search_results:
            return "no_content"
        
        has_content = any(result.content for result in state.search_results)
        return "has_content" if has_content else "no_content"


"""
SearchGraph Main Implementation
Main search graph orchestrator that coordinates the search workflow
"""

import asyncio
import time
from typing import Dict, List, Any
import logging

from .nodes.search_nodes import (
    GraphContext,
    SearchRouterNode,
    PrimarySearchNode,
    AnalyzeResultsNode,
    SynthesizeResponseNode,
    FallbackSearchNode,
    NodeStatus
)
from app.providers.router import SmartSearchRouter

logger = logging.getLogger(__name__)

class SearchGraph:
    """Main search graph orchestrator"""
    def __init__(self, search_router: SmartSearchRouter):
        self.search_router = search_router
        self.nodes = {}
        self.execution_results = {}
        self._setup_graph()
    def _setup_graph(self):
        """Setup the search graph with all nodes"""
        self.nodes = {
            "search_router": SearchRouterNode(),
            "primary_search": PrimarySearchNode(search_router=self.search_router),
            "analyze_results": AnalyzeResultsNode(),
            "synthesize_response": SynthesizeResponseNode(),
            "fallback_search": FallbackSearchNode()
        }
        self.nodes["primary_search"].add_dependency("search_router")
        self.nodes["analyze_results"].add_dependency("primary_search")
        self.nodes["synthesize_response"].add_dependency("analyze_results")
        self.nodes["fallback_search"].add_dependency("primary_search")
    async def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        context = GraphContext(
            query=query,
            user_budget=kwargs.get("budget", 2.0),
            quality_requirement=kwargs.get("quality", "standard"),
            max_results=kwargs.get("max_results", 10),
            conversation_history=kwargs.get("history", [])
        )
        logger.info(f"🚀 Starting SearchGraph execution for: '{query}'")
        try:
            execution_path = await self._execute_graph(context)
            context.execution_time = time.time() - start_time
            final_result = {
                "query": query,
                "response": self._get_final_response(context),
                "citations": context.citations,
                "metadata": {
                    "execution_time": context.execution_time,
                    "total_cost": context.total_cost,
                    "execution_path": execution_path,
                    "results_count": len(context.search_results),
                    "confidence_score": self._get_confidence_score(context),
                    "analysis": context.analysis
                },
                "performance_metrics": self._get_performance_metrics(context)
            }
            logger.info(f"✅ SearchGraph completed in {context.execution_time:.2f}s, cost: ₹{context.total_cost:.2f}")
            return final_result
        except Exception as e:
            logger.error(f"❌ SearchGraph execution failed: {str(e)}")
            return {
                "query": query,
                "response": f"I encountered an error while searching: {str(e)}",
                "citations": [],
                "metadata": {
                    "execution_time": time.time() - start_time,
                    "total_cost": 0.0,
                    "error": str(e)
                }
            }
    async def _execute_graph(self, context: GraphContext) -> List[str]:
        execution_path = []
        current_nodes = ["search_router"]
        while current_nodes:
            next_nodes = []
            for node_id in current_nodes:
                if node_id not in self.nodes:
                    continue
                node = self.nodes[node_id]
                if not self._dependencies_satisfied(node, execution_path):
                    continue
                if not node.should_execute(context):
                    logger.info(f"⏭️  Skipping node {node_id} (conditions not met)")
                    continue
                logger.info(f"🔄 Executing node: {node_id}")
                result = await node.execute(context)
                self.execution_results[node_id] = result
                execution_path.append(node_id)
                if result.status == NodeStatus.COMPLETED:
                    next_nodes.extend(result.next_nodes)
                    logger.info(f"✅ Node {node_id} completed in {result.execution_time:.2f}s")
                else:
                    logger.error(f"❌ Node {node_id} failed: {result.error}")
                    break
            current_nodes = list(set(next_nodes))
        return execution_path
    def _dependencies_satisfied(self, node, execution_path: List[str]) -> bool:
        return all(dep in execution_path for dep in node.dependencies)
    def _get_final_response(self, context: GraphContext) -> str:
        synthesize_result = self.execution_results.get("synthesize_response")
        if synthesize_result and synthesize_result.status == NodeStatus.COMPLETED:
            return synthesize_result.data.get("synthesized_response", "No response generated.")
        if context.search_results:
            return f"I found {len(context.search_results)} results for your query about '{context.query}'."
        else:
            return "I couldn't find any relevant information for your query."
    def _get_confidence_score(self, context: GraphContext) -> float:
        synthesize_result = self.execution_results.get("synthesize_response")
        if synthesize_result and synthesize_result.status == NodeStatus.COMPLETED:
            return synthesize_result.data.get("confidence_score", 0.0)
        return 0.0
    def _get_performance_metrics(self, context: GraphContext) -> Dict[str, Any]:
        metrics = {
            "node_execution_times": {},
            "total_nodes_executed": len(self.execution_results),
            "successful_nodes": sum(1 for r in self.execution_results.values() if r.status == NodeStatus.COMPLETED),
            "failed_nodes": sum(1 for r in self.execution_results.values() if r.status == NodeStatus.FAILED)
        }
        for node_id, result in self.execution_results.items():
            metrics["node_execution_times"][node_id] = result.execution_time
        return metrics
