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


# Search Provider Implementations

class BaseSearchProvider:
    """Base class for search providers"""
    
    def __init__(self):
        self.cost_per_search = 0.008  # Default cost
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform search and return results"""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if provider is available"""
        return True
    
    def get_cost(self) -> float:
        """Get cost of last search"""
        return self.cost_per_search


class BraveSearchProvider(BaseSearchProvider):
    """Brave Search API provider"""
    
    def __init__(self):
        super().__init__()
        self.api_key = get_settings().brave_search_api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.cost_per_search = 0.008
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search using Brave Search API"""
        if not self.api_key:
            raise Exception("Brave Search API key not configured")
        
        results = []
        
        for search_term in query.expanded_queries[:2]:  # Limit to 2 queries
            try:
                params = {
                    "q": search_term,
                    "count": min(query.max_results, 10),
                    "country": "US",
                    "search_lang": query.language,
                    "ui_lang": query.language
                }
                
                headers = {
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self.api_key
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.base_url, params=params, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for item in data.get("web", {}).get("results", []):
                                result = SearchResult(
                                    title=item.get("title", ""),
                                    url=item.get("url", ""),
                                    snippet=item.get("description", ""),
                                    source="brave",
                                    metadata={
                                        "published": item.get("age"),
                                        "language": item.get("language")
                                    }
                                )
                                results.append(result)
                        else:
                            logger.warning(f"Brave Search API error: {response.status}")
                            
            except Exception as e:
                logger.error(f"Brave Search error: {e}")
        
        return results
    
    def is_available(self) -> bool:
        """Check if Brave Search is available"""
        return bool(self.api_key)


class DuckDuckGoProvider(BaseSearchProvider):
    """DuckDuckGo search provider (free, no API key required)"""
    
    def __init__(self):
        super().__init__()
        self.cost_per_search = 0.0  # Free
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search using DuckDuckGo (simplified implementation)"""
        results = []
        
        try:
            # This is a simplified implementation
            # In production, you'd use a proper DDG API wrapper
            search_url = "https://html.duckduckgo.com/html/"
            
            for search_term in query.expanded_queries[:1]:  # Limit to 1 query for free tier
                params = {
                    "q": search_term
                }
                
                headers = {
                    "User-Agent": "Mozilla/5.0 (compatible; AI-Search-Bot/1.0)"
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(search_url, params=params, headers=headers) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            # Parse DuckDuckGo results (simplified)
                            for result_div in soup.find_all('div', class_='result')[:query.max_results]:
                                title_elem = result_div.find('a', class_='result__a')
                                snippet_elem = result_div.find('a', class_='result__snippet')
                                
                                if title_elem:
                                    result = SearchResult(
                                        title=title_elem.get_text(strip=True),
                                        url=title_elem.get('href', ''),
                                        snippet=snippet_elem.get_text(strip=True) if snippet_elem else "",
                                        source="duckduckgo"
                                    )
                                    results.append(result)
                        
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
        
        return results


class GoogleSearchProvider(BaseSearchProvider):
    """Google Custom Search API provider"""
    
    def __init__(self):
        super().__init__()
        self.api_key = get_settings().google_search_api_key if hasattr(get_settings(), 'google_search_api_key') else None
        self.search_engine_id = get_settings().google_search_engine_id if hasattr(get_settings(), 'google_search_engine_id') else None
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.cost_per_search = 0.005
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search using Google Custom Search API"""
        if not self.api_key or not self.search_engine_id:
            raise Exception("Google Search API not properly configured")
        
        results = []
        
        for search_term in query.expanded_queries[:1]:  # Limit due to API quotas
            try:
                params = {
                    "key": self.api_key,
                    "cx": self.search_engine_id,
                    "q": search_term,
                    "num": min(query.max_results, 10)
                }
                
                # Add time filter if specified
                if query.time_filter:
                    params["dateRestrict"] = query.time_filter
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.base_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for item in data.get("items", []):
                                result = SearchResult(
                                    title=item.get("title", ""),
                                    url=item.get("link", ""),
                                    snippet=item.get("snippet", ""),
                                    source="google",
                                    metadata={
                                        "published": item.get("pagemap", {}).get("metatags", [{}])[0].get("article:published_time"),
                                        "image": item.get("pagemap", {}).get("cse_image", [{}])[0].get("src")
                                    }
                                )
                                results.append(result)
                        else:
                            logger.warning(f"Google Search API error: {response.status}")
                            
            except Exception as e:
                logger.error(f"Google Search error: {e}")
        
        return results
    
    def is_available(self) -> bool:
        """Check if Google Search is available"""
        return bool(self.api_key and self.search_engine_id)


# Helper function to create and build search graph
async def create_search_graph(
    model_manager: ModelManager, 
    cache_manager: CacheManager
) -> SearchGraph:
    """Create and build a search graph instance"""
    graph = SearchGraph(model_manager, cache_manager)
    graph.build()
    return graph
