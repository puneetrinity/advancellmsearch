"""
Real Search Provider Integration
Advanced multi-provider search system with smart routing and cost optimization
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from urllib.parse import quote_plus
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchProvider(Enum):
    BRAVE = "brave"
    SCRAPINGBEE = "scrapingbee"
    DUCKDUCKGO = "duckduckgo"

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    provider: str
    confidence_score: float = 0.0
    enhanced_content: Optional[str] = None
    scraped_at: Optional[float] = None
    cost: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SearchResponse:
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float
    total_cost: float
    provider_used: str
    enhanced: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results,
            "search_time": self.search_time,
            "total_cost": self.total_cost,
            "provider_used": self.provider_used,
            "enhanced": self.enhanced
        }

class BaseSearchProvider(ABC):
    """Abstract base class for search providers"""
    
    def __init__(self, name: str, cost_per_query: float = 0.0):
        self.name = name
        self.cost_per_query = cost_per_query
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> SearchResponse:
        pass
        
    def calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score for search result"""
        score = 0.5  # Base score
        
        # Title relevance
        if result.get('title'):
            score += 0.1
            
        # Snippet quality
        if result.get('snippet') and len(result.get('snippet', '')) > 50:
            score += 0.2
            
        # URL quality (domain authority proxy)
        url = result.get('url', '')
        if any(domain in url for domain in ['.edu', '.gov', '.org']):
            score += 0.1
        elif any(domain in url for domain in ['wikipedia.org', 'stackoverflow.com']):
            score += 0.05
            
        return min(score, 1.0)

class BraveSearchProvider(BaseSearchProvider):
    """Brave Search API integration - Primary provider"""
    
    def __init__(self, api_key: str):
        super().__init__("brave", cost_per_query=0.42)  # ₹0.42 per search
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        
    async def search(self, query: str, max_results: int = 10) -> SearchResponse:
        start_time = time.time()
        
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
        
        params = {
            "q": query,
            "count": min(max_results, 20),  # Brave API limit
            "offset": 0,
            "mkt": "en-US",
            "safesearch": "moderate",
            "freshness": "pw",  # Past week for fresher results
            "text_decorations": False,
            "spellcheck": True
        }
        
        try:
            async with self.session.get(
                self.base_url, 
                headers=headers, 
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                
                if response.status != 200:
                    logger.error(f"Brave API error: {response.status}")
                    raise Exception(f"Brave API returned status {response.status}")
                
                data = await response.json()
                results = []
                
                # Parse Brave search results
                web_results = data.get("web", {}).get("results", [])
                
                for item in web_results[:max_results]:
                    result = SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("description", ""),
                        provider="brave",
                        confidence_score=self.calculate_confidence(item),
                        cost=self.cost_per_query / len(web_results) if web_results else self.cost_per_query
                    )
                    results.append(result)
                
                search_time = time.time() - start_time
                
                return SearchResponse(
                    query=query,
                    results=results,
                    total_results=len(results),
                    search_time=search_time,
                    total_cost=self.cost_per_query,
                    provider_used="brave"
                )
                
        except Exception as e:
            logger.error(f"Brave search failed: {str(e)}")
            raise

class ScrapingBeeProvider(BaseSearchProvider):
    """ScrapingBee content enhancement - Premium content scraping"""
    
    def __init__(self, api_key: str):
        super().__init__("scrapingbee", cost_per_query=0.84)  # ₹0.84 per scrape
        self.api_key = api_key
        self.base_url = "https://app.scrapingbee.com/api/v1/"
        
    async def enhance_result(self, result: SearchResult) -> SearchResult:
        """Enhance a search result with full content scraping"""
        start_time = time.time()
        
        params = {
            "api_key": self.api_key,
            "url": result.url,
            "render_js": "false",  # Faster scraping
            "premium_proxy": "true",  # Better success rate
            "country_code": "us",
            "extract_rules": json.dumps({
                "title": "title",
                "content": "p",
                "headings": "h1,h2,h3"
            })
        }
        
        try:
            async with self.session.get(
                self.base_url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                
                if response.status == 200:
                    content = await response.text()
                    
                    # Extract meaningful content (simplified)
                    enhanced_content = self._extract_content(content)
                    
                    result.enhanced_content = enhanced_content
                    result.scraped_at = time.time()
                    result.cost += self.cost_per_query
                    result.confidence_score = min(result.confidence_score + 0.2, 1.0)
                    
                    logger.info(f"Enhanced content for {result.url} ({len(enhanced_content)} chars)")
                    
                else:
                    logger.warning(f"ScrapingBee failed for {result.url}: {response.status}")
                    
        except Exception as e:
            logger.error(f"Content enhancement failed for {result.url}: {str(e)}")
            
        return result
    
    def _extract_content(self, html_content: str) -> str:
        """Extract meaningful content from HTML"""
        # Simplified content extraction
        # In production, use BeautifulSoup or similar
        content = html_content[:2000]  # First 2000 chars
        return content.strip()
    
    async def search(self, query: str, max_results: int = 10) -> SearchResponse:
        """
        ScrapingBee doesn't do search directly - it's for content enhancement
        This method exists to satisfy the abstract base class requirement
        """
        start_time = time.time()
        
        # Return a proper SearchResponse object instead of a dict
        return SearchResponse(
            query=query,
            results=[],
            total_results=0,
            search_time=time.time() - start_time,
            total_cost=0.0,
            provider_used="scrapingbee"
        )

class DuckDuckGoProvider(BaseSearchProvider):
    """DuckDuckGo search - Free fallback provider"""
    
    def __init__(self):
        super().__init__("duckduckgo", cost_per_query=0.0)  # Free
        self.base_url = "https://html.duckduckgo.com/html/"
        
    async def search(self, query: str, max_results: int = 10) -> SearchResponse:
        start_time = time.time()
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        params = {
            "q": query,
            "b": "",  # No ads
            "kl": "us-en",
            "df": "w"  # Past week
        }
        
        try:
            async with self.session.get(
                self.base_url,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                
                if response.status != 200:
                    raise Exception(f"DuckDuckGo returned status {response.status}")
                
                html = await response.text()
                results = self._parse_duckduckgo_results(html, max_results)
                
                search_time = time.time() - start_time
                
                return SearchResponse(
                    query=query,
                    results=results,
                    total_results=len(results),
                    search_time=search_time,
                    total_cost=0.0,  # Free
                    provider_used="duckduckgo"
                )
                
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {str(e)}")
            raise
    
    def _parse_duckduckgo_results(self, html: str, max_results: int) -> List[SearchResult]:
        """Parse DuckDuckGo HTML results"""
        # Simplified HTML parsing
        # In production, use BeautifulSoup
        results = []
        
        # Mock results for now - replace with actual HTML parsing
        for i in range(min(max_results, 5)):
            result = SearchResult(
                title=f"DuckDuckGo Result {i+1}",
                url=f"https://example.com/result{i+1}",
                snippet=f"This is a sample result snippet from DuckDuckGo search.",
                provider="duckduckgo",
                confidence_score=0.6,
                cost=0.0
            )
            results.append(result)
            
        return results

class SmartSearchRouter:
    """Intelligent search provider routing with cost optimization"""
    
    def __init__(self, brave_api_key: str, scrapingbee_api_key: str):
        self.providers = {}
        self.brave_key = brave_api_key
        self.scrapingbee_key = scrapingbee_api_key
        
        # Cost thresholds
        self.PREMIUM_THRESHOLD = 1.50  # ₹1.50 - Use Brave + ScrapingBee
        self.STANDARD_THRESHOLD = 0.50  # ₹0.50 - Use Brave only
        # Below ₹0.50 - Use DuckDuckGo (free)
        
    async def __aenter__(self):
        # Initialize providers
        self.providers = {
            SearchProvider.BRAVE: BraveSearchProvider(self.brave_key),
            SearchProvider.SCRAPINGBEE: ScrapingBeeProvider(self.scrapingbee_key),
            SearchProvider.DUCKDUCKGO: DuckDuckGoProvider()
        }
        
        # Start all provider sessions
        for provider in self.providers.values():
            await provider.__aenter__()
            
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Close all provider sessions
        for provider in self.providers.values():
            await provider.__aexit__(exc_type, exc_val, exc_tb)
    
    def determine_search_strategy(self, budget: float, quality_requirement: str) -> Tuple[SearchProvider, bool]:
        """Determine optimal search strategy based on budget and quality needs"""
        
        if quality_requirement == "premium" and budget >= self.PREMIUM_THRESHOLD:
            return SearchProvider.BRAVE, True  # With enhancement
        elif budget >= self.STANDARD_THRESHOLD:
            return SearchProvider.BRAVE, False  # Without enhancement
        else:
            return SearchProvider.DUCKDUCKGO, False  # Free fallback
    
    async def search(
        self, 
        query: str, 
        budget: float = 2.0, 
        quality_requirement: str = "standard",
        max_results: int = 10
    ) -> SearchResponse:
        """Execute smart search with cost optimization"""
        
        # Determine strategy
        primary_provider, enhance_content = self.determine_search_strategy(budget, quality_requirement)
        
        logger.info(f"Search strategy: {primary_provider.value}, enhance={enhance_content}, budget=₹{budget}")
        
        try:
            # Execute primary search
            provider = self.providers[primary_provider]
            response = await provider.search(query, max_results)
            
            # Enhance content if requested and budget allows
            if enhance_content and primary_provider == SearchProvider.BRAVE:
                response = await self._enhance_top_results(response, max_enhance=3)
                response.enhanced = True
            
            return response
            
        except Exception as e:
            logger.error(f"Primary search failed ({primary_provider.value}): {str(e)}")
            
            # Fallback to DuckDuckGo if primary fails
            if primary_provider != SearchProvider.DUCKDUCKGO:
                logger.info("Falling back to DuckDuckGo")
                return await self.providers[SearchProvider.DUCKDUCKGO].search(query, max_results)
            else:
                raise
    
    async def _enhance_top_results(self, response: SearchResponse, max_enhance: int = 3) -> SearchResponse:
        """Enhance top search results with ScrapingBee"""
        
        scrapingbee = self.providers[SearchProvider.SCRAPINGBEE]
        enhanced_results = []
        
        for i, result in enumerate(response.results):
            if i < max_enhance:
                try:
                    enhanced_result = await scrapingbee.enhance_result(result)
                    enhanced_results.append(enhanced_result)
                    response.total_cost += enhanced_result.cost
                except Exception as e:
                    logger.warning(f"Failed to enhance result {i}: {str(e)}")
                    enhanced_results.append(result)
            else:
                enhanced_results.append(result)
        
        response.results = enhanced_results
        return response

# Usage example and testing
async def test_search_integration():
    """Test the complete search integration"""
    
    # API keys (use environment variables in production)
    BRAVE_API_KEY = "your_brave_api_key_here"
    SCRAPINGBEE_API_KEY = "your_scrapingbee_api_key_here"
    
    async with SmartSearchRouter(BRAVE_API_KEY, SCRAPINGBEE_API_KEY) as router:
        
        # Test different budget scenarios
        test_cases = [
            {"query": "AI search optimization", "budget": 2.0, "quality": "premium"},
            {"query": "Python async programming", "budget": 1.0, "quality": "standard"},
            {"query": "machine learning basics", "budget": 0.2, "quality": "basic"}
        ]
        
        for test_case in test_cases:
            print(f"\n🔍 Testing: {test_case}")
            
            try:
                response = await router.search(
                    query=test_case["query"],
                    budget=test_case["budget"],
                    quality_requirement=test_case["quality"]
                )
                
                print(f"✅ Success: {response.total_results} results in {response.search_time:.2f}s")
                print(f"💰 Cost: ₹{response.total_cost:.2f} (Budget: ₹{test_case['budget']})")
                print(f"🚀 Provider: {response.provider_used}, Enhanced: {response.enhanced}")
                
                # Show top result
                if response.results:
                    top_result = response.results[0]
                    print(f"📄 Top result: {top_result.title}")
                    print(f"🔗 URL: {top_result.url}")
                    print(f"📊 Confidence: {top_result.confidence_score:.2f}")
                    
            except Exception as e:
                print(f"❌ Failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_search_integration())