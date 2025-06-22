# app/providers/router.py

"""
Smart Search Router
Intelligent search provider routing with cost optimization and fallback handling
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)

class SearchProvider(Enum):
    """Available search providers"""
    BRAVE = "brave"
    SCRAPINGBEE = "scrapingbee"
    DUCKDUCKGO = "duckduckgo"  # Added DuckDuckGo as a search provider

class ScrapingBeeProvider:
    """ScrapingBee provider with the missing search method"""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://app.scrapingbee.com/api/v1"
        self.cost_per_search = 0.002
    async def search(self, query):
        """Required search method - ScrapingBee is primarily for content enhancement"""
        logger.info("ScrapingBee search called - returning empty (use for content enhancement)")
        return []
    def is_available(self) -> bool:
        """Check if ScrapingBee is available"""
        return bool(self.api_key)

class BraveSearchProvider:
    """Simple Brave Search provider"""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.cost_per_search = 0.008
    async def search(self, query):
        """Search using Brave Search API"""
        if not self.api_key:
            logger.warning("Brave Search API key not configured")
            return []
        try:
            import aiohttp
            import json
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key
            }
            params = {
                "q": query,
                "count": 10,
                "offset": 0,
                "mkt": "en-US",
                "safesearch": "moderate",
                "text_decorations": False,
                "spellcheck": True
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.base_url, 
                    headers=headers, 
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        logger.error(f"Brave API error: {response.status}")
                        return []
                    data = await response.json()
                    results = []
                    web_results = data.get("web", {}).get("results", [])
                    for item in web_results:
                        result = {
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "snippet": item.get("description", ""),
                            "provider": "brave",
                            "relevance_score": self._calculate_relevance_score(item),
                            "metadata": {
                                "brave_score": item.get("score", 0),
                                "age": item.get("age", ""),
                                "family_friendly": item.get("family_friendly", True)
                            }
                        }
                        results.append(result)
                    logger.info(f"Brave search returned {len(results)} results for: {query}")
                    return results
        except Exception as e:
            logger.error(f"Brave search failed: {str(e)}")
            return []

    def _calculate_relevance_score(self, result_item):
        """Calculate relevance score for search result"""
        score = 0.5  # Base score
        brave_score = result_item.get("score", 0)
        if brave_score > 0:
            score += min(brave_score / 100, 0.3)
        if result_item.get("title"):
            score += 0.1
        description = result_item.get("description", "")
        if len(description) > 50:
            score += 0.1
        return min(score, 1.0)
        
class SmartSearchRouter:
    """
    The SmartSearchRouter class that your main.py is trying to import and initialize
    """
    def __init__(self, brave_api_key: Optional[str] = None, scrapingbee_api_key: Optional[str] = None):
        self.brave_api_key = brave_api_key
        self.scrapingbee_api_key = scrapingbee_api_key
        self.providers = {}
        self.initialized = False
        logger.info("SmartSearchRouter created", 
                   brave_available=bool(brave_api_key),
                   scrapingbee_available=bool(scrapingbee_api_key))
    async def __aenter__(self):
        """Async context manager entry - this is what's called on line 89 of main.py"""
        try:
            logger.info("Initializing SmartSearchRouter...")
            # Initialize providers
            self.providers = {
                SearchProvider.BRAVE: BraveSearchProvider(self.brave_api_key),
                SearchProvider.SCRAPINGBEE: ScrapingBeeProvider(self.scrapingbee_api_key)
            }
            # Filter out unavailable providers
            available_providers = {
                provider: instance for provider, instance in self.providers.items()
                if instance.is_available()
            }
            self.providers = available_providers
            self.initialized = True
            logger.info("SmartSearchRouter initialized successfully", 
                       available_providers=list(self.providers.keys()))
            return self
        except Exception as e:
            logger.error(f"Failed to initialize SmartSearchRouter: {e}")
            raise
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        logger.info("SmartSearchRouter cleanup completed")
        self.initialized = False
    def determine_search_strategy(self, budget: float, quality_requirement: str) -> Tuple[SearchProvider, bool]:
        if quality_requirement == "premium" and budget >= 1.50:
            return SearchProvider.BRAVE, True
        elif budget >= 0.50:
            return SearchProvider.BRAVE, False
        else:
            return SearchProvider.DUCKDUCKGO, False
    async def search(
        self, 
        query: str, 
        budget: float = 2.0, 
        quality_requirement: str = "standard",
        max_results: int = 10
    ) -> List[dict]:
        primary_provider, enhance_content = self.determine_search_strategy(budget, quality_requirement)
        logger.info(f"Search strategy: {primary_provider.value}, enhance={enhance_content}, budget=₹{budget}")
        try:
            provider = self.providers[primary_provider]
            response = await provider.search(query)
            if enhance_content and primary_provider == SearchProvider.BRAVE:
                response = await self._enhance_top_results(response, max_enhance=3)
                response.enhanced = True
            return response
        except Exception as e:
            logger.error(f"Primary search failed ({primary_provider.value}): {str(e)}")
            if primary_provider != SearchProvider.DUCKDUCKGO:
                logger.info("Falling back to DuckDuckGo")
                return await self.providers[SearchProvider.DUCKDUCKGO].search(query)
            else:
                raise
    async def _enhance_top_results(self, response: List[dict], max_enhance: int = 3) -> List[dict]:
        scrapingbee = self.providers[SearchProvider.SCRAPINGBEE]
        enhanced_results = []
        for i, result in enumerate(response):
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
    async def search_duckduckgo(self, query: str) -> List[dict]:
        """DuckDuckGo search implementation"""
        logger.info(f"DuckDuckGo search called for: {query}")
        # TODO: Implement actual DuckDuckGo search API call
        return []
