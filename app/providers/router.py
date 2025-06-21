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
        # TODO: Implement actual Brave Search API call
        logger.info(f"Brave search called for: {query}")
        return []
    def is_available(self) -> bool:
        """Check if Brave Search is available"""
        return bool(self.api_key)

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
