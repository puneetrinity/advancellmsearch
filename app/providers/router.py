# app/providers/router.py

"""
Smart Search Router
Intelligent search provider routing with cost optimization and fallback handling
"""

import asyncio
import time
from typing import Tuple
import logging

from .search_providers import (
    SearchProvider,
    SearchResponse,
    BraveSearchProvider,
    ScrapingBeeProvider,
    DuckDuckGoProvider
)

logger = logging.getLogger(__name__)

class SmartSearchRouter:
    """Intelligent search provider routing with cost optimization"""
    def __init__(self, brave_api_key: str, scrapingbee_api_key: str):
        self.providers = {}
        self.brave_key = brave_api_key
        self.scrapingbee_key = scrapingbee_api_key
        self.PREMIUM_THRESHOLD = 1.50  # ₹1.50 - Use Brave + ScrapingBee
        self.STANDARD_THRESHOLD = 0.50  # ₹0.50 - Use Brave only
    async def __aenter__(self):
        self.providers = {
            SearchProvider.BRAVE: BraveSearchProvider(self.brave_key),
            SearchProvider.SCRAPINGBEE: ScrapingBeeProvider(self.scrapingbee_key),
            SearchProvider.DUCKDUCKGO: DuckDuckGoProvider()
        }
        for provider in self.providers.values():
            await provider.__aenter__()
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for provider in self.providers.values():
            await provider.__aexit__(exc_type, exc_val, exc_tb)
    def determine_search_strategy(self, budget: float, quality_requirement: str) -> Tuple[SearchProvider, bool]:
        if quality_requirement == "premium" and budget >= self.PREMIUM_THRESHOLD:
            return SearchProvider.BRAVE, True
        elif budget >= self.STANDARD_THRESHOLD:
            return SearchProvider.BRAVE, False
        else:
            return SearchProvider.DUCKDUCKGO, False
    async def search(
        self, 
        query: str, 
        budget: float = 2.0, 
        quality_requirement: str = "standard",
        max_results: int = 10
    ) -> SearchResponse:
        primary_provider, enhance_content = self.determine_search_strategy(budget, quality_requirement)
        logger.info(f"Search strategy: {primary_provider.value}, enhance={enhance_content}, budget=₹{budget}")
        try:
            provider = self.providers[primary_provider]
            response = await provider.search(query, max_results)
            if enhance_content and primary_provider == SearchProvider.BRAVE:
                response = await self._enhance_top_results(response, max_enhance=3)
                response.enhanced = True
            return response
        except Exception as e:
            logger.error(f"Primary search failed ({primary_provider.value}): {str(e)}")
            if primary_provider != SearchProvider.DUCKDUCKGO:
                logger.info("Falling back to DuckDuckGo")
                return await self.providers[SearchProvider.DUCKDUCKGO].search(query, max_results)
            else:
                raise
    async def _enhance_top_results(self, response: SearchResponse, max_enhance: int = 3) -> SearchResponse:
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
