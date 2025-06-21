# app/providers/provider_factory.py
"""
Provider Factory for standardized provider instantiation with environment-based config and validation.
"""
from typing import Optional
from app.core.config import get_settings
from app.providers.brave_search_provider import BraveSearchProvider, ProviderConfig as BraveConfig
from app.providers.scrapingbee_provider import ScrapingBeeProvider, ProviderConfig as ScrapingBeeConfig
import logging

logger = logging.getLogger(__name__)

def create_brave_search_provider() -> Optional[BraveSearchProvider]:
    settings = get_settings()
    api_key = settings.brave_search_api_key or getattr(settings, "BRAVE_API_KEY", None)
    if not api_key:
        logger.warning("Brave Search API key is missing in settings.")
        return None
    config = BraveConfig(
        api_key=api_key,
        cost_per_request=0.008,
        timeout=30
    )
    return BraveSearchProvider(config)

def create_scrapingbee_provider() -> Optional[ScrapingBeeProvider]:
    settings = get_settings()
    api_key = settings.scrapingbee_api_key or getattr(settings, "SCRAPINGBEE_API_KEY", None)
    if not api_key:
        logger.warning("ScrapingBee API key is missing in settings.")
        return None
    config = ScrapingBeeConfig(
        api_key=api_key,
        cost_per_request=0.002,
        timeout=60
    )
    return ScrapingBeeProvider(config)
