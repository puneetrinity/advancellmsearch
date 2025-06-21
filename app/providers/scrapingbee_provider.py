# app/providers/scrapingbee_provider.py
"""
ScrapingBee API provider for content extraction with standardized interface.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import aiohttp
import asyncio
from app.providers.base_provider import BaseProvider, ProviderConfig, ProviderResult, ProviderError
import logging

@dataclass
class ScrapingQuery:
    url: str
    extract_rules: Optional[Dict[str, str]] = None
    screenshot: bool = False
    render_js: bool = True
    premium_proxy: bool = False
    country_code: Optional[str] = None
    device: str = "desktop"
    wait: int = 0
    wait_for: Optional[str] = None

@dataclass
class ScrapingResult:
    url: str
    html: str
    text: str
    title: str
    status_code: int
    headers: Dict[str, str]
    screenshot_url: Optional[str] = None
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ScrapingBeeProvider(BaseProvider):
    def __init__(self, config: ProviderConfig, logger: Optional[logging.Logger] = None):
        if not config.base_url:
            config.base_url = "https://app.scrapingbee.com/api/v1"
        if config.cost_per_request == 0.0:
            config.cost_per_request = 0.002
        super().__init__(config, logger)

    async def initialize(self) -> None:
        if not self.config.api_key:
            raise ProviderError(
                message="ScrapingBee API key is required",
                provider=self.get_provider_name(),
                error_code="MISSING_API_KEY"
            )
        self._session = await self._create_session()
        try:
            await self._test_api_connection()
            self._initialized = True
            self.logger.info(f"Initialized {self.get_provider_name()}")
        except Exception as e:
            await self.cleanup()
            raise ProviderError(
                message=f"Failed to initialize ScrapingBee: {str(e)}",
                provider=self.get_provider_name(),
                error_code="INITIALIZATION_FAILED",
                original_error=e
            )

    async def cleanup(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        self._initialized = False

    def is_available(self) -> bool:
        return bool(self.config.api_key) and self._initialized

    def get_provider_name(self) -> str:
        return "scrapingbee"

    async def scrape(self, query: ScrapingQuery) -> ProviderResult:
        if not self.is_available():
            raise ProviderError(
                message="Provider not initialized or unavailable",
                provider=self.get_provider_name(),
                error_code="PROVIDER_UNAVAILABLE"
            )
        self._validate_scraping_query(query)
        async def _scrape_operation():
            params = {
                "api_key": self.config.api_key,
                "url": query.url,
                "render_js": str(query.render_js).lower(),
                "premium_proxy": str(query.premium_proxy).lower(),
                "device": query.device
            }
            if query.screenshot:
                params["screenshot"] = "true"
            if query.country_code:
                params["country_code"] = query.country_code
            if query.wait > 0:
                params["wait"] = str(query.wait)
            if query.wait_for:
                params["wait_for"] = query.wait_for
            if query.extract_rules:
                for key, selector in query.extract_rules.items():
                    params[f"extract_rules[{key}]"] = selector
            async with self._session.get(self.config.base_url, params=params) as response:
                if response.status == 429:
                    raise ProviderError(
                        message="Rate limit exceeded",
                        provider=self.get_provider_name(),
                        error_code="RATE_LIMIT_EXCEEDED"
                    )
                elif response.status == 401:
                    raise ProviderError(
                        message="Invalid API key",
                        provider=self.get_provider_name(),
                        error_code="INVALID_API_KEY"
                    )
                elif response.status == 422:
                    error_text = await response.text()
                    raise ProviderError(
                        message=f"Invalid parameters: {error_text}",
                        provider=self.get_provider_name(),
                        error_code="INVALID_PARAMETERS"
                    )
                elif response.status != 200:
                    error_text = await response.text()
                    raise ProviderError(
                        message=f"Scraping failed: {response.status} - {error_text}",
                        provider=self.get_provider_name(),
                        error_code=f"HTTP_{response.status}"
                    )
                html_content = await response.text()
                response_headers = dict(response.headers)
                result = self._parse_scraping_result(
                    query.url,
                    html_content,
                    response.status,
                    response_headers,
                    query.extract_rules
                )
                return result
        return await self._execute_with_retry(_scrape_operation)

    async def scrape_multiple(self, queries: List[ScrapingQuery], max_concurrent: int = 5) -> List[ProviderResult]:
        semaphore = asyncio.Semaphore(max_concurrent)
        async def _scrape_with_semaphore(query: ScrapingQuery):
            async with semaphore:
                return await self.scrape(query)
        tasks = [_scrape_with_semaphore(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_result = ProviderResult(
                    success=False,
                    error=str(result),
                    provider_name=self.get_provider_name(),
                    metadata={"original_url": queries[i].url}
                )
                processed_results.append(failed_result)
            else:
                processed_results.append(result)
        return processed_results

    def _validate_scraping_query(self, query: ScrapingQuery) -> None:
        if not query.url.strip():
            raise ProviderError(
                message="URL cannot be empty",
                provider=self.get_provider_name(),
                error_code="INVALID_URL"
            )
        if not query.url.startswith(("http://", "https://")):
            raise ProviderError(
                message="URL must start with http:// or https://",
                provider=self.get_provider_name(),
                error_code="INVALID_URL_SCHEME"
            )

    def _parse_scraping_result(self, url: str, html: str, status_code: int, headers: Dict[str, str], extract_rules: Optional[Dict[str, str]] = None) -> ScrapingResult:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        title_elem = soup.find('title')
        title = title_elem.get_text(strip=True) if title_elem else ""
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = ' '.join(chunk for chunk in chunks if chunk)
        extracted_data = {}
        if extract_rules:
            for key, selector in extract_rules.items():
                elements = soup.select(selector)
                if elements:
                    extracted_data[key] = [elem.get_text(strip=True) for elem in elements]
                else:
                    extracted_data[key] = []
        return ScrapingResult(
            url=url,
            html=html,
            text=clean_text,
            title=title,
            status_code=status_code,
            headers=headers,
            extracted_data=extracted_data,
            metadata={
                "content_length": len(html),
                "text_length": len(clean_text),
                "extraction_rules": bool(extract_rules)
            }
        )

    async def _test_api_connection(self) -> None:
        test_params = {
            "api_key": self.config.api_key,
            "url": "https://httpbin.org/html",
            "render_js": "false"
        }
        async with self._session.get(self.config.base_url, params=test_params) as response:
            if response.status not in [200, 429]:
                raise Exception(f"API test failed: {response.status}")
