# app/providers/base_provider.py
"""
Standardized base provider for all search and scraping providers.
"""
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp


@dataclass
class ProviderConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    cost_per_request: float = 0.0
    rate_limit_per_minute: int = 60
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    provider_name: str = ""


class ProviderError(Exception):
    def __init__(
        self,
        message: str,
        provider: str,
        error_code: str = None,
        original_error: Exception = None,
    ):
        self.message = message
        self.provider = provider
        self.error_code = error_code
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")


class BaseProvider(ABC):
    def __init__(self, config: ProviderConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self._session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        self._stats = {
            "requests_made": 0,
            "total_cost": 0.0,
            "total_execution_time": 0.0,
            "errors": 0,
            "last_request": None,
        }

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        pass

    async def _create_session(self) -> aiohttp.ClientSession:
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        return aiohttp.ClientSession(timeout=timeout)

    def _update_stats(self, execution_time: float, cost: float, success: bool):
        self._stats["requests_made"] += 1
        self._stats["total_execution_time"] += execution_time
        self._stats["total_cost"] += cost
        self._stats["last_request"] = datetime.now().isoformat()
        if not success:
            self._stats["errors"] += 1

    def get_stats(self) -> Dict[str, Any]:
        stats = self._stats.copy()
        if stats["requests_made"] > 0:
            stats["success_rate"] = (
                (stats["requests_made"] - stats["errors"]) / stats["requests_made"]
            ) * 100
            stats["avg_execution_time"] = (
                stats["total_execution_time"] / stats["requests_made"]
            )
        else:
            stats["success_rate"] = 0.0
            stats["avg_execution_time"] = 0.0
        return stats

    async def _execute_with_retry(
        self, operation_func, *args, **kwargs
    ) -> ProviderResult:
        start_time = time.time()
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                result = await operation_func(*args, **kwargs)
                execution_time = time.time() - start_time
                if not isinstance(result, ProviderResult):
                    result = ProviderResult(
                        success=True,
                        data=result,
                        execution_time=execution_time,
                        cost=self.config.cost_per_request,
                        provider_name=self.get_provider_name(),
                    )
                else:
                    result.execution_time = execution_time
                    result.provider_name = self.get_provider_name()
                    if result.cost == 0.0:
                        result.cost = self.config.cost_per_request
                self._update_stats(execution_time, result.cost, result.success)
                return result
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"Provider operation failed (attempt {attempt + 1}/{self.config.max_retries + 1})",
                    extra={
                        "provider": self.get_provider_name(),
                        "error": str(e),
                        "attempt": attempt + 1,
                    },
                )
                if attempt < self.config.max_retries:
                    await asyncio.sleep(2**attempt)
        execution_time = time.time() - start_time
        self._update_stats(execution_time, 0.0, False)
        raise ProviderError(
            message=f"Operation failed after {self.config.max_retries + 1} attempts: {str(last_error)}",
            provider=self.get_provider_name(),
            error_code="MAX_RETRIES_EXCEEDED",
            original_error=last_error,
        )
