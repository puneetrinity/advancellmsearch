# app/core/async_utils.py
"""
Complete async safety utilities to prevent coroutine leaks.
This is the FULL working version that solves all your coroutine issues.
"""

import asyncio
import logging
from functools import wraps
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)
T = TypeVar("T")


async def ensure_awaited(obj: Union[T, Awaitable[T]], max_depth: int = 10) -> T:
    """
    Ensure an object is fully awaited, handling nested coroutines.

    Args:
        obj: Object that might be a coroutine or regular value
        max_depth: Maximum number of await attempts (prevents infinite loops)

    Returns:
        The fully awaited result
    """
    depth = 0

    while asyncio.iscoroutine(obj) and depth < max_depth:
        try:
            obj = await obj
            depth += 1
            logger.debug(f"Awaited coroutine at depth {depth}")
        except Exception as e:
            logger.error(f"Error awaiting coroutine at depth {depth}: {e}")
            raise

    if depth >= max_depth:
        logger.error(
            f"Max await depth ({max_depth}) exceeded - possible infinite coroutine chain"
        )
        raise RuntimeError(f"Max await depth exceeded: {max_depth}")

    if asyncio.iscoroutine(obj):
        logger.warning(
            "Object is still a coroutine after max depth - this should not happen"
        )

    return obj


async def safe_execute(
    coro_func: Callable[..., Awaitable[T]],
    *args,
    timeout: Optional[float] = None,
    **kwargs,
) -> T:
    """
    Safely execute an async function and ensure result is awaited.

    Args:
        coro_func: Async function to execute
        *args: Positional arguments for the function
        timeout: Optional timeout in seconds
        **kwargs: Keyword arguments for the function

    Returns:
        Fully awaited result
    """
    try:
        # Execute the coroutine function
        coro = coro_func(*args, **kwargs)

        # Apply timeout if specified
        if timeout:
            result = await asyncio.wait_for(coro, timeout=timeout)
        else:
            result = await coro

        # Ensure the result is fully awaited
        return await ensure_awaited(result)

    except asyncio.TimeoutError:
        logger.error(f"Timeout ({timeout}s) executing {coro_func.__name__}")
        raise
    except Exception as e:
        logger.error(f"Error executing {coro_func.__name__}: {e}")
        raise


def coroutine_safe(timeout: Optional[float] = None):
    """
    Decorator to make async functions coroutine-safe.

    Usage:
        @coroutine_safe(timeout=30.0)
        async def my_endpoint_handler(...):
            result = await some_async_operation()
            return result
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await safe_execute(func, *args, timeout=timeout, **kwargs)

        return wrapper

    return decorator


async def safe_gather(*coroutines, return_exceptions: bool = False) -> list:
    """
    Safely gather multiple coroutines and ensure all results are awaited.

    Args:
        *coroutines: Coroutines to gather
        return_exceptions: Whether to return exceptions instead of raising

    Returns:
        List of fully awaited results
    """
    try:
        results = await asyncio.gather(*coroutines, return_exceptions=return_exceptions)

        # Ensure all results are fully awaited
        safe_results = []
        for result in results:
            if isinstance(result, Exception) and not return_exceptions:
                raise result
            safe_results.append(await ensure_awaited(result))

        return safe_results

    except Exception as e:
        logger.error(f"Error in safe_gather: {e}")
        raise


class AsyncSafetyValidator:
    """
    Validator to check for coroutine leaks in responses.
    Use in testing to catch coroutine safety issues.
    """

    @staticmethod
    def validate_response(response: Any, path: str = "root") -> bool:
        """
        Recursively validate that a response contains no coroutines.

        Args:
            response: Response object to validate
            path: Current path in the object tree (for debugging)

        Returns:
            True if safe, False if coroutines found
        """
        if asyncio.iscoroutine(response):
            logger.error(f"Coroutine found in response at path: {path}")
            return False

        if isinstance(response, dict):
            for key, value in response.items():
                if not AsyncSafetyValidator.validate_response(value, f"{path}.{key}"):
                    return False

        elif isinstance(response, (list, tuple)):
            for i, item in enumerate(response):
                if not AsyncSafetyValidator.validate_response(item, f"{path}[{i}]"):
                    return False

        elif hasattr(response, "__dict__"):
            # Handle Pydantic models and other objects
            for key, value in response.__dict__.items():
                if not AsyncSafetyValidator.validate_response(value, f"{path}.{key}"):
                    return False

        return True

    @staticmethod
    def assert_no_coroutines(
        response: Any, message: str = "Response contains coroutines"
    ):
        """
        Assert that response contains no coroutines. Raises AssertionError if found.
        """
        if not AsyncSafetyValidator.validate_response(response):
            raise AssertionError(message)


# Convenience functions for common patterns


async def safe_model_call(model_manager, method_name: str, *args, **kwargs):
    """
    Safely call a model manager method and ensure result is awaited.
    """
    method = getattr(model_manager, method_name)
    return await safe_execute(method, *args, **kwargs)


async def safe_graph_execute(graph, state, timeout: float = 30.0):
    """
    Safely execute a graph and ensure result is awaited.
    """
    return await safe_execute(graph.execute, state, timeout=timeout)


async def safe_search_execute(search_system, query: str, **kwargs):
    """
    Safely execute a search and ensure result is awaited.
    """
    return await safe_execute(
        search_system.execute_optimized_search, query=query, **kwargs
    )
