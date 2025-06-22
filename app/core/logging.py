"""
Real structured logging implementation with correlation IDs and performance monitoring.
Replaces dummy logging.py
"""
import logging
import logging.config
import sys
import uuid
from typing import Any, Dict, Optional
from contextvars import ContextVar
import structlog
from pythonjsonlogger import jsonlogger

# Context variable for correlation ID tracking across async calls
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)

def get_correlation_id() -> str:
    """Get current correlation ID or generate a new one."""
    current_id = correlation_id.get()
    if current_id is None:
        current_id = str(uuid.uuid4())
        correlation_id.set(current_id)
    return current_id

def set_correlation_id(cid: str) -> None:
    """Set correlation ID for current context."""
    correlation_id.set(cid)

class CorrelationProcessor:
    """Structlog processor to add correlation ID to all log entries."""
    
    def __call__(self, logger, method_name, event_dict):
        event_dict['correlation_id'] = get_correlation_id()
        return event_dict

class PerformanceProcessor:
    """Structlog processor to add performance context."""
    
    def __call__(self, logger, method_name, event_dict):
        # Add timestamp for performance tracking
        event_dict['logger_name'] = logger.name
        event_dict['level'] = method_name.upper()
        return event_dict

class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = get_correlation_id() or "-"
        return True

def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",  # json or text
    enable_file_logging: bool = True,
    log_file_path: str = "logs/app.log"
) -> None:
    """
    Set up structured logging with correlation IDs and performance monitoring.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Output format (json or text) 
        enable_file_logging: Whether to log to file
        log_file_path: Path for log file if file logging enabled
    """
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        CorrelationProcessor(),
        PerformanceProcessor(),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    # Add appropriate renderer based on format
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set up standard logging handlers
    handlers = []
    
    # Console handler with appropriate formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.addFilter(CorrelationIdFilter())
    if log_format == "json":
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(correlation_id)s %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(message)s'
        )
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # File handler if enabled
    if enable_file_logging:
        import os
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.addFilter(CorrelationIdFilter())
        file_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(correlation_id)s %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        format='%(message)s'  # structlog handles formatting
    )
    
    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Log successful setup
    logger = structlog.get_logger("logging")
    logger.info(
        "Logging system initialized",
        level=log_level,
        format=log_format,
        file_logging=enable_file_logging,
        correlation_id=get_correlation_id()
    )

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)

class LoggingMiddleware:
    """FastAPI middleware for request/response logging with performance tracking."""
    
    def __init__(self, app):
        self.app = app
        self.logger = get_logger("middleware.logging")
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Generate correlation ID for this request
        request_id = str(uuid.uuid4())
        set_correlation_id(request_id)
        
        # Extract request info
        method = scope["method"]
        path = scope["path"]
        
        import time
        start_time = time.time()
        
        self.logger.info(
            "Request started",
            method=method,
            path=path,
            correlation_id=request_id
        )
        
        # Process request
        try:
            await self.app(scope, receive, send)
            
            duration = time.time() - start_time
            self.logger.info(
                "Request completed",
                method=method,
                path=path,
                duration_ms=round(duration * 1000, 2),
                correlation_id=request_id
            )
        
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                "Request failed",
                method=method,
                path=path,
                duration_ms=round(duration * 1000, 2),
                error=str(e),
                correlation_id=request_id,
                exc_info=True
            )
            raise

# Performance monitoring utilities
class PerformanceLogger:
    """Context manager for performance logging."""
    
    def __init__(self, operation_name: str, logger_name: str = "performance"):
        self.operation_name = operation_name
        self.logger = get_logger(logger_name)
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.debug(f"{self.operation_name} started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(
                f"{self.operation_name} completed",
                duration_ms=round(duration * 1000, 2)
            )
        else:
            self.logger.error(
                f"{self.operation_name} failed",
                duration_ms=round(duration * 1000, 2),
                error=str(exc_val)
            )

def log_performance(operation_name: str):
    """Decorator for automatic performance logging - FastAPI compatible."""
    def decorator(func):
        import functools
        import asyncio
        if asyncio.iscoroutinefunction(func):  # async function
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                import time
                import structlog
                logger = structlog.get_logger("performance")
                start_time = time.time()
                logger.debug(f"{operation_name} started")
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    logger.info(
                        f"{operation_name} completed",
                        duration_ms=round(duration * 1000, 2),
                        correlation_id=get_correlation_id()
                    )
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(
                        f"{operation_name} failed",
                        duration_ms=round(duration * 1000, 2),
                        error=str(e),
                        correlation_id=get_correlation_id()
                    )
                    raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                import time
                import structlog
                logger = structlog.get_logger("performance")
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    logger.info(f"{operation_name} completed", duration_ms=round(duration * 1000, 2))
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(f"{operation_name} failed", duration_ms=round(duration * 1000, 2), error=str(e))
                    raise
            return sync_wrapper
    return decorator

# Export main functions
__all__ = [
    'setup_logging',
    'get_logger', 
    'get_correlation_id',
    'set_correlation_id',
    'LoggingMiddleware',
    'PerformanceLogger',
    'log_performance'
]
