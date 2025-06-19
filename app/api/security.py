"""
Input validation and basic security middleware.
Implements security foundations early to prevent rework.
"""
import html
import re
import time
from typing import Any, Dict, List, Optional, Set
from fastapi import HTTPException, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, field_validator

from app.core.logging import get_logger, get_correlation_id
from app.schemas.responses import create_error_response


logger = get_logger("security")

# Security configuration
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
MAX_QUERY_LENGTH = 8000
MAX_FIELD_LENGTH = 1000
RATE_LIMIT_WINDOW = 60  # seconds
DEFAULT_RATE_LIMIT = 60  # requests per window

# Dangerous patterns to detect
DANGEROUS_PATTERNS = [
    r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
    r'javascript:',  # Javascript URLs
    r'on\w+\s*=',  # Event handlers
    r'<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>',  # Iframes
    r'data:text/html',  # Data URLs
    r'vbscript:',  # VBScript
    r'<object\b[^<]*(?:(?!<\/object>)<[^<]*)*<\/object>',  # Object tags
    r'<embed\b[^<]*(?:(?!<\/embed>)<[^<]*)*<\/embed>',  # Embed tags
]

# SQL injection patterns
SQL_PATTERNS = [
    r'\bunion\s+select\b',
    r'\bselect\s+.*\bfrom\b',
    r'\binsert\s+into\b',
    r'\bupdate\s+.*\bset\b',
    r'\bdelete\s+from\b',
    r'\bdrop\s+table\b',
    r'\balter\s+table\b',
    r';\s*(select|insert|update|delete|drop|alter)',
    r'--\s*$',  # SQL comments
    r'/\*.*\*/',  # SQL block comments
]

# Compiled regex patterns for performance
COMPILED_DANGEROUS_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in DANGEROUS_PATTERNS]
COMPILED_SQL_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in SQL_PATTERNS]


class SecurityViolation(Exception):
    """Raised when security validation fails."""
    
    def __init__(self, violation_type: str, details: str, field: Optional[str] = None):
        self.violation_type = violation_type
        self.details = details
        self.field = field
        super().__init__(f"{violation_type}: {details}")


class InputSanitizer:
    """Input sanitization and validation utilities."""
    
    @staticmethod
    def sanitize_html(text: str) -> str:
        """Sanitize HTML content to prevent XSS."""
        if not isinstance(text, str):
            return str(text)
        
        # HTML escape
        sanitized = html.escape(text)
        
        # Remove dangerous patterns
        for pattern in COMPILED_DANGEROUS_PATTERNS:
            sanitized = pattern.sub('', sanitized)
        
        return sanitized
    
    @staticmethod
    def validate_sql_injection(text: str, field_name: str = "input") -> None:
        """Check for potential SQL injection attempts."""
        if not isinstance(text, str):
            return
        
        text_lower = text.lower()
        for pattern in COMPILED_SQL_PATTERNS:
            if pattern.search(text_lower):
                logger.warning(
                    "Potential SQL injection detected",
                    field=field_name,
                    pattern=pattern.pattern,
                    correlation_id=get_correlation_id()
                )
                raise SecurityViolation(
                    "SQL_INJECTION_ATTEMPT",
                    f"Potentially dangerous SQL pattern detected in {field_name}",
                    field_name
                )
    
    @staticmethod
    def validate_length(text: str, max_length: int, field_name: str = "input") -> None:
        """Validate text length."""
        if isinstance(text, str) and len(text) > max_length:
            raise SecurityViolation(
                "INPUT_TOO_LONG",
                f"{field_name} exceeds maximum length of {max_length} characters",
                field_name
            )
    
    @staticmethod
    def validate_content(text: str, field_name: str = "input") -> str:
        """Comprehensive input validation and sanitization."""
        if not isinstance(text, str):
            text = str(text)
        
        # Length validation
        InputSanitizer.validate_length(text, MAX_FIELD_LENGTH, field_name)
        
        # SQL injection check
        InputSanitizer.validate_sql_injection(text, field_name)
        
        # XSS sanitization
        sanitized = InputSanitizer.sanitize_html(text)
        
        # Log if sanitization changed the input
        if sanitized != text:
            logger.info(
                "Input sanitized",
                field=field_name,
                original_length=len(text),
                sanitized_length=len(sanitized),
                correlation_id=get_correlation_id()
            )
        
        return sanitized.strip()


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, identifier: str, limit: int = DEFAULT_RATE_LIMIT, window: int = RATE_LIMIT_WINDOW) -> bool:
        """Check if request is within rate limits."""
        now = time.time()
        
        # Clean old entries
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if now - req_time < window
            ]
        else:
            self.requests[identifier] = []
        
        # Check current count
        current_requests = len(self.requests[identifier])
        
        if current_requests >= limit:
            logger.warning(
                "Rate limit exceeded",
                identifier=identifier,
                current_requests=current_requests,
                limit=limit,
                correlation_id=get_correlation_id()
            )
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True
    
    def get_remaining(self, identifier: str, limit: int = DEFAULT_RATE_LIMIT) -> int:
        """Get remaining requests for identifier."""
        if identifier not in self.requests:
            return limit
        
        current_requests = len(self.requests[identifier])
        return max(0, limit - current_requests)


# Global rate limiter instance
rate_limiter = RateLimiter()


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for input validation and basic protection."""
    
    def __init__(self, app, enable_rate_limiting: bool = True):
        super().__init__(app)
        self.enable_rate_limiting = enable_rate_limiting
        self.logger = get_logger("middleware.security")
    
    async def dispatch(self, request: Request, call_next):
        """Apply security validations."""
        start_time = time.time()
        
        try:
            # Content length validation
            content_length = request.headers.get('content-length')
            if content_length and int(content_length) > MAX_REQUEST_SIZE:
                return self._create_error_response(
                    "REQUEST_TOO_LARGE",
                    f"Request size exceeds maximum of {MAX_REQUEST_SIZE} bytes",
                    413
                )
            
            # Rate limiting
            if self.enable_rate_limiting:
                client_ip = self._get_client_ip(request)
                if not rate_limiter.is_allowed(client_ip):
                    remaining = rate_limiter.get_remaining(client_ip)
                    return self._create_error_response(
                        "RATE_LIMIT_EXCEEDED",
                        "Too many requests. Please try again later.",
                        429,
                        retry_after=RATE_LIMIT_WINDOW,
                        additional_headers={"X-Rate-Limit-Remaining": str(remaining)}
                    )
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            
            # Log successful request
            duration = time.time() - start_time
            self.logger.debug(
                "Security validation passed",
                path=request.url.path,
                method=request.method,
                duration_ms=round(duration * 1000, 2),
                correlation_id=get_correlation_id()
            )
            
            return response
            
        except SecurityViolation as e:
            self.logger.warning(
                "Security violation detected",
                violation_type=e.violation_type,
                details=e.details,
                field=e.field,
                path=request.url.path,
                correlation_id=get_correlation_id()
            )
            return self._create_error_response(
                e.violation_type,
                e.details,
                400
            )
        
        except Exception as e:
            self.logger.error(
                "Security middleware error",
                error=str(e),
                path=request.url.path,
                correlation_id=get_correlation_id(),
                exc_info=True
            )
            return self._create_error_response(
                "SECURITY_ERROR",
                "Security validation failed",
                500
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP for rate limiting."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to client host
        if hasattr(request.client, 'host'):
            return request.client.host
        
        return "unknown"
    
    def _create_error_response(
        self,
        error_code: str,
        message: str,
        status_code: int,
        retry_after: Optional[int] = None,
        additional_headers: Optional[Dict[str, str]] = None
    ) -> Response:
        """Create standardized error response."""
        from fastapi.responses import JSONResponse
        
        error_response = create_error_response(
            message=message,
            error_code=error_code,
            correlation_id=get_correlation_id()
        )
        
        headers = additional_headers or {}
        if retry_after:
            headers["Retry-After"] = str(retry_after)
        
        return JSONResponse(
            status_code=status_code,
            content=error_response.dict(),
            headers=headers
        )


class AuthenticationStub:
    """Basic authentication system stub for development."""
    
    def __init__(self):
        self.logger = get_logger("auth.stub")
        # Simple user database for development
        self.users = {
            "dev-user": {
                "user_id": "dev-user-001",
                "tier": "pro",
                "monthly_budget": 500.0,
                "permissions": ["chat", "search", "analytics"]
            },
            "test-user": {
                "user_id": "test-user-001", 
                "tier": "free",
                "monthly_budget": 20.0,
                "permissions": ["chat"]
            }
        }
    
    def authenticate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Authenticate bearer token (stub implementation)."""
        # Simple token validation for development
        if token.startswith("dev-"):
            user_key = token.replace("dev-", "").replace("-token", "")
            return self.users.get(user_key)
        
        return None
    
    def create_anonymous_user(self, client_ip: str) -> Dict[str, Any]:
        """Create anonymous user for development."""
        return {
            "user_id": f"anon-{hash(client_ip) % 10000}",
            "tier": "free",
            "monthly_budget": 5.0,  # Very limited for anonymous
            "permissions": ["chat"],
            "is_anonymous": True
        }


# Global authentication stub
auth_stub = AuthenticationStub()
security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = None
) -> Dict[str, Any]:
    """Get current user from token or create anonymous user."""
    correlation_id = get_correlation_id()
    
    if credentials and credentials.credentials:
        # Try to authenticate with token
        user = auth_stub.authenticate_token(credentials.credentials)
        if user:
            logger.debug(
                "User authenticated with token",
                user_id=user["user_id"],
                tier=user["tier"],
                correlation_id=correlation_id
            )
            return user
        else:
            logger.warning(
                "Invalid authentication token provided",
                correlation_id=correlation_id
            )
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token"
            )
    
    # Create anonymous user
    client_ip = request.client.host if request.client else "unknown"
    anonymous_user = auth_stub.create_anonymous_user(client_ip)
    
    logger.debug(
        "Anonymous user created",
        user_id=anonymous_user["user_id"],
        client_ip=client_ip,
        correlation_id=correlation_id
    )
    
    return anonymous_user


def require_permission(permission: str):
    """Decorator to require specific permission."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract user from kwargs (injected by dependency)
            current_user = None
            for arg in args:
                if isinstance(arg, dict) and "permissions" in arg:
                    current_user = arg
                    break
            
            if not current_user:
                for value in kwargs.values():
                    if isinstance(value, dict) and "permissions" in value:
                        current_user = value
                        break
            
            if not current_user:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required"
                )
            
            if permission not in current_user.get("permissions", []):
                logger.warning(
                    "Permission denied",
                    user_id=current_user.get("user_id"),
                    required_permission=permission,
                    user_permissions=current_user.get("permissions", []),
                    correlation_id=get_correlation_id()
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission '{permission}' required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_tier(min_tier: str):
    """Decorator to require minimum user tier."""
    tier_levels = {"free": 0, "pro": 1, "enterprise": 2}
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract user from kwargs
            current_user = None
            for value in kwargs.values():
                if isinstance(value, dict) and "tier" in value:
                    current_user = value
                    break
            
            if not current_user:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required"
                )
            
            user_tier = current_user.get("tier", "free")
            if tier_levels.get(user_tier, 0) < tier_levels.get(min_tier, 999):
                logger.warning(
                    "Insufficient tier level",
                    user_id=current_user.get("user_id"),
                    user_tier=user_tier,
                    required_tier=min_tier,
                    correlation_id=get_correlation_id()
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"Tier '{min_tier}' or higher required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Input validation models
class SecureTextInput(BaseModel):
    """Base model with security validation for text inputs."""
    
    @field_validator('*', mode='before')
    @classmethod
    def validate_and_sanitize_strings(cls, v, info):
        """Validate and sanitize all string fields."""
        if isinstance(v, str):
            field_name = info.field_name or "input"
            return InputSanitizer.validate_content(v, field_name)
        return v


class SecureChatInput(SecureTextInput):
    """Secure chat input with additional validation."""
    message: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    
    @field_validator('message')
    @classmethod
    def validate_message_content(cls, v):
        """Additional validation for chat messages."""
        if not v.strip():
            raise ValueError("Message cannot be empty")
        
        # Check for obvious spam patterns
        if len(set(v.lower())) < 3 and len(v) > 10:
            raise ValueError("Message appears to be spam")
        
        return v


# Security utilities
def check_content_policy(text: str) -> Dict[str, Any]:
    """Basic content policy checking."""
    issues = []
    
    # Check for excessive repetition (spam)
    if len(set(text.lower())) < max(3, len(text) // 20):
        issues.append("excessive_repetition")
    
    # Check for extremely long messages
    if len(text) > MAX_QUERY_LENGTH:
        issues.append("message_too_long")
    
    # Check for potential prompt injection
    injection_patterns = [
        r'ignore.*previous.*instructions',
        r'forget.*above',
        r'new.*instructions',
        r'system.*prompt',
        r'jailbreak',
        r'pretend.*you.*are'
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, text.lower()):
            issues.append("potential_prompt_injection")
            break
    
    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "risk_level": "high" if issues else "low"
    }


# Export security components
__all__ = [
    'SecurityMiddleware',
    'InputSanitizer', 
    'RateLimiter',
    'AuthenticationStub',
    'get_current_user',
    'require_permission',
    'require_tier',
    'SecureTextInput',
    'SecureChatInput',
    'check_content_policy',
    'SecurityViolation',
    'rate_limiter',
    'auth_stub',
    'security'
]
