# AI Search System - Developer Onboarding Guide

## 📚 Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [Core Classes & Components](#core-classes--components)
4. [API Reference](#api-reference)
5. [Development Workflow](#development-workflow)
6. [Configuration Guide](#configuration-guide)
7. [Testing Framework](#testing-framework)
8. [Troubleshooting](#troubleshooting)
9. [Contributing Guidelines](#contributing-guidelines)

---

## 🎯 System Overview

### Core Philosophy
**Intelligence lives in APIs, not interfaces.** The AI Search System treats LLMs as interchangeable graph nodes within LangGraph orchestration, prioritizing local inference for cost efficiency and using dual-layer metadata infrastructure for speed and continuous learning.

### Key Principles
- **LLMs are workers, not rulers** - Models are interchangeable processors
- **LangGraph is the conductor** - Orchestrates intelligent workflows
- **APIs are the intelligence layer** - Chat is just one interface
- **85% local inference** - Cost-efficient via Ollama with API fallbacks
- **Metadata-driven learning** - Pattern recognition over ML overhead

### Technology Stack
```yaml
Backend: FastAPI + Python 3.11+
Orchestration: LangGraph
Local LLMs: Ollama
Hot Cache: Redis
API LLMs: OpenAI, Anthropic (fallback)
Containerization: Docker + Compose
Testing: pytest + httpx
Monitoring: Prometheus + Grafana
```

---

## 🏗️ Architecture Deep Dive

### Layer Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Interface Layer                          │
│  Web Chat │ Mobile App │ Slack Bot │ API Integrations      │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  LangGraph Intelligence APIs                │
│  /chat/stream │ /search/analyze │ /research/deep-dive      │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│               Graph Orchestration Layer                     │
│  Chat Graph │ Search Graph │ Analysis Graph │ Synthesis    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                Model Execution Layer                        │
│  Ollama (Local) │ OpenAI/Claude (Fallback) │ Specialized   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Dual-Layer Metadata System                     │
│  Hot Cache (Redis) │ Cold Storage (ClickHouse)             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow
1. **Request** → API Endpoint
2. **Authentication** → Middleware validates user
3. **Rate Limiting** → Check user tier limits
4. **Graph Selection** → Route to appropriate graph
5. **State Creation** → Initialize GraphState
6. **Node Execution** → Process through graph nodes
7. **Model Inference** → Local or API model calls
8. **Cache Updates** → Store patterns and results
9. **Response** → Return formatted response

---

## 🔧 Core Classes & Components

### 1. Configuration Management

#### `Settings` (app/core/config.py)
Central configuration class using Pydantic BaseSettings.

```python
class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "AI Search System"
    debug: bool = False
    environment: str = "development"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    allowed_origins: List[str] = ["http://localhost:3000"]
    
    # Ollama Configuration
    ollama_host: str = "http://localhost:11434"
    ollama_timeout: int = 60
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    redis_max_connections: int = 20
    
    # Model Configuration
    default_model: str = "phi:mini"
    fallback_model: str = "llama2:7b"
    max_concurrent_models: int = 3
    
    # Cost & Budget
    cost_per_api_call: float = 0.008
    default_monthly_budget: float = 20.0
    cost_tracking_enabled: bool = True
    
    # Performance Targets
    target_response_time: float = 2.5
    target_local_processing: float = 0.85
    target_cache_hit_rate: float = 0.80
```

**Key Methods:**
- `get_settings()` - Cached settings retrieval

**Configuration Constants:**
```python
MODEL_ASSIGNMENTS = {
    "simple_classification": "phi:mini",
    "qa_and_summary": "llama2:7b",
    "analytical_reasoning": "mistral:7b",
    "deep_research": "llama2:13b",
    "code_tasks": "codellama",
    "multilingual": "aya:8b"
}

PRIORITY_TIERS = {
    "T0": ["phi:mini"],        # Always loaded (hot)
    "T1": ["llama2:7b"],       # Loaded on first use, kept warm
    "T2": ["llama2:13b", "mistral:7b"]  # Load/unload per request
}

RATE_LIMITS = {
    "free": {"requests_per_minute": 10, "cost_per_month": 20.0},
    "pro": {"requests_per_minute": 100, "cost_per_month": 500.0},
    "enterprise": {"requests_per_minute": 1000, "cost_per_month": 5000.0}
}
```

---

### 2. Graph System (LangGraph)

#### `GraphState` (app/graphs/base.py)
Shared state object that flows through all graph nodes.

```python
@dataclass
class GraphState:
    """Shared state across all graphs"""
    
    # Core request data
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    original_query: str = ""
    
    # Processing context
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    search_results: List[Dict[str, Any]] = field(default_factory=list)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    
    # User preferences and constraints
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    cost_budget_remaining: float = 20.0
    max_execution_time: float = 30.0
    quality_requirement: str = "balanced"  # minimal, balanced, high, premium
    
    # Execution metadata
    start_time: datetime = field(default_factory=datetime.now)
    execution_path: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    costs_incurred: Dict[str, float] = field(default_factory=dict)
    
    # Final output
    final_response: str = ""
    response_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
```

**Key Methods:**
- `add_execution_step(step_name, result)` - Track node execution
- `calculate_total_cost()` - Sum all costs incurred
- `calculate_total_time()` - Calculate execution duration
- `get_avg_confidence()` - Average confidence across nodes

#### `NodeResult` (app/graphs/base.py)
Standard result format for all graph nodes.

```python
class NodeResult(BaseModel):
    """Standard result format for graph nodes"""
    success: bool
    data: Dict[str, Any] = {}
    confidence: float = 0.0
    execution_time: float = 0.0
    cost: float = 0.0
    model_used: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}
```

#### `BaseGraphNode` (app/graphs/base.py)
Abstract base class for all graph nodes.

```python
class BaseGraphNode(ABC):
    """Base class for all graph nodes"""
    
    def __init__(self, name: str, node_type: str = "processing"):
        self.name = name
        self.node_type = node_type
        self.logger = structlog.get_logger(f"node.{name}")
    
    @abstractmethod
    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        """Execute the node logic"""
        pass
    
    async def __call__(self, state: GraphState, **kwargs) -> GraphState:
        """Node execution wrapper with error handling and timing"""
        # Implementation handles timing, error handling, state updates
```

#### `BaseGraph` (app/graphs/base.py)
Abstract base class for all graph implementations.

```python
class BaseGraph(ABC):
    """Base class for all graph implementations"""
    
    def __init__(self, graph_type: GraphType, name: str):
        self.graph_type = graph_type
        self.name = name
        self.logger = structlog.get_logger(f"graph.{name}")
        self.nodes: Dict[str, BaseGraphNode] = {}
        self.graph: Optional[StateGraph] = None
    
    @abstractmethod
    def define_nodes(self) -> Dict[str, BaseGraphNode]:
        """Define graph nodes"""
        return {
            "start": StartNode(),
            "my_processor": MyNewNode(self.dependency_service),
            "end": EndNode(),
            "error_handler": ErrorHandlerNode()
        }
    
    def define_edges(self) -> List[tuple]:
        """Define graph edges"""
        return [
            ("start", "my_processor"),
            ("my_processor", "end"),
            ("my_processor", self._error_condition, {
                "error": "error_handler",
                "continue": "end"
            }),
            ("error_handler", "end")
        ]
    
    def _error_condition(self, state: GraphState) -> str:
        """Check for errors"""
        if state.errors:
            return "error"
        return "continue"
```

#### 3. Adding a New API Endpoint

```python
# app/api/my_new_endpoint.py
from fastapi import APIRouter, Depends, HTTPException
from app.schemas.requests import MyNewRequest
from app.schemas.responses import MyNewResponse

router = APIRouter()

@router.post("/my-endpoint", response_model=MyNewResponse)
async def my_new_endpoint(
    request: MyNewRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """New endpoint implementation"""
    try:
        # Process request
        result = await process_request(request)
        
        return MyNewResponse(
            status="success",
            data=result,
            metadata={
                "query_id": str(uuid.uuid4()),
                "execution_time": 1.0,
                "cost": 0.005
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Register in main.py
from app.api.my_new_endpoint import router as my_new_router
app.include_router(my_new_router, prefix="/api/v1/my-new", tags=["my-new"])
```

#### 4. Adding Request/Response Schemas

```python
# app/schemas/requests.py - Add new request
class MyNewRequest(BaseModel):
    """My new request schema"""
    input_data: str = Field(..., description="Input data")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('input_data')
    def validate_input(cls, v):
        if len(v) < 1:
            raise ValueError("Input data cannot be empty")
        return v

# app/schemas/responses.py - Add new response
class MyNewResponse(BaseResponse):
    """My new response schema"""
    data: Dict[str, Any] = Field(..., description="Response data")
```

### Testing Guidelines

#### Unit Tests
```python
# tests/test_my_new_feature.py
import pytest
from app.graphs.my_new_node import MyNewNode
from app.graphs.base import GraphState

@pytest.mark.asyncio
async def test_my_new_node():
    """Test my new node"""
    # Setup
    mock_service = MockDependencyService()
    node = MyNewNode(mock_service)
    state = GraphState(original_query="test query")
    
    # Execute
    result = await node.execute(state)
    
    # Assert
    assert result.success is True
    assert result.confidence > 0.7
    assert "processed_data" in result.data

@pytest.mark.asyncio
async def test_my_new_node_error_handling():
    """Test error handling"""
    # Setup failing service
    failing_service = FailingDependencyService()
    node = MyNewNode(failing_service)
    state = GraphState(original_query="test query")
    
    # Execute
    result = await node.execute(state)
    
    # Assert
    assert result.success is False
    assert result.error is not None
```

#### Integration Tests
```python
# tests/integration/test_my_new_api.py
import pytest
from fastapi.testclient import TestClient

@pytest.mark.integration
def test_my_new_endpoint_integration():
    """Test full API integration"""
    client = TestClient(app)
    
    payload = {
        "input_data": "test input",
        "options": {"param1": "value1"}
    }
    
    response = client.post("/api/v1/my-new/my-endpoint", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data
```

---

## ⚙️ Configuration Guide

### Environment Variables

#### Core Configuration
```bash
# Environment
ENVIRONMENT=development          # development, production, testing
DEBUG=true                      # Enable debug mode
LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR

# API Configuration
API_HOST=0.0.0.0               # API host
API_PORT=8000                  # API port
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000

# Service URLs
REDIS_URL=redis://localhost:6379
OLLAMA_HOST=http://localhost:11434

# Model Configuration
DEFAULT_MODEL=phi:mini          # Default model for simple tasks
FALLBACK_MODEL=llama2:7b       # Fallback when primary fails
MAX_CONCURRENT_MODELS=3        # Maximum models loaded simultaneously

# Performance Targets
TARGET_RESPONSE_TIME=2.5       # Target response time in seconds
TARGET_LOCAL_PROCESSING=0.85   # Target percentage of local processing
TARGET_CACHE_HIT_RATE=0.80     # Target cache hit rate

# Cost Management
DEFAULT_MONTHLY_BUDGET=100.0   # Default user budget in INR
COST_TRACKING_ENABLED=true     # Enable cost tracking
COST_PER_API_CALL=0.008       # Cost per external API call

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60      # Default rate limit
RATE_LIMIT_BURST=10           # Burst allowance

# Security
JWT_SECRET_KEY=your-secret-key-here  # Change in production!
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=24

# External APIs (optional)
BRAVE_SEARCH_API_KEY=         # Brave search API key
ZEROWS_API_KEY=              # Zerows scraping API key
OPENAI_API_KEY=              # OpenAI API key (fallback)
ANTHROPIC_API_KEY=           # Anthropic API key (fallback)
```

#### Production Configuration
```bash
# Production-specific settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
LOG_FORMAT=json

# Enhanced security
JWT_SECRET_KEY=<strong-random-key>
ALLOWED_ORIGINS=https://yourdomain.com

# Performance optimization
REDIS_MAX_CONNECTIONS=50
OLLAMA_TIMEOUT=30
RATE_LIMIT_PER_MINUTE=100

# Resource limits
MAX_CONCURRENT_MODELS=5
MODEL_MEMORY_THRESHOLD=0.7
```

### Configuration Classes

#### Development Settings
```python
class DevelopmentSettings(Settings):
    """Development environment settings"""
    debug: bool = True
    log_level: str = "DEBUG"
    environment: str = "development"
    
    # More permissive for development
    rate_limit_per_minute: int = 1000
    cost_tracking_enabled: bool = False
```

#### Production Settings
```python
class ProductionSettings(Settings):
    """Production environment settings"""
    debug: bool = False
    log_level: str = "INFO"
    environment: str = "production"
    
    # More restrictive for production
    redis_max_connections: int = 50
    rate_limit_per_minute: int = 100
    target_response_time: float = 2.0
```

### Model Configuration

#### Model Assignments
```python
MODEL_ASSIGNMENTS = {
    "simple_classification": "phi:mini",      # Ultra-fast, minimal resources
    "qa_and_summary": "llama2:7b",           # Balanced performance
    "analytical_reasoning": "mistral:7b",     # Logic and reasoning
    "deep_research": "llama2:13b",           # Complex understanding
    "code_tasks": "codellama",               # Specialized programming
    "multilingual": "aya:8b"                 # Language diversity
}
```

#### Priority Tiers
```python
PRIORITY_TIERS = {
    "T0": ["phi:mini"],                      # Always loaded (hot)
    "T1": ["llama2:7b"],                     # Loaded on first use, kept warm
    "T2": ["llama2:13b", "mistral:7b"]       # Load/unload per request
}
```

#### Cost Structure
```python
API_COSTS = {
    # Local models (free)
    "phi:mini": 0.0,
    "llama2:7b": 0.0,
    "mistral:7b": 0.0,
    "llama2:13b": 0.0,
    "codellama": 0.0,
    
    # API models (paid)
    "gpt-4": 0.06,                          # ₹0.06 per request
    "claude-haiku": 0.01,                   # ₹0.01 per request
    
    # External services
    "brave_search": 0.008,                  # ₹0.008 per search
    "zerows_scraping": 0.02                 # ₹0.02 per scrape
}
```

---

## 🧪 Testing Framework

### Test Structure
```
tests/
├── conftest.py              # Test configuration and fixtures
├── unit/                    # Unit tests for individual components
│   ├── test_models.py      # Model management tests
│   ├── test_cache.py       # Cache functionality tests
│   ├── test_graphs.py      # Graph logic tests
│   └── test_config.py      # Configuration tests
├── integration/             # Integration tests
│   ├── test_api_integration.py
│   └── test_graph_integration.py
├── test_health.py          # Health endpoint tests
├── test_chat_api.py        # Chat API tests
└── test_middleware.py      # Middleware tests
```

### Test Categories

#### Unit Tests
Focus on individual components in isolation:

```python
# Example: Testing model selection logic
@pytest.mark.asyncio
async def test_model_selection():
    """Test model selection for different tasks"""
    manager = ModelManager("http://localhost:11434")
    
    # Test simple classification
    model = manager.select_optimal_model("simple_classification", "minimal")
    assert model == "phi:mini"
    
    # Test complex reasoning
    model = manager.select_optimal_model("analytical_reasoning", "high")
    assert model == "mistral:7b"
    
    # Test premium quality
    model = manager.select_optimal_model("deep_research", "premium")
    assert model == "llama2:13b"
```

#### Integration Tests
Test complete workflows:

```python
# Example: Testing chat flow
@pytest.mark.integration
async def test_chat_flow_integration():
    """Test complete chat conversation flow"""
    # This requires running services
    async with TestClient(app) as client:
        # Send first message
        response1 = await client.post("/api/v1/chat/complete", json={
            "message": "Hello, my name is Alice",
            "session_id": "test_session"
        })
        assert response1.status_code == 200
        
        # Send follow-up message
        response2 = await client.post("/api/v1/chat/complete", json={
            "message": "What's my name?",
            "session_id": "test_session"
        })
        assert response2.status_code == 200
        
        # Check if context was maintained
        data = response2.json()
        assert "Alice" in data["data"]["response"]
```

### Test Fixtures

#### Mock Services
```python
# conftest.py
@pytest.fixture
async def mock_model_manager():
    """Mock model manager for testing"""
    class MockModelManager:
        async def generate(self, model_name, prompt, **kwargs):
            return ModelResult(
                success=True,
                text=f"Mock response from {model_name}",
                cost=0.001,
                execution_time=0.5,
                model_used=model_name
            )
        
        def select_optimal_model(self, task_type, quality="balanced"):
            return MODEL_ASSIGNMENTS.get(task_type, "llama2:7b")
    
    return MockModelManager()

@pytest.fixture
async def mock_cache_manager():
    """Mock cache manager for testing"""
    class MockCacheManager:
        def __init__(self):
            self._cache = {}
        
        async def get(self, key, default=None):
            return self._cache.get(key, default)
        
        async def set(self, key, value, ttl=None):
            self._cache[key] = value
            return True
    
    return MockCacheManager()
```

### Running Tests

#### Basic Test Commands
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_chat_api.py

# Run specific test function
pytest tests/test_chat_api.py::test_chat_complete_basic

# Run tests with coverage
pytest --cov=app tests/

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/ -m integration

# Run tests in parallel
pytest -n 4
```

#### Test Categories
```bash
# Run fast tests only
pytest -m "not slow"

# Run integration tests
pytest -m integration

# Run specific markers
pytest -m "unit and not slow"
```

### Test Configuration

#### pytest.ini
```ini
[tool:pytest]
minversion = 6.0
addopts = 
    -ra
    --strict-markers
    --disable-warnings
    --tb=short
testpaths = tests
markers =
    integration: Integration tests (require running services)
    slow: Slow tests (> 1 second)
    unit: Unit tests
    api: API tests
asyncio_mode = auto
```

#### Coverage Configuration
```ini
# .coveragerc
[run]
source = app
omit = 
    app/main.py
    */tests/*
    */venv/*
    */__pycache__/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Ollama Connection Issues

**Problem**: `ConnectionError: Cannot connect to Ollama`

**Solutions**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
docker-compose up ollama

# Check Ollama logs
docker-compose logs ollama

# Pull required models
docker-compose exec ollama ollama pull phi:mini
docker-compose exec ollama ollama pull llama2:7b
```

**Environment Variables**:
```bash
OLLAMA_HOST=http://localhost:11434
OLLAMA_TIMEOUT=60
OLLAMA_MAX_RETRIES=3
```

#### 2. Redis Connection Issues

**Problem**: `ConnectionError: Cannot connect to Redis`

**Solutions**:
```bash
# Check Redis status
docker-compose ps redis

# Start Redis
docker-compose up -d redis

# Check Redis connectivity
redis-cli ping

# Clear Redis cache if needed
redis-cli FLUSHALL
```

**Environment Variables**:
```bash
REDIS_URL=redis://localhost:6379
REDIS_MAX_CONNECTIONS=20
REDIS_TIMEOUT=5
```

#### 3. Model Loading Issues

**Problem**: `Model 'llama2:7b' not found or failed to load`

**Solutions**:
```bash
# List available models
docker-compose exec ollama ollama list

# Pull missing model
docker-compose exec ollama ollama pull llama2:7b

# Check model status
curl http://localhost:11434/api/tags

# Restart model manager
docker-compose restart ai-search-api
```

**Debug Steps**:
```python
# Check model status in logs
import structlog
logger = structlog.get_logger(__name__)

# Enable debug logging
LOG_LEVEL=DEBUG

# Check model registry
GET /api/v1/models/status
```

#### 4. Memory Issues

**Problem**: `OutOfMemoryError` or slow model loading

**Solutions**:
```bash
# Check system memory
free -h

# Monitor Docker memory usage
docker stats

# Reduce concurrent models
MAX_CONCURRENT_MODELS=2

# Use smaller models
DEFAULT_MODEL=phi:mini
```

**Memory Optimization**:
```python
# Model tier configuration
PRIORITY_TIERS = {
    "T0": ["phi:mini"],           # Keep only essential models hot
    "T1": [],                     # Reduce warm models
    "T2": ["llama2:7b", "mistral:7b"]  # Load on demand
}

# Memory threshold
MODEL_MEMORY_THRESHOLD=0.6  # Unload at 60% memory usage
```

#### 5. Rate Limiting Issues

**Problem**: `429 Too Many Requests`

**Solutions**:
```bash
# Check current rate limits
GET /api/v1/user/stats

# Increase rate limits for development
RATE_LIMIT_PER_MINUTE=1000

# Clear rate limit cache
redis-cli DEL rate:user_id

# Use different user ID
# Change Authorization header
```

#### 6. API Response Issues

**Problem**: Slow responses or timeouts

**Debug Steps**:
```bash
# Check API health
curl http://localhost:8000/health

# Monitor response times
curl -w "%{time_total}" http://localhost:8000/api/v1/chat/complete

# Check logs
docker-compose logs ai-search-api

# Enable debug mode
DEBUG=true
LOG_LEVEL=DEBUG
```

**Performance Tuning**:
```python
# Reduce timeouts for testing
OLLAMA_TIMEOUT=30
GRAPH_TIMEOUT=15

# Enable more caching
CACHE_TTL_DEFAULT=7200  # 2 hours

# Use faster models
DEFAULT_MODEL=phi:mini
```

### Debugging Tools

#### Logging Configuration
```python
# Enable debug logging
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
```

#### Health Check Endpoints
```bash
# System health
GET /health

# Detailed metrics
GET /metrics

# Model status
GET /api/v1/models/status

# Cache statistics
GET /api/v1/cache/stats
```

#### Development Tools
```bash
# Redis GUI
http://localhost:8081  # Redis Commander

# API Documentation
http://localhost:8000/docs  # Swagger UI
http://localhost:8000/redoc  # ReDoc

# Metrics Dashboard (if enabled)
http://localhost:3000  # Grafana
http://localhost:9090  # Prometheus
```

### Error Codes Reference

| Code | Description | Solution |
|------|-------------|----------|
| `RATE_LIMIT_EXCEEDED` | Too many requests | Wait or upgrade tier |
| `BUDGET_EXCEEDED` | Cost budget exhausted | Add budget or wait for reset |
| `MODEL_UNAVAILABLE` | Model not loaded | Check Ollama status |
| `INVALID_INPUT` | Request validation failed | Check request format |
| `TIMEOUT` | Request took too long | Reduce complexity or increase timeout |
| `AUTHENTICATION_REQUIRED` | Missing or invalid token | Provide valid authorization |
| `PERMISSION_DENIED` | Insufficient permissions | Upgrade user tier |
| `INTERNAL_ERROR` | Server error | Check logs for details |

---

## 🤝 Contributing Guidelines

### Development Process

#### 1. Setting Up Development Environment
```bash
# Fork the repository
git clone https://github.com/your-username/ai-search-system.git
cd ai-search-system

# Create development branch
git checkout -b feature/my-awesome-feature

# Set up environment
./scripts/setup.sh

# Start development services
./scripts/dev.sh start
```

#### 2. Code Standards

**Python Style Guide**:
- Follow PEP 8
- Use Black for formatting
- Use isort for import sorting
- Use type hints
- Maximum line length: 88 characters

**Code Quality Tools**:
```bash
# Format code
black app/ tests/
isort app/ tests/

# Type checking
mypy app/

# Linting
flake8 app/ tests/

# Run all quality checks
./scripts/dev.sh lint
```

**Naming Conventions**:
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`
- Async functions: `async def` prefix where applicable

#### 3. Documentation Standards

**Docstring Format**:
```python
def my_function(param1: str, param2: int = 10) -> bool:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of param1
        param2: Description of param2 with default value
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param1 is empty
        ConnectionError: When service is unavailable
    
    Example:
        >>> result = my_function("test", 5)
        >>> print(result)
        True
    """
```

**Class Documentation**:
```python
class MyClass:
    """
    Brief description of the class purpose.
    
    This class handles... and provides methods for...
    
    Attributes:
        attr1: Description of attribute1
        attr2: Description of attribute2
    
    Example:
        >>> instance = MyClass("config")
        >>> result = instance.process()
    """
    
    def __init__(self, config: str):
        """Initialize MyClass with configuration."""
        self.config = config
```

#### 4. Testing Requirements

**Test Coverage**: Maintain > 80% test coverage

**Required Tests**:
- Unit tests for all new functions/classes
- Integration tests for API endpoints
- Error handling tests
- Performance tests for critical paths

**Test Naming**:
```python
def test_function_name_should_behavior_when_condition():
    """Test that function_name behaves correctly when condition."""
    # Arrange
    setup_data = create_test_data()
    
    # Act
    result = function_name(setup_data)
    
    # Assert
    assert result.success is True
    assert result.data == expected_data
```

#### 5. Commit Message Format

Use conventional commit format:

```
type(scope): brief description

Detailed explanation of changes made and why.

Fixes #123
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(chat): add streaming response support

Implement Server-Sent Events for real-time chat responses.
Includes proper error handling and connection management.

Fixes #45

fix(cache): resolve Redis connection timeout

Increase connection timeout and add retry logic for
Redis operations during high load scenarios.

Fixes #67

docs(api): update authentication documentation

Add examples for JWT token usage and rate limiting.
Include troubleshooting section for common auth issues.
```

#### 6. Pull Request Process

**Before Creating PR**:
```bash
# Ensure all tests pass
./scripts/dev.sh test

# Run linting
./scripts/dev.sh lint

# Update documentation if needed
# Add/update tests for new features
# Update CHANGELOG.md
```

**PR Template**:
```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass locally
- [ ] No breaking changes (or properly documented)

## Screenshots (if applicable)

## Related Issues
Fixes #123
Related to #456
```

### Architecture Guidelines

#### 1. Adding New Components

**Graph Nodes**:
- Inherit from `BaseGraphNode`
- Implement `execute()` method
- Return `NodeResult` object
- Handle errors gracefully
- Add appropriate logging

**API Endpoints**:
- Use FastAPI router pattern
- Include request/response schemas
- Add authentication/authorization
- Implement rate limiting
- Include proper error handling

**Services**:
- Follow dependency injection pattern
- Use async/await for I/O operations
- Implement health checks
- Add metrics collection

#### 2. Performance Considerations

**Async/Await Usage**:
```python
# Good: Proper async/await usage
async def fetch_data(client: AsyncClient) -> Dict:
    async with client.get("/api/data") as response:
        return await response.json()

# Bad: Blocking operations in async function
async def bad_fetch_data():
    response = requests.get("/api/data")  # Blocking!
    return response.json()
```

**Caching Strategy**:
```python
# Good: Check cache first, then compute
async def expensive_operation(key: str) -> Result:
    cached = await cache.get(f"operation:{key}")
    if cached:
        return cached
    
    result = await compute_expensive_result(key)
    await cache.set(f"operation:{key}", result, ttl=3600)
    return result
```

**Error Handling**:
```python
# Good: Graceful error handling with fallbacks
async def resilient_model_call(prompt: str) -> ModelResult:
    try:
        return await primary_model.generate(prompt)
    except ModelUnavailableError:
        logger.warning("Primary model unavailable, using fallback")
        return await fallback_model.generate(prompt)
    except Exception as e:
        logger.error(f"Model call failed: {e}")
        return ModelResult(success=False, error=str(e))
```

#### 3. Security Guidelines

**Input Validation**:
```python
# Use Pydantic for validation
class SecureRequest(BaseModel):
    user_input: str = Field(..., max_length=8000)
    session_id: str = Field(..., regex=r'^[a-zA-Z0-9_-]+[str, BaseGraphNode]:
        """Define the nodes for this graph"""
        pass
    
    @abstractmethod
    def define_edges(self) -> List[tuple]:
        """Define the edges (connections) between nodes"""
        pass
    
    def build(self):
        """Build the LangGraph instance"""
    
    async def execute(self, state: GraphState) -> GraphState:
        """Execute the graph with the given state"""
```

#### `ChatGraph` (app/graphs/chat_graph.py)
Main implementation for conversational AI.

**Nodes:**
- `ContextManagerNode` - Manages conversation history and context
- `IntentClassifierNode` - Classifies user intent for routing
- `ResponseGeneratorNode` - Generates final response using appropriate model
- `CacheUpdateNode` - Updates conversation cache

**Key Features:**
- Intelligent model selection based on intent
- Context-aware conversation management
- User preference inference
- Cost-aware routing decisions

---

### 3. Model Management

#### `ModelManager` (app/models/manager.py)
Central hub for all model operations and lifecycle management.

```python
class ModelManager:
    """Manages model lifecycle and selection"""
    
    def __init__(self, ollama_host: str, cache_manager: Optional[CacheManager] = None):
        self.ollama_client = OllamaClient(ollama_host)
        self.cache_manager = cache_manager
        self.models: Dict[str, ModelInfo] = {}
        self.loaded_models: Dict[str, datetime] = {}
        self._loading_locks: Dict[str, asyncio.Lock] = {}
    
    async def initialize(self):
        """Initialize the model manager"""
    
    async def generate(
        self, model_name: str, prompt: str, max_tokens: int = 300,
        temperature: float = 0.7, fallback: bool = True
    ) -> ModelResult:
        """Generate text using specified model with fallback"""
    
    def select_optimal_model(self, task_type: str, quality_requirement: str = "balanced") -> str:
        """Select optimal model for a task"""
    
    async def preload_models(self, model_names: List[str]):
        """Preload critical models"""
```

#### `ModelInfo` (app/models/manager.py)
Model metadata and statistics tracking.

```python
@dataclass
class ModelInfo:
    """Information about a model"""
    name: str
    status: ModelStatus
    last_used: datetime
    load_time: float = 0.0
    total_requests: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    memory_usage: float = 0.0
    tier: str = "T2"
    
    def update_stats(self, execution_time: float, cost: float):
        """Update model statistics"""
```

#### `OllamaClient` (app/models/manager.py)
Async HTTP client for Ollama API communication.

```python
class OllamaClient:
    """Async client for Ollama API"""
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from registry"""
    
    async def generate(
        self, model_name: str, prompt: str, max_tokens: int = 300,
        temperature: float = 0.7, stop: Optional[List[str]] = None
    ) -> ModelResult:
        """Generate text using a model"""
    
    async def check_model_status(self, model_name: str) -> bool:
        """Check if a model is loaded and ready"""
```

#### `ModelResult` (app/models/manager.py)
Result object from model generation.

```python
@dataclass
class ModelResult:
    """Result from model generation"""
    success: bool
    text: str = ""
    cost: float = 0.0
    execution_time: float = 0.0
    model_used: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
```

---

### 4. Cache Management

#### `CacheManager` (app/cache/redis_client.py)
Redis-based cache manager for the hot layer of metadata infrastructure.

```python
class CacheManager:
    """Redis-based cache manager for hot layer"""
    
    def __init__(self, redis_url: str, max_connections: int = 20):
        self.redis_url = redis_url
        self.max_connections = max_connections
        self.redis: Optional[redis.Redis] = None
        self.metrics = CacheMetrics()
        self._local_cache: Dict[str, tuple[Any, datetime]] = {}
    
    async def initialize(self):
        """Initialize Redis connection"""
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
```

**Specialized Methods:**
```python
# Routing and patterns
async def get_cached_route(self, query: str) -> Optional[List[str]]
async def cache_successful_route(self, query: str, route: List[str], cost: float)
async def get_user_pattern(self, user_id: str) -> Optional[Dict[str, Any]]
async def update_user_pattern(self, user_id: str, pattern_data: Dict[str, Any])

# Conversation management
async def get_conversation_history(self, session_id: str) -> Optional[List[Dict[str, Any]]]
async def update_conversation_history(self, session_id: str, history: List[Dict[str, Any]])

# Budget and rate limiting
async def get_remaining_budget(self, user_id: str) -> float
async def deduct_budget(self, user_id: str, cost: float) -> float
async def check_rate_limit(self, user_id: str, limit_per_minute: int = 60) -> tuple[bool, int]

# Performance optimization
async def cache_performance_hint(self, query_type: str, expected_time: float, expected_confidence: float)
async def cache_optimal_model(self, task_type: str, model_name: str, success_rate: float)
```

#### `CacheKey` (app/cache/redis_client.py)
Cache key management and generation utilities.

```python
class CacheKey:
    """Cache key constants and generators"""
    
    # Key prefixes
    ROUTE_PREFIX = "route:"
    PATTERN_PREFIX = "pattern:"
    CONVERSATION_PREFIX = "conv:"
    BUDGET_PREFIX = "budget:"
    RATE_PREFIX = "rate:"
    
    @staticmethod
    def query_hash(query: str) -> str:
        """Generate consistent hash for query"""
    
    @staticmethod
    def route_key(query: str) -> str:
        """Generate route cache key"""
    
    @staticmethod
    def conversation_key(session_id: str) -> str:
        """Generate conversation history key"""
```

#### `CacheMetrics` (app/cache/redis_client.py)
Performance metrics tracking for cache operations.

```python
class CacheMetrics(BaseModel):
    """Cache performance metrics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    avg_response_time: float = 0.0
    memory_usage: int = 0
    
    def update_hit(self, response_time: float):
        """Update metrics for cache hit"""
    
    def update_miss(self, response_time: float):
        """Update metrics for cache miss"""
```

---

### 5. API Layer

#### API Endpoints

**Chat Endpoints** (app/api/chat.py):
```python
@router.post("/stream")
async def chat_stream(request: ChatStreamRequest, ...) -> StreamingResponse:
    """Streaming chat endpoint compatible with OpenAI API format"""

@router.post("/complete") 
async def chat_complete(request: ChatRequest, ...) -> ChatResponse:
    """Non-streaming chat completion endpoint"""

@router.get("/history/{session_id}")
async def get_conversation_history(session_id: str, ...) -> Dict:
    """Get conversation history for a session"""

@router.delete("/history/{session_id}")
async def clear_conversation_history(session_id: str, ...) -> Dict:
    """Clear conversation history for a session"""
```

#### Request Schemas (app/schemas/requests.py)

**ChatRequest:**
```python
class ChatRequest(BaseModel):
    """Non-streaming chat request"""
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: Optional[str] = None
    context: Optional[Context] = None
    constraints: Optional[Constraints] = None
```

**ChatStreamRequest:**
```python
class ChatStreamRequest(BaseModel):
    """Streaming chat request (OpenAI-compatible)"""
    messages: List[ChatMessage] = Field(...)
    session_id: Optional[str] = None
    model: Optional[str] = Field("auto")
    max_tokens: Optional[int] = Field(300)
    temperature: Optional[float] = Field(0.7)
    stream: bool = Field(True)
```

**Constraints:**
```python
class Constraints(BaseModel):
    """Request constraints"""
    max_cost: Optional[float] = Field(0.05, description="Maximum cost in INR")
    max_time: Optional[float] = Field(5.0, description="Maximum execution time in seconds")
    quality_requirement: Optional[str] = Field("balanced", description="Quality level: minimal, balanced, high, premium")
    force_local_only: Optional[bool] = Field(False, description="Force local models only")
```

#### Response Schemas (app/schemas/responses.py)

**ChatResponse:**
```python
class ChatResponse(BaseResponse):
    """Chat completion response"""
    data: Dict[str, Any] = Field(..., description="Chat response data")
    
    # Inherits from BaseResponse:
    # - status: str
    # - metadata: ResponseMetadata
    # - cost_prediction: CostPrediction  
    # - developer_hints: DeveloperHints
```

**ResponseMetadata:**
```python
class ResponseMetadata(BaseModel):
    """Response metadata"""
    query_id: str
    execution_time: float
    cost: float
    models_used: List[str]
    confidence: float
    cached: bool
    timestamp: Optional[str]
```

**CostPrediction:**
```python
class CostPrediction(BaseModel):
    """Cost prediction and optimization"""
    estimated_cost: float
    cost_breakdown: List[CostBreakdown]
    savings_tips: List[str]
    alternative_workflows: Optional[List[Dict[str, Any]]]
```

---

### 6. Middleware & Security

#### `RateLimitMiddleware` (app/api/middleware.py)
Request rate limiting and performance tracking.

```python
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting and add performance headers"""
```

#### `CostTrackingMiddleware` (app/api/middleware.py)
Cost attribution and budget tracking.

```python
class CostTrackingMiddleware(BaseHTTPMiddleware):
    """Cost tracking middleware"""
    
    async def dispatch(self, request: Request, call_next):
        """Track costs per request"""
```

#### Authentication Functions (app/api/middleware.py)

```python
async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Dict[str, Any]:
    """Get current user from token or create anonymous user"""

async def check_rate_limit(user_id: str, cache_manager: CacheManager, endpoint: str = "general") -> bool:
    """Check if user is within rate limits"""

async def track_cost(user_id: str, cost: float, cache_manager: CacheManager) -> float:
    """Track and deduct cost from user budget"""

async def check_budget(user_id: str, estimated_cost: float, cache_manager: CacheManager) -> bool:
    """Check if user has sufficient budget"""
```

#### Permission Decorators:

```python
def require_permission(permission: str):
    """Decorator to require specific permission"""

def require_tier(min_tier: str):
    """Decorator to require minimum user tier"""
```

---

## 📡 API Reference

### Base URLs
- **Development**: `http://localhost:8000`
- **Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

### Authentication
```bash
# Bearer token (development)
Authorization: Bearer <your-token>

# Anonymous access (limited)
# No header required for basic testing
```

### Core Endpoints

#### Chat Completion
```http
POST /api/v1/chat/complete
Content-Type: application/json

{
  "message": "Explain async/await in Python",
  "session_id": "session_123",
  "constraints": {
    "max_cost": 0.10,
    "quality_requirement": "high"
  }
}
```

Response:
```json
{
  "status": "success",
  "data": {
    "response": "Async/await in Python allows you to write asynchronous code...",
    "session_id": "session_123"
  },
  "metadata": {
    "query_id": "query_456",
    "execution_time": 1.23,
    "cost": 0.008,
    "models_used": ["llama2:7b"],
    "confidence": 0.89,
    "cached": false
  },
  "cost_prediction": {
    "estimated_cost": 0.008,
    "cost_breakdown": [
      {"step": "classification", "model": "phi:mini", "cost": 0.0},
      {"step": "generation", "model": "llama2:7b", "cost": 0.0}
    ],
    "savings_tips": ["Use phi:mini for simpler questions"]
  }
}
```

#### Streaming Chat
```http
POST /api/v1/chat/stream
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "What is machine learning?"}
  ],
  "stream": true
}
```

Response (Server-Sent Events):
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1677652288,"model":"llama2:7b","choices":[{"index":0,"delta":{"content":"Machine"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1677652288,"model":"llama2:7b","choices":[{"index":0,"delta":{"content":" learning"},"finish_reason":null}]}

data: [DONE]
```

#### System Health
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "components": {
    "cache": "healthy",
    "models": "healthy"
  },
  "version": "1.0.0"
}
```

#### Metrics
```http
GET /metrics
```

Response:
```json
{
  "status": "success",
  "metrics": {
    "cache": {
      "hit_rate": 0.85,
      "total_requests": 1000
    },
    "models": {
      "total_requests": 500,
      "local_percentage": 87.5
    }
  }
}
```

### Error Responses

```json
{
  "status": "error",
  "message": "Rate limit exceeded",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "query_id": "query_789",
  "timestamp": "2025-06-19T10:30:00Z"
}
```

Common Error Codes:
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `BUDGET_EXCEEDED` - Insufficient budget
- `MODEL_UNAVAILABLE` - Model not ready
- `INVALID_INPUT` - Request validation failed
- `TIMEOUT` - Request took too long

---

## 🔧 Development Workflow

### Project Structure
```
ai-search-system/
├── app/                           # Main application code
│   ├── __init__.py
│   ├── main.py                   # FastAPI application entry point
│   ├── api/                      # API layer
│   │   ├── __init__.py
│   │   ├── chat.py              # Chat endpoints
│   │   ├── search.py            # Search endpoints (future)
│   │   └── middleware.py        # Authentication, rate limiting
│   ├── core/                     # Core configuration and utilities
│   │   ├── __init__.py
│   │   ├── config.py            # Settings management
│   │   ├── logging.py           # Logging configuration
│   │   └── security.py          # Security utilities
│   ├── graphs/                   # LangGraph implementations
│   │   ├── __init__.py
│   │   ├── base.py              # Base classes for graphs and nodes
│   │   ├── chat_graph.py        # Chat conversation graph
│   │   ├── search_graph.py      # Search and analysis graph (future)
│   │   └── orchestrator.py      # Master orchestration (future)
│   ├── models/                   # Model management
│   │   ├── __init__.py
│   │   ├── manager.py           # Model registry and lifecycle
│   │   ├── ollama_client.py     # Ollama API client
│   │   └── selectors.py         # Model selection logic
│   ├── cache/                    # Caching layer
│   │   ├── __init__.py
│   │   ├── redis_client.py      # Redis cache implementation
│   │   └── strategies.py        # Caching strategies
│   └── schemas/                  # Request/response models
│       ├── __init__.py
│       ├── requests.py          # API request schemas
│       └── responses.py         # API response schemas
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py             # Test configuration
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── test_*.py              # Individual test files
├── docker/                      # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── prometheus.yml
├── scripts/                     # Development scripts
│   ├── setup.sh               # Environment setup
│   └── dev.sh                 # Development helpers
├── docs/                       # Documentation
│   └── api/                   # API documentation
├── requirements.txt           # Python dependencies
├── requirements-dev.txt       # Development dependencies
├── .env                      # Environment variables
├── .gitignore               # Git ignore rules
└── README.md               # Project README
```

### Development Commands

#### Environment Setup
```bash
# Initial setup
chmod +x scripts/setup.sh scripts/dev.sh
./scripts/setup.sh

# Start development environment
./scripts/dev.sh start

# Stop environment
./scripts/dev.sh stop

# View logs
./scripts/dev.sh logs

# Run tests
./scripts/dev.sh test

# Code linting
./scripts/dev.sh lint

# Pull additional models
./scripts/dev.sh models

# Clean up
./scripts/dev.sh clean
```

#### Local Development
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt

# Run server with hot reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest

# Run with coverage
pytest --cov=app tests/

# Format code
black app/ tests/
isort app/ tests/

# Type checking
mypy app/

# Linting
flake8 app/ tests/
```

### Adding New Features

#### 1. Adding a New Graph Node

```python
# app/graphs/my_new_node.py
from app.graphs.base import BaseGraphNode, GraphState, NodeResult

class MyNewNode(BaseGraphNode):
    def __init__(self, dependency_service):
        super().__init__("my_new_node", "processing")
        self.dependency_service = dependency_service
    
    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        try:
            # Your node logic here
            result_data = await self.dependency_service.process(state.original_query)
            
            return NodeResult(
                success=True,
                confidence=0.8,
                data={"processed_data": result_data},
                cost=0.001,
                model_used="my_model"
            )
        except Exception as e:
            return NodeResult(
                success=False,
                error=f"Processing failed: {str(e)}",
                confidence=0.0
            )
```

#### 2. Adding a New Graph

```python
# app/graphs/my_new_graph.py
from app.graphs.base import BaseGraph, GraphType
from app.graphs.my_new_node import MyNewNode

class MyNewGraph(BaseGraph):
    def __init__(self, dependency_service):
        super().__init__(GraphType.CUSTOM, "my_new_graph")
        self.dependency_service = dependency_service
    
    def define_nodes(self) -> Dict)
    
    @validator('user_input')
    def sanitize_input(cls, v):
        # Remove potentially dangerous content
        return html.escape(v.strip())
```

**Authentication**:
```python
# Always check authentication for protected endpoints
@router.post("/protected-endpoint")
async def protected_endpoint(
    request: ProtectedRequest,
    current_user: Dict = Depends(get_current_user)  # Required
):
    # Endpoint logic here
    pass
```

**Logging Security**:
```python
# Good: Don't log sensitive data
logger.info("User authenticated", user_id=user.id)

# Bad: Logging sensitive information
logger.info("User login", username=username, password=password)  # Never!
```

### Release Process

#### 1. Version Management

Follow Semantic Versioning (SemVer):
- `MAJOR.MINOR.PATCH`
- `MAJOR`: Breaking changes
- `MINOR`: New features (backward compatible)
- `PATCH`: Bug fixes (backward compatible)

#### 2. Release Checklist

**Pre-release**:
- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in relevant files
- [ ] Security review completed
- [ ] Performance testing completed

**Release**:
- [ ] Create release branch
- [ ] Run full test suite
- [ ] Build and test Docker images
- [ ] Create GitHub release
- [ ] Deploy to staging
- [ ] Deploy to production
- [ ] Monitor deployment

**Post-release**:
- [ ] Verify deployment
- [ ] Monitor error rates
- [ ] Update documentation
- [ ] Communicate changes to team

---

## 📋 Quick Reference

### Key Commands
```bash
# Development
./scripts/setup.sh              # Initial setup
./scripts/dev.sh start          # Start services
./scripts/dev.sh logs           # View logs
./scripts/dev.sh test           # Run tests

# Testing
pytest                          # Run all tests
pytest -v                       # Verbose output
pytest --cov=app tests/         # With coverage
pytest -m integration           # Integration tests only

# Code Quality
black app/ tests/               # Format code
flake8 app/ tests/              # Lint code
mypy app/                       # Type checking
isort app/ tests/               # Sort imports
```

### Important URLs
```
API Documentation: http://localhost:8000/docs
Health Check:      http://localhost:8000/health
Redis GUI:         http://localhost:8081
Grafana:          http://localhost:3000
Prometheus:       http://localhost:9090
```

### Configuration Files
```
.env                    # Environment variables
docker-compose.yml      # Service orchestration
requirements.txt        # Python dependencies
pytest.ini             # Test configuration
.coveragerc            # Coverage configuration
```

### Key Classes to Remember
```python
# Core Infrastructure
Settings                # Configuration management
GraphState             # Shared state across graphs
ModelManager           # Model lifecycle management
CacheManager           # Redis caching layer

# Graph System
BaseGraph              # Base class for all graphs
BaseGraphNode          # Base class for all nodes
ChatGraph              # Main chat implementation

# API Layer
ChatRequest/Response   # API schemas
RateLimitMiddleware    # Request limiting
CostTrackingMiddleware # Budget tracking
```

---

This comprehensive guide should help new developers quickly understand the AI Search System architecture, navigate the codebase, and contribute effectively. The system's modular design and clear separation of concerns make it easy to extend and maintain while keeping the core philosophy of "intelligence in APIs, not interfaces."

For additional help, check the `/docs` directory, GitHub issues, or reach out to the development team. Happy coding! 🚀[str, BaseGraphNode]:
        """Define the nodes for this graph"""
        pass
    
    @abstractmethod
    def define_edges(self) -> List[tuple]:
        """Define the edges (connections) between nodes"""
        pass
    
    def build(self):
        """Build the LangGraph instance"""
    
    async def execute(self, state: GraphState) -> GraphState:
        """Execute the graph with the given state"""
```

#### `ChatGraph` (app/graphs/chat_graph.py)
Main implementation for conversational AI.

**Nodes:**
- `ContextManagerNode` - Manages conversation history and context
- `IntentClassifierNode` - Classifies user intent for routing
- `ResponseGeneratorNode` - Generates final response using appropriate model
- `CacheUpdateNode` - Updates conversation cache

**Key Features:**
- Intelligent model selection based on intent
- Context-aware conversation management
- User preference inference
- Cost-aware routing decisions

---

### 3. Model Management

#### `ModelManager` (app/models/manager.py)
Central hub for all model operations and lifecycle management.

```python
class ModelManager:
    """Manages model lifecycle and selection"""
    
    def __init__(self, ollama_host: str, cache_manager: Optional[CacheManager] = None):
        self.ollama_client = OllamaClient(ollama_host)
        self.cache_manager = cache_manager
        self.models: Dict[str, ModelInfo] = {}
        self.loaded_models: Dict[str, datetime] = {}
        self._loading_locks: Dict[str, asyncio.Lock] = {}
    
    async def initialize(self):
        """Initialize the model manager"""
    
    async def generate(
        self, model_name: str, prompt: str, max_tokens: int = 300,
        temperature: float = 0.7, fallback: bool = True
    ) -> ModelResult:
        """Generate text using specified model with fallback"""
    
    def select_optimal_model(self, task_type: str, quality_requirement: str = "balanced") -> str:
        """Select optimal model for a task"""
    
    async def preload_models(self, model_names: List[str]):
        """Preload critical models"""
```

#### `ModelInfo` (app/models/manager.py)
Model metadata and statistics tracking.

```python
@dataclass
class ModelInfo:
    """Information about a model"""
    name: str
    status: ModelStatus
    last_used: datetime
    load_time: float = 0.0
    total_requests: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    memory_usage: float = 0.0
    tier: str = "T2"
    
    def update_stats(self, execution_time: float, cost: float):
        """Update model statistics"""
```

#### `OllamaClient` (app/models/manager.py)
Async HTTP client for Ollama API communication.

```python
class OllamaClient:
    """Async client for Ollama API"""
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from registry"""
    
    async def generate(
        self, model_name: str, prompt: str, max_tokens: int = 300,
        temperature: float = 0.7, stop: Optional[List[str]] = None
    ) -> ModelResult:
        """Generate text using a model"""
    
    async def check_model_status(self, model_name: str) -> bool:
        """Check if a model is loaded and ready"""
```

#### `ModelResult` (app/models/manager.py)
Result object from model generation.

```python
@dataclass
class ModelResult:
    """Result from model generation"""
    success: bool
    text: str = ""
    cost: float = 0.0
    execution_time: float = 0.0
    model_used: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
```

---

### 4. Cache Management

#### `CacheManager` (app/cache/redis_client.py)
Redis-based cache manager for the hot layer of metadata infrastructure.

```python
class CacheManager:
    """Redis-based cache manager for hot layer"""
    
    def __init__(self, redis_url: str, max_connections: int = 20):
        self.redis_url = redis_url
        self.max_connections = max_connections
        self.redis: Optional[redis.Redis] = None
        self.metrics = CacheMetrics()
        self._local_cache: Dict[str, tuple[Any, datetime]] = {}
    
    async def initialize(self):
        """Initialize Redis connection"""
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
```

**Specialized Methods:**
```python
# Routing and patterns
async def get_cached_route(self, query: str) -> Optional[List[str]]
async def cache_successful_route(self, query: str, route: List[str], cost: float)
async def get_user_pattern(self, user_id: str) -> Optional[Dict[str, Any]]
async def update_user_pattern(self, user_id: str, pattern_data: Dict[str, Any])

# Conversation management
async def get_conversation_history(self, session_id: str) -> Optional[List[Dict[str, Any]]]
async def update_conversation_history(self, session_id: str, history: List[Dict[str, Any]])

# Budget and rate limiting
async def get_remaining_budget(self, user_id: str) -> float
async def deduct_budget(self, user_id: str, cost: float) -> float
async def check_rate_limit(self, user_id: str, limit_per_minute: int = 60) -> tuple[bool, int]

# Performance optimization
async def cache_performance_hint(self, query_type: str, expected_time: float, expected_confidence: float)
async def cache_optimal_model(self, task_type: str, model_name: str, success_rate: float)
```

#### `CacheKey` (app/cache/redis_client.py)
Cache key management and generation utilities.

```python
class CacheKey:
    """Cache key constants and generators"""
    
    # Key prefixes
    ROUTE_PREFIX = "route:"
    PATTERN_PREFIX = "pattern:"
    CONVERSATION_PREFIX = "conv:"
    BUDGET_PREFIX = "budget:"
    RATE_PREFIX = "rate:"
    
    @staticmethod
    def query_hash(query: str) -> str:
        """Generate consistent hash for query"""
    
    @staticmethod
    def route_key(query: str) -> str:
        """Generate route cache key"""
    
    @staticmethod
    def conversation_key(session_id: str) -> str:
        """Generate conversation history key"""
```

#### `CacheMetrics` (app/cache/redis_client.py)
Performance metrics tracking for cache operations.

```python
class CacheMetrics(BaseModel):
    """Cache performance metrics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    avg_response_time: float = 0.0
    memory_usage: int = 0
    
    def update_hit(self, response_time: float):
        """Update metrics for cache hit"""
    
    def update_miss(self, response_time: float):
        """Update metrics for cache miss"""
```

---

### 5. API Layer

#### API Endpoints

**Chat Endpoints** (app/api/chat.py):
```python
@router.post("/stream")
async def chat_stream(request: ChatStreamRequest, ...) -> StreamingResponse:
    """Streaming chat endpoint compatible with OpenAI API format"""

@router.post("/complete") 
async def chat_complete(request: ChatRequest, ...) -> ChatResponse:
    """Non-streaming chat completion endpoint"""

@router.get("/history/{session_id}")
async def get_conversation_history(session_id: str, ...) -> Dict:
    """Get conversation history for a session"""

@router.delete("/history/{session_id}")
async def clear_conversation_history(session_id: str, ...) -> Dict:
    """Clear conversation history for a session"""
```

#### Request Schemas (app/schemas/requests.py)

**ChatRequest:**
```python
class ChatRequest(BaseModel):
    """Non-streaming chat request"""
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: Optional[str] = None
    context: Optional[Context] = None
    constraints: Optional[Constraints] = None
```

**ChatStreamRequest:**
```python
class ChatStreamRequest(BaseModel):
    """Streaming chat request (OpenAI-compatible)"""
    messages: List[ChatMessage] = Field(...)
    session_id: Optional[str] = None
    model: Optional[str] = Field("auto")
    max_tokens: Optional[int] = Field(300)
    temperature: Optional[float] = Field(0.7)
    stream: bool = Field(True)
```

**Constraints:**
```python
class Constraints(BaseModel):
    """Request constraints"""
    max_cost: Optional[float] = Field(0.05, description="Maximum cost in INR")
    max_time: Optional[float] = Field(5.0, description="Maximum execution time in seconds")
    quality_requirement: Optional[str] = Field("balanced", description="Quality level: minimal, balanced, high, premium")
    force_local_only: Optional[bool] = Field(False, description="Force local models only")
```

#### Response Schemas (app/schemas/responses.py)

**ChatResponse:**
```python
class ChatResponse(BaseResponse):
    """Chat completion response"""
    data: Dict[str, Any] = Field(..., description="Chat response data")
    
    # Inherits from BaseResponse:
    # - status: str
    # - metadata: ResponseMetadata
    # - cost_prediction: CostPrediction  
    # - developer_hints: DeveloperHints
```

**ResponseMetadata:**
```python
class ResponseMetadata(BaseModel):
    """Response metadata"""
    query_id: str
    execution_time: float
    cost: float
    models_used: List[str]
    confidence: float
    cached: bool
    timestamp: Optional[str]
```

**CostPrediction:**
```python
class CostPrediction(BaseModel):
    """Cost prediction and optimization"""
    estimated_cost: float
    cost_breakdown: List[CostBreakdown]
    savings_tips: List[str]
    alternative_workflows: Optional[List[Dict[str, Any]]]
```

---

### 6. Middleware & Security

#### `RateLimitMiddleware` (app/api/middleware.py)
Request rate limiting and performance tracking.

```python
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting and add performance headers"""
```

#### `CostTrackingMiddleware` (app/api/middleware.py)
Cost attribution and budget tracking.

```python
class CostTrackingMiddleware(BaseHTTPMiddleware):
    """Cost tracking middleware"""
    
    async def dispatch(self, request: Request, call_next):
        """Track costs per request"""
```

#### Authentication Functions (app/api/middleware.py)

```python
async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Dict[str, Any]:
    """Get current user from token or create anonymous user"""

async def check_rate_limit(user_id: str, cache_manager: CacheManager, endpoint: str = "general") -> bool:
    """Check if user is within rate limits"""

async def track_cost(user_id: str, cost: float, cache_manager: CacheManager) -> float:
    """Track and deduct cost from user budget"""

async def check_budget(user_id: str, estimated_cost: float, cache_manager: CacheManager) -> bool:
    """Check if user has sufficient budget"""
```

#### Permission Decorators:

```python
def require_permission(permission: str):
    """Decorator to require specific permission"""

def require_tier(min_tier: str):
    """Decorator to require minimum user tier"""
```

---

## 📡 API Reference

### Base URLs
- **Development**: `http://localhost:8000`
- **Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

### Authentication
```bash
# Bearer token (development)
Authorization: Bearer <your-token>

# Anonymous access (limited)
# No header required for basic testing
```

### Core Endpoints

#### Chat Completion
```http
POST /api/v1/chat/complete
Content-Type: application/json

{
  "message": "Explain async/await in Python",
  "session_id": "session_123",
  "constraints": {
    "max_cost": 0.10,
    "quality_requirement": "high"
  }
}
```

Response:
```json
{
  "status": "success",
  "data": {
    "response": "Async/await in Python allows you to write asynchronous code...",
    "session_id": "session_123"
  },
  "metadata": {
    "query_id": "query_456",
    "execution_time": 1.23,
    "cost": 0.008,
    "models_used": ["llama2:7b"],
    "confidence": 0.89,
    "cached": false
  },
  "cost_prediction": {
    "estimated_cost": 0.008,
    "cost_breakdown": [
      {"step": "classification", "model": "phi:mini", "cost": 0.0},
      {"step": "generation", "model": "llama2:7b", "cost": 0.0}
    ],
    "savings_tips": ["Use phi:mini for simpler questions"]
  }
}
```

#### Streaming Chat
```http
POST /api/v1/chat/stream
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "What is machine learning?"}
  ],
  "stream": true
}
```

Response (Server-Sent Events):
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1677652288,"model":"llama2:7b","choices":[{"index":0,"delta":{"content":"Machine"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1677652288,"model":"llama2:7b","choices":[{"index":0,"delta":{"content":" learning"},"finish_reason":null}]}

data: [DONE]
```

#### System Health
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "components": {
    "cache": "healthy",
    "models": "healthy"
  },
  "version": "1.0.0"
}
```

#### Metrics
```http
GET /metrics
```

Response:
```json
{
  "status": "success",
  "metrics": {
    "cache": {
      "hit_rate": 0.85,
      "total_requests": 1000
    },
    "models": {
      "total_requests": 500,
      "local_percentage": 87.5
    }
  }
}
```

### Error Responses

```json
{
  "status": "error",
  "message": "Rate limit exceeded",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "query_id": "query_789",
  "timestamp": "2025-06-19T10:30:00Z"
}
```

Common Error Codes:
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `BUDGET_EXCEEDED` - Insufficient budget
- `MODEL_UNAVAILABLE` - Model not ready
- `INVALID_INPUT` - Request validation failed
- `TIMEOUT` - Request took too long

---

## 🔧 Development Workflow

### Project Structure
```
ai-search-system/
├── app/                           # Main application code
│   ├── __init__.py
│   ├── main.py                   # FastAPI application entry point
│   ├── api/                      # API layer
│   │   ├── __init__.py
│   │   ├── chat.py              # Chat endpoints
│   │   ├── search.py            # Search endpoints (future)
│   │   └── middleware.py        # Authentication, rate limiting
│   ├── core/                     # Core configuration and utilities
│   │   ├── __init__.py
│   │   ├── config.py            # Settings management
│   │   ├── logging.py           # Logging configuration
│   │   └── security.py          # Security utilities
│   ├── graphs/                   # LangGraph implementations
│   │   ├── __init__.py
│   │   ├── base.py              # Base classes for graphs and nodes
│   │   ├── chat_graph.py        # Chat conversation graph
│   │   ├── search_graph.py      # Search and analysis graph (future)
│   │   └── orchestrator.py      # Master orchestration (future)
│   ├── models/                   # Model management
│   │   ├── __init__.py
│   │   ├── manager.py           # Model registry and lifecycle
│   │   ├── ollama_client.py     # Ollama API client
│   │   └── selectors.py         # Model selection logic
│   ├── cache/                    # Caching layer
│   │   ├── __init__.py
│   │   ├── redis_client.py      # Redis cache implementation
│   │   └── strategies.py        # Caching strategies
│   └── schemas/                  # Request/response models
│       ├── __init__.py
│       ├── requests.py          # API request schemas
│       └── responses.py         # API response schemas
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py             # Test configuration
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── test_*.py              # Individual test files
├── docker/                      # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── prometheus.yml
├── scripts/                     # Development scripts
│   ├── setup.sh               # Environment setup
│   └── dev.sh                 # Development helpers
├── docs/                       # Documentation
│   └── api/                   # API documentation
├── requirements.txt           # Python dependencies
├── requirements-dev.txt       # Development dependencies
├── .env                      # Environment variables
├── .gitignore               # Git ignore rules
└── README.md               # Project README
```

### Development Commands

#### Environment Setup
```bash
# Initial setup
chmod +x scripts/setup.sh scripts/dev.sh
./scripts/setup.sh

# Start development environment
./scripts/dev.sh start

# Stop environment
./scripts/dev.sh stop

# View logs
./scripts/dev.sh logs

# Run tests
./scripts/dev.sh test

# Code linting
./scripts/dev.sh lint

# Pull additional models
./scripts/dev.sh models

# Clean up
./scripts/dev.sh clean
```

#### Local Development
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt

# Run server with hot reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest

# Run with coverage
pytest --cov=app tests/

# Format code
black app/ tests/
isort app/ tests/

# Type checking
mypy app/

# Linting
flake8 app/ tests/
```

### Adding New Features

#### 1. Adding a New Graph Node

```python
# app/graphs/my_new_node.py
from app.graphs.base import BaseGraphNode, GraphState, NodeResult

class MyNewNode(BaseGraphNode):
    def __init__(self, dependency_service):
        super().__init__("my_new_node", "processing")
        self.dependency_service = dependency_service
    
    async def execute(self, state: GraphState, **kwargs) -> NodeResult:
        try:
            # Your node logic here
            result_data = await self.dependency_service.process(state.original_query)
            
            return NodeResult(
                success=True,
                confidence=0.8,
                data={"processed_data": result_data},
                cost=0.001,
                model_used="my_model"
            )
        except Exception as e:
            return NodeResult(
                success=False,
                error=f"Processing failed: {str(e)}",
                confidence=0.0
            )
```

#### 2. Adding a New Graph

```python
# app/graphs/my_new_graph.py
from app.graphs.base import BaseGraph, GraphType
from app.graphs.my_new_node import MyNewNode

class MyNewGraph(BaseGraph):
    def __init__(self, dependency_service):
        super().__init__(GraphType.CUSTOM, "my_new_graph")
        self.dependency_service = dependency_service
    
    def define_nodes(self) -> Dict
