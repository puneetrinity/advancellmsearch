### ⭐ **Phase 5 - Next-Gen Intelligence (Future)**
- 🎯 **Advanced Intelligence** - Multi-modal support and custom model fine-tuning
- 🧠 **Neural Optimization** - Self-improving routing algorithms
- 🌍 **Global Scale** - Multi-region deployment with edge computing
- 🔮 **Predictive Analytics** - Anticipatory search and response generation### 🚀 **Phase 4 - Enterprise & Scale (In Progress)**
- 🔐 **Enterprise Security** - JWT authentication, SSO, and audit logging
- 🏗️ **Kubernetes Deployment** - Auto-scaling, load balancing, service mesh
- 📋 **Compliance** - SOC2, GDPR, HIPAA ready with audit trails
- 🌐 **Multi-agent Workflows** - Complex task orchestration and parallel processing# AI Search System 🚀

## Neural Search Architecture - API-First Intelligence Platform

Revolutionary AI search system where **intelligence lives in APIs, not interfaces**. Built with LangGraph orchestration, local-first processing via Ollama, and dual-layer metadata infrastructure for cost-efficient, intelligent search capabilities.

## 🎯 Core Philosophy

- **LLMs are workers, not rulers** - Treat models as interchangeable graph nodes
- **LangGraph is the conductor** - Orchestrates intelligent workflows through smart routing
- **APIs are the intelligence layer** - Chat is just one interface consuming smart APIs
- **85% local inference** - Cost-efficient processing via Ollama with API fallbacks
- **Metadata-driven learning** - Continuous optimization through pattern recognition

## 🏗️ Neural Search Architecture

The system implements a sophisticated LangGraph Router Architecture with intelligent orchestration:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface                           │
│           Multi-Channel Access & Interactions               │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  API Gateway                                │
│            /api/v1/* endpoints                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│               LangGraph Router                              │
│              Smart Orchestration                            │
└─────────┬─────────┬─────────┬───────────────────────────────┘
          │         │         │
┌─────────▼───┐ ┌──▼──────┐ ┌──▼──────────┐
│ Chat Graph  │ │ Search  │ │ Research    │
│ Local + Cloud│ │ Graph   │ │ Graph       │
│ LLMs        │ │ Web +   │ │ Multi-Source│
└─────────────┘ │Analysis │ │ Synthesis   │
                └─────────┘ └─────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│               Model Selection                               │
│            Cost Optimized Routing                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Response Assembly                              │
│            Structured Output + Metadata                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│               Smart Cache + Analytics                       │
│        Redis + ClickHouse Performance Layer                │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 🧠 **LangGraph Router** - Smart Orchestration
- **Intelligent routing** based on query patterns and user context
- **Dynamic workflow composition** for complex multi-step tasks
- **Cost-aware optimization** with real-time budget management
- **Pattern learning** from successful execution paths

#### 🔍 **Multi-Graph System**
- **Chat Graph**: Conversational AI with context management
- **Search Graph**: Web search + content analysis + citation
- **Research Graph**: Multi-source research with synthesis
- **Custom Graphs**: Extensible for domain-specific workflows

#### 💡 **Model Selection Engine**
- **Local-first processing**: Ollama models (phi:mini, llama2, mistral)
- **Smart fallbacks**: OpenAI/Claude for complex reasoning
- **Performance tracking**: Automatic model optimization
- **Cost prediction**: Transparent cost attribution

## ✨ Features

### 🚀 **Phase 1 - Core Intelligence (Current)**
- ✅ **Chat API** - Streaming and non-streaming conversational AI
- ✅ **Intelligent Routing** - Context-aware model selection with LangGraph
- ✅ **Multi-Provider Search** - Brave, DuckDuckGo, Google Custom Search
- ✅ **Smart Caching** - Redis-based speed optimization with pattern learning
- ✅ **Cost Tracking** - Transparent cost attribution and budget management
- ✅ **Rate Limiting** - Tiered access control with usage analytics
- ✅ **Performance Monitoring** - Real-time metrics and optimization

### ✅ **Phase 2 - Advanced Features (COMPLETED)**
- ✅ **Advanced Search** - Multi-provider search (Brave, DuckDuckGo, Google) with content analysis
- ✅ **Analytics Dashboard** - Performance monitoring and usage insights
- ✅ **Pattern Recognition** - ML-based routing with TF-IDF and cosine similarity
- ✅ **Auto-Learning** - Continuous improvement from successful execution patterns
- ✅ **Cost Optimization** - Intelligent model selection with performance tracking
- ✅ **Advanced Caching** - Multi-strategy caching (LRU, LFU, TTL, Predictive)

### ✅ **Phase 3 - Production Ready (COMPLETED)**
- ✅ **Real Search Integration** - Production Brave Search and ScrapingBee APIs implemented
- ✅ **Enhanced Analytics** - ClickHouse cold storage with advanced pattern discovery
- ✅ **Smart Provider Routing** - Cost-aware search strategy (Brave → DuckDuckGo fallback)
- ✅ **Content Enhancement** - ScrapingBee integration for premium content extraction
- ✅ **Advanced Caching** - Multi-strategy caching with predictive capabilities

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)
- 8GB+ RAM (for local LLM inference)
- Redis (for caching)
- Ollama (for local models)

### 1. Clone and Setup
```bash
git clone <your-repo>
cd ai-search-system

# Make scripts executable
chmod +x scripts/setup.sh scripts/dev.sh

# Initial setup (installs dependencies, pulls models)
./scripts/setup.sh
```

### 2. Start Development Environment
```bash
# Start all services (Redis, Ollama, API server)
./scripts/dev.sh start

# View real-time logs
./scripts/dev.sh logs

# Check system health
curl http://localhost:8000/health
```

### 3. Test the APIs
```bash
# Chat API
curl -X POST "http://localhost:8000/api/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain quantum computing in simple terms",
    "session_id": "test-session"
  }'

# Search API
curl -X POST "http://localhost:8000/api/v1/search/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest developments in renewable energy",
    "search_depth": "deep",
    "sources": ["web", "news"]
  }'
```

## 📡 API Reference

### Core Intelligence APIs

#### 💬 Chat API
```http
POST /api/v1/chat/stream
Content-Type: application/json

{
  "message": "Your question or request",
  "session_id": "unique-session-id",
  "context": {
    "user_preferences": {},
    "conversation_history": []
  },
  "constraints": {
    "max_cost": 0.05,
    "max_time": 5.0,
    "quality_requirement": "high"
  }
}
```

#### 🔍 Search API
```http
POST /api/v1/search/analyze
Content-Type: application/json

{
  "query": "search query",
  "search_depth": "shallow|deep|comprehensive",
  "sources": ["web", "academic", "news"],
  "analysis_type": "summary|detailed|comparative"
}
```

#### 🧪 Research API
```http
POST /api/v1/research/deep-dive
Content-Type: application/json

{
  "research_question": "complex research question",
  "methodology": "systematic|exploratory|comparative",
  "time_budget": 300,
  "cost_budget": 0.50
}
```

### Utility APIs
- `GET /health` - System health and model status
- `GET /api/v1/models/available` - Available models and capabilities
- `GET /api/v1/analytics/usage-summary` - Usage analytics
- `GET /api/v1/routing/explain/{query_id}` - Routing decision explanation

## 🛠️ Development

### Project Structure
```
ai-search-system/
├── app/
│   ├── api/                    # API endpoints and routing
│   ├── graphs/                 # LangGraph implementations
│   │   ├── chat_graph.py      # Conversational workflows
│   │   ├── search_graph.py    # Search and analysis
│   │   ├── research_graph.py  # Multi-source research
│   │   └── intelligent_router.py # Smart routing logic
│   ├── models/                # Model management and selection
│   ├── cache/                 # Redis caching layer
│   ├── optimization/          # Cost and performance optimization
│   ├── schemas/              # Request/response models
│   └── core/                 # Configuration and utilities
├── docker/                   # Docker configuration
├── scripts/                  # Development and deployment scripts
├── tests/                    # Comprehensive test suite
└── docs/                     # Documentation and guides
```

### Available Models
- **phi:mini** - Ultra-fast classification and simple queries (T0)
- **llama2:7b** - Balanced performance for general tasks (T1)
- **mistral:7b** - Analytical reasoning and complex queries (T2)
- **llama2:13b** - Deep understanding and research (T2)
- **codellama** - Programming and technical assistance (T2)
- **OpenAI/Claude** - Premium models for critical tasks (T3)

### Development Commands
```bash
# Development environment
./scripts/dev.sh start          # Start all services
./scripts/dev.sh stop           # Stop services
./scripts/dev.sh logs           # View logs
./scripts/dev.sh restart        # Restart services

# Testing and quality
./scripts/dev.sh test           # Run full test suite
./scripts/dev.sh test-unit      # Unit tests only
./scripts/dev.sh test-integration # Integration tests
./scripts/dev.sh lint           # Code linting and formatting
./scripts/dev.sh coverage       # Test coverage report

# Model management
./scripts/dev.sh models         # Pull additional models
./scripts/dev.sh models-status  # Check model health
./scripts/dev.sh models-cleanup # Clean unused models

# Utilities
./scripts/dev.sh clean          # Clean up containers and data
./scripts/dev.sh reset          # Complete reset and rebuild
```

## 📊 Performance & Metrics

### Technical KPIs
- **Response Time**: < 2.5s (P95) for chat, < 5s for search
- **Local Processing**: > 85% of requests handled locally
- **Cache Hit Rate**: > 80% for repeated queries
- **Cost per Query**: < ₹0.02 average
- **Accuracy**: > 90% for routing decisions

### Cost Efficiency Tiers
```yaml
Free Tier:     1,000 queries/month, ₹20 budget
Pro Tier:      10,000 queries/month, ₹500 budget  
Enterprise:    Unlimited, custom pricing
```

### Monitoring & Analytics
- **Health Checks**: `/health` endpoint with detailed system status
- **Prometheus Metrics**: Performance, cost, and usage metrics
- **Real-time Dashboards**: Grafana dashboards for monitoring
- **Cost Attribution**: Detailed breakdown by user, model, and query type
- **Performance Insights**: Automatic optimization recommendations

## 🔧 Configuration

### Environment Variables
```bash
# Core Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Services  
REDIS_URL=redis://localhost:6379
OLLAMA_HOST=http://localhost:11434
CLICKHOUSE_URL=http://localhost:8123

# API Keys (optional for premium models)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
BRAVE_SEARCH_API_KEY=your_brave_key

# Cost and Rate Limiting
DEFAULT_MONTHLY_BUDGET=100.0
RATE_LIMIT_PER_MINUTE=60
MAX_CONCURRENT_REQUESTS=10

# Model Configuration
DEFAULT_MODEL=phi:mini
FALLBACK_MODEL=llama2:7b
ENABLE_PREMIUM_FALLBACK=true
```

### Model Configuration
```yaml
# Model tier configuration
models:
  tier_0:  # Ultra-fast, low cost
    - phi:mini
    - tinyllama
  tier_1:  # Balanced performance
    - llama2:7b
    - mistral:7b
  tier_2:  # High quality, local
    - llama2:13b
    - codellama:13b
  tier_3:  # Premium API models
    - gpt-4
    - claude-3-sonnet
```

## 🧪 Testing

### Running Tests
```bash
# Full test suite
pytest

# With coverage report
pytest --cov=app --cov-report=html tests/

# Specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only
pytest -m e2e                    # End-to-end tests
pytest tests/test_chat_api.py     # Specific test file

# Performance testing
pytest tests/performance/         # Load and stress tests
```

### Test Structure
```
tests/
├── unit/               # Fast unit tests
├── integration/        # Service integration tests
├── e2e/               # End-to-end workflow tests
├── performance/       # Load and stress tests
└── fixtures/          # Test data and mocks
```

## 📈 Monitoring & Observability

### Health Monitoring
```bash
# System health
curl http://localhost:8000/health

# Detailed metrics
curl http://localhost:8000/metrics

# Service status
curl http://localhost:8000/api/v1/status/detailed
```

### Key Metrics Tracked
- **Request Latency**: P50, P95, P99 response times
- **Throughput**: Requests per second and concurrent users
- **Model Performance**: Success rates and execution times
- **Cost Attribution**: Per-user, per-model cost tracking
- **Cache Performance**: Hit rates and efficiency metrics
- **Error Rates**: Error types and frequency analysis

### Dashboards
- **Operations Dashboard**: Real-time system health and performance
- **Cost Dashboard**: Budget tracking and optimization insights
- **User Analytics**: Usage patterns and feature adoption
- **Model Performance**: Model efficiency and optimization opportunities

## 🤝 Contributing

### Development Workflow
1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Run tests** (`./scripts/dev.sh test`)
4. **Code quality** (`./scripts/dev.sh lint`)
5. **Commit** changes (`git commit -m 'Add amazing feature'`)
6. **Push** to branch (`git push origin feature/amazing-feature`)
7. **Open** Pull Request with detailed description

### Code Standards
- **Python 3.11+** with type hints
- **async/await** for all I/O operations
- **Comprehensive tests** (90%+ coverage)
- **Clear documentation** for public APIs
- **Performance conscious** code with profiling

### Adding New Features

#### New Graph Node
```python
# app/graphs/nodes/my_new_node.py
from app.graphs.base import BaseGraphNode, GraphState, NodeResult

class MyNewNode(BaseGraphNode):
    async def execute(self, state: GraphState) -> NodeResult:
        # Your implementation
        return NodeResult(success=True, data=result)
```

#### New API Endpoint
```python
# app/api/v1/my_endpoint.py
from fastapi import APIRouter
from app.schemas.requests import MyRequest
from app.schemas.responses import MyResponse

router = APIRouter()

@router.post("/my-endpoint", response_model=MyResponse)
async def my_endpoint(request: MyRequest):
    # Your implementation
    return MyResponse(...)
```

## 📚 Documentation

### Additional Resources
- **[Developer Onboarding Guide](docs/Developer%20Onboarding%20Guide.md)** - Comprehensive setup and development guide
- **[Architecture Documentation](docs/architecture/)** - Detailed system architecture
- **[API Documentation](http://localhost:8000/docs)** - Interactive API documentation
- **[Deployment Guide](docs/deployment/)** - Production deployment instructions
- **[Performance Tuning](docs/performance/)** - Optimization best practices

### Support
- **Issues**: [GitHub Issues](../../issues) for bug reports and feature requests
- **Discussions**: [GitHub Discussions](../../discussions) for questions and ideas
- **Documentation**: Check `/docs` directory for comprehensive guides
- **Community**: Join our developer community for support and collaboration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🚀 Ready to Build Intelligence?

This AI Search System provides the foundation for building intelligent, cost-effective search applications. With its API-first architecture, local-first processing, and intelligent routing, you can create powerful AI experiences that scale efficiently and transparently.

**Start building today:**
```bash
./scripts/setup.sh && ./scripts/dev.sh start
---

*Neural Search Architecture - Where Intelligence Lives in APIs, Not Interfaces*

**Built with ❤️ for intelligent, cost-effective AI search**

## License

This project is licensed under the Server Side Public License (SSPL) v1.  
See the LICENSE file for details.
