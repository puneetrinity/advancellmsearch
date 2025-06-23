# README.md
"""
AI Search System README
"""

# AI Search System 🚀

Revolutionary AI search system where **intelligence lives in APIs, not interfaces**. Built with LangGraph orchestration, local-first processing via Ollama, and dual-layer metadata infrastructure.

## 🎯 Core Philosophy

- **LLMs are workers, not rulers** - Treat models as interchangeable graph nodes
- **LangGraph is the conductor** - Orchestrates intelligent workflows  
- **APIs are the intelligence layer** - Chat is just one interface consuming smart APIs
- **85% local inference** - Cost-efficient processing via Ollama with API fallbacks
- **Metadata-driven learning** - Continuous optimization through pattern recognition

## ✨ Features

### Phase 1 (Current)
- 🔥 **Chat API** - Streaming and non-streaming conversational AI
- 🧠 **Intelligent Routing** - Context-aware model selection
- ⚡ **Hot Caching** - Redis-based speed optimization
- 💰 **Cost Tracking** - Transparent cost attribution and budgets
- 🛡️ **Rate Limiting** - Tiered access control
- 📊 **Real-time Metrics** - Performance monitoring

### Coming Soon
- 🔍 **Advanced Search** - Multi-source research capabilities
- 📈 **Analytics Dashboard** - Usage insights and optimization
- 🏢 **Enterprise Features** - SSO, tenant isolation, compliance
- 🌐 **Multi-agent Workflows** - Complex task orchestration

## 🏗️ Architecture

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

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)
- 8GB+ RAM (for local LLM inference)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd ai-search-system
chmod +x scripts/setup.sh scripts/dev.sh
./scripts/setup.sh
```

### 2. Start Development Environment
```bash
./scripts/dev.sh start
```

### 3. Access Services
- **API Documentation**: http://localhost:8000/docs
- **Chat Interface**: http://localhost:8000
- **Redis Commander**: http://localhost:8081
- **Health Check**: http://localhost:8000/health

## 📚 API Usage

### Chat Completion
```bash
curl -X POST "http://localhost:8000/api/v1/chat/complete" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain async/await in Python",
    "session_id": "session_123",
    "constraints": {
      "quality_requirement": "high",
      "max_cost": 0.10
    }
  }'
```

### Streaming Chat
```bash
curl -X POST "http://localhost:8000/api/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "stream": true
  }'
```

## 🛠️ Development

### Project Structure
```
ai-search-system/
├── app/
│   ├── api/           # API endpoints
│   ├── core/          # Configuration & logging
│   ├── graphs/        # LangGraph implementations
│   ├── models/        # Model management
│   ├── cache/         # Redis caching
│   └── schemas/       # Request/response models
├── docker/            # Docker configuration
├── scripts/           # Development scripts
└── tests/             # Test suite
```

### Available Models
- **phi:mini** - Ultra-fast classification (T0)
- **llama2:7b** - Balanced performance (T1)  
- **mistral:7b** - Analytical reasoning (T2)
- **llama2:13b** - Complex understanding (T2)
- **codellama** - Programming assistance (T2)

### Development Commands
```bash
# Start development environment
./scripts/dev.sh start

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

## 📈 Performance Targets

### Technical KPIs
- **Response Time**: < 2.5s (P95)
- **Local Processing**: > 85%
- **Cache Hit Rate**: > 80%
- **Cost per Query**: < ₹0.02

### Cost Efficiency
- **Free Tier**: 1,000 queries/month, ₹20 budget
- **Pro Tier**: 10,000 queries/month, ₹500 budget
- **Enterprise**: Unlimited, custom pricing

## 🔧 Configuration

Key environment variables:
```bash
# Core
ENVIRONMENT=development
DEBUG=true

# Services  
REDIS_URL=redis://localhost:6379
OLLAMA_HOST=http://localhost:11434

# Limits
DEFAULT_MONTHLY_BUDGET=100.0
RATE_LIMIT_PER_MINUTE=60

# Models
DEFAULT_MODEL=phi:mini
FALLBACK_MODEL=llama2:7b
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_chat_api.py

# Run integration tests
pytest tests/integration/
```

## 📊 Monitoring

### Health Checks
- `/health` - System health status
- `/metrics` - Prometheus metrics
- Redis, Ollama connectivity

### Metrics Collected
- Request latency & throughput
- Model usage & performance
- Cache hit rates
- Cost attribution
- Error rates

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`./scripts/dev.sh test`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: `/docs` directory

---

**Built with ❤️ for intelligent, cost-effective AI search**

## License

This project is licensed under the Server Side Public License (SSPL) v1.  
See the LICENSE file for details.
