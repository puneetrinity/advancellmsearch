# scripts/setup.sh
#!/bin/bash
"""
Development environment setup script
"""

set -e

echo "🚀 Setting up AI Search System development environment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose found"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# Environment Configuration
ENVIRONMENT=development
DEBUG=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_MAX_CONNECTIONS=20

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_TIMEOUT=60

# Model Configuration
DEFAULT_MODEL=phi:mini
FALLBACK_MODEL=llama2:7b

# Cost & Budget
DEFAULT_MONTHLY_BUDGET=100.0
COST_TRACKING_ENABLED=true

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Security (Change in production!)
JWT_SECRET_KEY=dev-secret-key-change-in-production
EOF
    echo "✅ .env file created"
else
    echo "✅ .env file already exists"
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p logs
mkdir -p data/redis
mkdir -p data/ollama
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p docs/api

echo "✅ Directories created"

# Start services
echo "🐳 Starting Docker services..."
docker-compose -f docker/docker-compose.yml up -d redis ollama

echo "⏳ Waiting for services to be ready..."
sleep 10

# Pull initial models
echo "📥 Pulling initial Ollama models..."
docker-compose -f docker/docker-compose.yml exec ollama ollama pull phi:mini
docker-compose -f docker/docker-compose.yml exec ollama ollama pull llama2:7b

echo "✅ Initial models pulled"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if command -v python3 &> /dev/null; then
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements-dev.txt
    echo "✅ Python dependencies installed"
else
    echo "⚠️  Python3 not found. Please install Python dependencies manually."
fi

echo """
🎉 Setup complete! 

Next steps:
1. Start the API server:
   docker-compose -f docker/docker-compose.yml up ai-search-api

2. Or run locally:
   source venv/bin/activate
   uvicorn app.main:app --reload

3. Access the services:
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Redis Commander: http://localhost:8081
   - Ollama: http://localhost:11434

4. Run tests:
   pytest

Happy coding! 🚀
"""

---
