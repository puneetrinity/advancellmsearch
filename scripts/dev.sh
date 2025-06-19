# scripts/dev.sh
#!/bin/bash
"""
Development helper script
"""

set -e

function start_dev() {
    echo "🚀 Starting development environment..."
    docker-compose -f docker/docker-compose.yml up -d
    echo "✅ Development environment started"
    echo "📖 API docs available at: http://localhost:8000/docs"
}

function stop_dev() {
    echo "🛑 Stopping development environment..."
    docker-compose -f docker/docker-compose.yml down
    echo "✅ Development environment stopped"
}

function restart_dev() {
    echo "🔄 Restarting development environment..."
    stop_dev
    start_dev
}

function logs() {
    docker-compose -f docker/docker-compose.yml logs -f ai-search-api
}

function test() {
    echo "🧪 Running tests..."
    docker-compose -f docker/docker-compose.yml exec ai-search-api pytest
}

function lint() {
    echo "🔍 Running linting..."
    black app/ tests/
    flake8 app/ tests/
    mypy app/
}

function pull_models() {
    echo "📥 Pulling additional Ollama models..."
    docker-compose -f docker/docker-compose.yml exec ollama ollama pull mistral:7b
    docker-compose -f docker/docker-compose.yml exec ollama ollama pull llama2:13b
    docker-compose -f docker/docker-compose.yml exec ollama ollama pull codellama
    echo "✅ Models pulled"
}

function clean() {
    echo "🧹 Cleaning up..."
    docker-compose -f docker/docker-compose.yml down -v
    docker system prune -f
    echo "✅ Cleanup complete"
}

function help() {
    echo """
AI Search System - Development Helper

Usage: ./scripts/dev.sh [command]

Commands:
  start     Start development environment
  stop      Stop development environment  
  restart   Restart development environment
  logs      Show API logs
  test      Run tests
  lint      Run code linting
  models    Pull additional Ollama models
  clean     Clean up Docker resources
  help      Show this help message

Examples:
  ./scripts/dev.sh start
  ./scripts/dev.sh logs
  ./scripts/dev.sh test
"""
}

# Main command dispatcher
case "$1" in
    start)
        start_dev
        ;;
    stop)
        stop_dev
        ;;
    restart)
        restart_dev
        ;;
    logs)
        logs
        ;;
    test)
        test
        ;;
    lint)
        lint
        ;;
    models)
        pull_models
        ;;
    clean)
        clean
        ;;
    help|*)
        help
        ;;
esac
