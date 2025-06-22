#!/bin/sh
# wait-for-ollama.sh
# Wait until the Ollama service is healthy before starting the API

set -ex

host="ollama"
port=11434
url="http://$host:$port/api/tags"

echo "Checking Ollama health at $url..."

# Wait for Ollama to be healthy
until curl -sf "$url" > /dev/null; do
  echo "Waiting for Ollama at $url..."
  sleep 3
done

echo "Ollama is up! Starting API..."

# Execute the command passed as arguments (the API startup)
exec "$@"
