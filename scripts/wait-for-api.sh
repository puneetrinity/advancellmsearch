#!/bin/sh
# wait-for-api.sh
# Wait until the API is healthy before running tests
set -e

API_URL="${1:-http://localhost:8000/health}"
MAX_ATTEMPTS=30
SLEEP_TIME=2

for i in $(seq 1 $MAX_ATTEMPTS); do
  echo "Checking API health at $API_URL (attempt $i/$MAX_ATTEMPTS)"
  if curl -fs $API_URL; then
    echo "API is healthy!"
    exit 0
  fi
  sleep $SLEEP_TIME
done

echo "API did not become healthy in time."
exit 1
