# Developer Onboarding

## Prerequisites
- Python 3.10+
- Ollama (for LLM backend)
- Redis (for cache)
- Docker (optional, for full stack)

## Setup Steps
1. Clone the repository from GitHub.
2. Copy `.env.example` to `.env` and fill in required values.
3. Pull required models with Ollama (e.g., `ollama pull tinyllama`).
4. Start Redis and Ollama.
5. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
6. Run the backend:
   ```
   python -m app.main
   ```
7. Run integration tests:
   ```
   pytest tests/integration
   ```

## Useful Commands
- `ollama list` — List available models
- `ollama run tinyllama` — Test model inference
- `docker-compose up` — Start all services (if using Docker)

## Troubleshooting
- Check logs in `logs/` and `diagnostic_log.txt` for errors.
- Ensure all environment variables are set.
- If you see "No models available", check Ollama and model assignments.

---

See `CONFIGURATION.md` and `PROJECT_STATUS.md` for more details.
