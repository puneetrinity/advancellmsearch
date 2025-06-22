# Environment & Configuration

## Required Environment Variables
- `OLLAMA_HOST` (e.g., http://localhost:11434)
- `BRAVE_API_KEY` (for search)
- `OPENAI_API_KEY` (if using OpenAI)
- `ANTHROPIC_API_KEY` (if using Anthropic)

## Model Assignment Example
```
MODEL_ASSIGNMENTS = {
    "simple_classification": "tinyllama:latest",
    "qa_and_summary": "llama2:7b",
    "analytical_reasoning": "mistral:7b",
    "deep_research": "llama2:13b",
    "code_tasks": "codellama",
    "multilingual": "aya:8b"
}
```

## Model Loading Tiers
- T0: Always loaded (e.g., tinyllama)
- T1: Kept warm (e.g., llama2:7b)
- T2: Load on demand (e.g., larger models)

## Quality-Based Selection
- Minimal quality: small/fast models
- Premium quality: larger models

## Setup Checklist
- [ ] Pull required models with Ollama
- [ ] Set all required environment variables
- [ ] Start Redis and other dependencies
- [ ] Run integration tests

---

See `PROJECT_STATUS.md` and `README.md` for more.
