import asyncio
import os

from dotenv import load_dotenv

from app.models.ollama_client import OllamaClient

load_dotenv()


async def main():
    client = OllamaClient(base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    await client.initialize()
    models = await client.list_models(force_refresh=True)
    print("Ollama models:", [m.get("name") for m in models])


if __name__ == "__main__":
    asyncio.run(main())
