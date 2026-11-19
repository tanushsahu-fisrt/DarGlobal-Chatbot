"""Check that your OpenRouter key and models work:  python scripts/ping_llm.py"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv  # noqa: E402

load_dotenv()
from app import llm  # noqa: E402


async def main():
    print("models:", llm.models())
    text, model = await llm.chat([{"role": "user", "content": "Reply with the single word: OK"}], max_tokens=300)
    print(f"answered by {model}: {text}")

asyncio.run(main())
