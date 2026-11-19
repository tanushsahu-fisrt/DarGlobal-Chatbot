"""Tiny OpenRouter client (OpenAI-compatible chat completions API).

Free models rotate in and out of OpenRouter, so the model list comes from the
OPENROUTER_MODELS env var (comma-separated, best first) and is sent as a
fallback list: if the first model is down or rate-limited, OpenRouter tries the next.
"""
import asyncio
import os
import re
import sys

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = ("google/gemma-4-31b-it:free,"
                  "nvidia/nemotron-3-ultra-550b-a55b:free,"
                  "google/gemma-4-26b-a4b-it:free")


class LLMError(Exception):
    pass


def models():
    raw = os.getenv("OPENROUTER_MODELS", DEFAULT_MODELS)
    return [m.strip() for m in raw.split(",") if m.strip()]


async def chat(messages, max_tokens=900, temperature=0.2):
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise LLMError("OPENROUTER_API_KEY is not set")
    ms = models()
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    payload.update({"models": ms} if len(ms) > 1 else {"model": ms[0]})
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": os.getenv("APP_URL", "http://localhost:8000"),
               "X-Title": "DarGlobal & Wasalt Assistant"}
    last = "unknown error"
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        for attempt in range(2):
            try:
                r = await client.post(URL, headers=headers, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = f"network error: {type(e).__name__}"
                await asyncio.sleep(1.5)
                continue
            if r.status_code == 200:
                data = r.json()
                try:
                    text = data["choices"][0]["message"].get("content") or ""
                except (KeyError, IndexError):
                    text = ""
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
                if text:
                    return text, data.get("model", ms[0])
                last = "empty answer (a reasoning model may have used up max_tokens)"
                continue
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code in (400, 401, 402, 403):    # retrying will not help
                break
            await asyncio.sleep(2 * (attempt + 1))        # 429 / 5xx: wait and retry once
    print("[llm] failed:", last, file=sys.stderr)
    raise LLMError(last)
