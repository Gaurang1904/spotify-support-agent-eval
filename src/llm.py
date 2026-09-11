"""Thin, provider-agnostic LLM client.

Any OpenAI-compatible endpoint works: local Ollama (default), OpenAI, or
Anthropic's OpenAI-compat layer. Configure with env vars (see .env.example).

Responses are cached in a sqlite file so re-running the eval harness is fast
and deterministic. Delete outputs/llm_cache.db to force fresh calls.
"""
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("LLM_API_KEY", "ollama")
MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL") or MODEL

CACHE_PATH = Path(__file__).resolve().parent.parent / "outputs" / "llm_cache.db"
_client = None
_conn = None


def _db():
    global _conn
    if _conn is None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(CACHE_PATH)
        _conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
    return _conn


def _api():
    global _client
    if _client is None:
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    return _client


def chat(messages, model=None, temperature=0.0, max_tokens=512, use_cache=True):
    """Return the assistant's text. Cached by (model, messages, temperature)."""
    model = model or MODEL
    key = hashlib.sha256(
        json.dumps([model, messages, temperature, max_tokens], sort_keys=True).encode()
    ).hexdigest()
    if use_cache:
        row = _db().execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone()
        if row:
            return row[0]

    last_err = None
    for attempt in range(3):
        try:
            resp = _api().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
            break
        except Exception as e:  # network blips / rate limits
            last_err = e
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"LLM call failed after 3 attempts: {last_err}")

    if use_cache:
        _db().execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, text))
        _db().commit()
    return text


def chat_json(messages, **kw):
    """chat() but parse the first JSON object in the reply. Returns {} on failure.

    Small local models ignore response_format, so we extract by brace matching
    rather than trusting the API to enforce JSON.
    """
    text = chat(messages, **kw)
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            try:
                return json.loads(text[start:i + 1])
            except json.JSONDecodeError:
                return {}
    return {}


if __name__ == "__main__":
    print(f"endpoint={BASE_URL} model={MODEL} judge={JUDGE_MODEL}")
    out = chat_json([{"role": "user", "content": 'Reply with only: {"ok": true}'}])
    assert out.get("ok") is True, f"smoke test failed, got: {out}"
    print("llm ok")
