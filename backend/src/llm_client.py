"""
llm_client.py — Unified LLM call with Groq-first, Gemini fallback.
"""

from __future__ import annotations

import re
import time
from typing import List, Optional


_RATE_LIMIT_SIGNALS = ("429", "rate", "quota", "limit", "exhausted", "too many")
_DAY_LIMIT_SIGNALS  = ("per day", "perday", "daily", "requests per day", "rpd")


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in _RATE_LIMIT_SIGNALS)


def _is_daily_limit(exc: Exception) -> bool:
    """True when the quota resets tomorrow — no point retrying today."""
    msg = str(exc).lower()
    return any(s in msg for s in _DAY_LIMIT_SIGNALS)


def _parse_retry_seconds(exc: Exception) -> int:
    """Extract retry_delay seconds from Gemini error body, default 60."""
    m = re.search(r"retry_delay\s*\{[^}]*seconds:\s*(\d+)", str(exc))
    return int(m.group(1)) + 2 if m else 60


def build_llm_clients(groq_api_key: str = "", gemini_api_key: str = ""):
    """
    Initialise and return (groq_client, gemini_client).
    Either may be None if the key is missing or SDK is unavailable.
    Uses the new google-genai SDK.
    """
    groq_client   = None
    gemini_client = None

    # Groq
    _groq_key = (groq_api_key or "").strip().strip('"').strip("'")
    if _groq_key:
        try:
            from groq import Groq as GroqClient
            groq_client = GroqClient(api_key=_groq_key)
            print("✅ Groq client initialised (llama-3.3-70b-versatile)")
        except Exception as e:
            print(f"❌ Groq init failed: {e}")

    # Gemini — new google-genai SDK
    _gemini_key = (gemini_api_key or "").strip().strip('"').strip("'")
    if _gemini_key:
        try:
            from google import genai as google_genai
            gemini_client = google_genai.Client(api_key=_gemini_key)
            print("✅ Gemini fallback initialised (gemini-3.5-flash)")
        except Exception as e:
            print(f"❌ Gemini init failed: {e}")

    if not groq_client and not gemini_client:
        print("⚠️  No LLM available — set GROQ_API_KEY or GEMINI_API_KEY in .env")

    return groq_client, gemini_client


def call_llm(
    groq_client,
    gemini_client,
    messages: List[dict],
    max_tokens: int = 1024,
    temperature: float = 0.2,
    max_retries: int = 2,
) -> Optional[str]:
    """
    Call LLM with Groq-first, Gemini fallback on rate limit.
    On a daily quota exhaustion, stops immediately (no retry).
    """
    # ── Groq ────────────────────────────────────────────────────────────────
    if groq_client is not None:
        for attempt in range(max_retries + 1):
            try:
                resp = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = resp.choices[0].message.content
                return text.strip() if text else None
            except Exception as e:
                if _is_rate_limit(e):
                    print(f"⚠️  Groq rate limit — falling back to Gemini")
                    break
                if attempt < max_retries:
                    time.sleep(1)
                else:
                    print(f"❌ Groq failed: {str(e)[:100]}")

    # ── Gemini fallback ──────────────────────────────────────────────────────
    if gemini_client is not None:
        prompt = "\n\n".join(
            f"[{m['role'].upper()}]\n{m['content']}" for m in messages
        )
        for attempt in range(max_retries + 1):
            try:
                from google.genai import types as genai_types
                resp = gemini_client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
                text = resp.text
                print("📡 Used Gemini fallback")
                return text.strip() if text else None
            except Exception as e:
                if _is_daily_limit(e):
                    print(f"❌ Gemini daily quota exhausted — no retry until tomorrow")
                    return None
                if _is_rate_limit(e):
                    wait = _parse_retry_seconds(e)
                    print(f"⚠️  Gemini rate limit, waiting {wait}s")
                    time.sleep(wait)
                elif attempt < max_retries:
                    print(f"🔄 Gemini retry {attempt + 1}: {str(e)[:80]}")
                    time.sleep(1)
                else:
                    print(f"❌ Gemini failed: {str(e)[:100]}")

    return None
