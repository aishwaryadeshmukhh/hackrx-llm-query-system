"""
cache.py — Redis-backed cache for parsed chunks, Pinecone index state, and query results.

Uses Upstash Redis over HTTP (no persistent TCP connection needed).
Falls back to an in-memory dict if Redis is not configured, so local dev
works with zero setup.

Environment variables required for Redis:
    UPSTASH_REDIS_REST_URL   e.g. https://your-db.upstash.io
    UPSTASH_REDIS_REST_TOKEN e.g. AXxxxxxxxxxxxxxxxxxxxx==
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── TTLs ──────────────────────────────────────────────────────────────────────
CHUNK_TTL   = 60 * 60 * 24 * 7   # 7 days  — parsed chunks are stable
QUERY_TTL   = 60 * 60 * 24       # 24 hours — query results may change with model updates

# ── Redis client (lazy-initialised) ───────────────────────────────────────────
_redis = None

def _get_redis():
    global _redis
    if _redis is not None:
        return _redis

    url   = os.getenv("UPSTASH_REDIS_REST_URL")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN")

    if not url or not token:
        logger.info("Redis not configured — using in-memory fallback cache")
        return None

    try:
        from upstash_redis import Redis
        _redis = Redis(url=url, token=token)
        # Verify connection with a ping
        _redis.set("__ping__", "1", ex=10)
        print("✅ Upstash Redis connected and verified")
        logger.info("✅ Upstash Redis connected and verified")
        return _redis
    except Exception as e:
        logger.warning(f"Redis init failed ({e}) — falling back to in-memory cache")
        print(f"⚠️ Redis init failed ({e}) — using in-memory cache")
        return None


# ── In-memory fallback ────────────────────────────────────────────────────────
_memory: dict[str, Any] = {}


# ── Internal get/set wrappers ─────────────────────────────────────────────────
def _get(key: str) -> Optional[str]:
    r = _get_redis()
    if r:
        try:
            return r.get(key)
        except Exception as e:
            logger.warning(f"Redis GET failed ({e}), using memory")
    return _memory.get(key)


def _set(key: str, value: str, ttl: int):
    r = _get_redis()
    if r:
        try:
            r.set(key, value, ex=ttl)
            return
        except Exception as e:
            logger.warning(f"Redis SET failed ({e}), using memory")
    _memory[key] = value


def _delete(key: str):
    r = _get_redis()
    if r:
        try:
            r.delete(key)
        except Exception as e:
            logger.warning(f"Redis DELETE failed ({e})")
    _memory.pop(key, None)


# ── Public API ────────────────────────────────────────────────────────────────

def get_chunks(file_hash: str) -> Optional[dict]:
    """
    Returns {"chunks": [...], "indexed_in_pinecone": bool} or None.
    """
    raw = _get(f"chunks:{file_hash}")
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def set_chunks(file_hash: str, chunks: list, indexed: bool = False):
    """Store parsed chunks for a PDF (keyed by MD5 hash of file content)."""
    payload = json.dumps({"chunks": chunks, "indexed_in_pinecone": indexed})
    _set(f"chunks:{file_hash}", payload, CHUNK_TTL)
    backend = "Redis" if _get_redis() else "memory"
    print(f"💾 Cached {len(chunks)} chunks for {file_hash[:8]} → {backend} (indexed={indexed})")


def mark_indexed(file_hash: str):
    """Flip indexed_in_pinecone = True without re-serialising chunks."""
    cached = get_chunks(file_hash)
    if cached is None:
        return
    cached["indexed_in_pinecone"] = True
    _set(f"chunks:{file_hash}", json.dumps(cached), CHUNK_TTL)


def get_query_result(file_hash: str, question: str) -> Optional[dict]:
    """
    Returns a cached answer dict or None.
    Cache key = file_hash + normalised question text.
    """
    key = f"qr:{file_hash}:{_normalise(question)}"
    raw = _get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def set_query_result(file_hash: str, question: str, result: dict):
    """Cache the full answer dict for a (file, question) pair."""
    key = f"qr:{file_hash}:{_normalise(question)}"
    _set(key, json.dumps(result), QUERY_TTL)


def delete_query_result(file_hash: str, question: str):
    """Remove a single cached query result (e.g. after a bad answer is detected)."""
    key = f"qr:{file_hash}:{_normalise(question)}"
    _delete(key)
    print(f"🗑️ Deleted query cache: {key[:40]}…")


def flush_all_query_results():
    """
    Delete ALL query result cache entries from Redis (keys matching qr:*).
    Falls back to clearing the in-memory dict entries with qr: prefix.
    """
    r = _get_redis()
    if r:
        try:
            keys = r.keys("qr:*")
            if keys:
                r.delete(*keys)
                print(f"🗑️ Flushed {len(keys)} query cache entries from Redis")
            else:
                print("🗑️ No query cache entries found in Redis")
        except Exception as e:
            print(f"⚠️ Redis flush failed ({e})")
    # Also clear in-memory fallback
    stale = [k for k in _memory if k.startswith("qr:")]
    for k in stale:
        del _memory[k]
    if stale:
        print(f"🗑️ Flushed {len(stale)} query cache entries from memory")


def _normalise(text: str) -> str:
    """Lowercase + collapse whitespace so minor phrasing differences still hit cache."""
    import re
    return re.sub(r"\s+", " ", text.lower().strip())
