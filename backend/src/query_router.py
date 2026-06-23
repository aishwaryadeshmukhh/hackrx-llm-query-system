"""
query_router.py — Classify a query as simple or complex before processing.

Simple  → single retrieval + single LLM call (fast, ~2s)
Complex → ReAct agent loop with tool calls (thorough, ~15s)

A query is complex when answering it correctly requires:
- Checking waiting periods alongside coverage
- Reasoning about policy duration / inception date
- Pre-existing conditions interacting with coverage
- Multiple conditions or benefits in one question
- Exclusion clauses that could override a coverage answer
"""

from __future__ import annotations

import re
from typing import Literal

QueryType = Literal["simple", "complex"]

# ── Keyword-based fast path (no LLM needed) ──────────────────────────────────

_COMPLEX_PATTERNS = [
    r"\bwaiting period\b",
    r"\bpre.?existing\b",
    r"\bmonths? after\b",
    r"\bdays? after\b",
    r"\byears? after\b",
    r"\binception\b",
    r"\bpolicy age\b",
    r"\bpolicy duration\b",
    r"\bnew policy\b",
    r"\brecent policy\b",
    r"\bjust (bought|started|took)\b",
    r"\band .{3,40} (covered|excluded|waiting)\b",
    r"\bif .{3,60} (covered|eligible|claim)\b",
    r"\bboth\b.{3,40}\band\b",
    r"\b(first|second|third) year\b",
    r"\b\d+ months? (old|ago|since)\b",
    r"\b\d+ years? (old policy|since inception)\b",
    # specific medical conditions — always need exclusion check
    r"\b(mental illness|psychiatric|schizophrenia|depression|anxiety disorder)\b",
    r"\b(cancer|tumor|tumour|oncolog)\b",
    r"\b(maternity|pregnancy|childbirth|delivery|newborn)\b",
    r"\b(bariatric|obesity|weight loss surgery)\b",
    r"\b(cosmetic|aesthetic|plastic surgery)\b",
    r"\b(cataract|knee replacement|hip replacement|joint replacement)\b",
    r"\b(abroad|overseas|international|outside india|outside usa|usa|foreign)\b",
    r"\b(excluded|exclusion|not covered|limitation)\b",
    r"\b(diagnosed|diagnosis|condition)\b",
    r"\b(admitted|admission|inpatient|hospitali[sz]ed)\b",
    r"\bis .{3,60} covered\b",
    r"\bcovered (for|under|by)\b",
]

_SIMPLE_PATTERNS = [
    r"\bwhat is\b",
    r"\bdefine\b",
    r"\bmeaning of\b",
    r"\bsum insured\b",
    r"\bpremium\b",
    r"\bnetwork hospital\b",
    r"\bcontact\b",
    r"\bhow (much|many)\b",
    r"\blist (of|the)\b",
]


def classify_query_fast(query: str) -> QueryType | None:
    """
    Keyword-based classifier. Returns 'simple' or 'complex', or None if uncertain.
    None means fall through to the LLM classifier.
    """
    q = query.lower()

    for pattern in _COMPLEX_PATTERNS:
        if re.search(pattern, q):
            return "complex"

    # Only call simple if no complex signal AND a simple signal is present
    for pattern in _SIMPLE_PATTERNS:
        if re.search(pattern, q):
            return "simple"

    return None  # uncertain — use LLM classifier


def classify_query_llm(query: str, llm, gemini_model=None) -> QueryType:
    """
    LLM-based classifier for ambiguous queries. Single call, max_tokens=10.
    """
    prompt = (
        "Classify this insurance query as 'simple' or 'complex'.\n\n"
        "simple = a direct factual lookup (what is covered, what is the definition, "
        "what is the limit, is X covered — with no conditional reasoning needed)\n"
        "complex = requires reasoning about waiting periods, policy age, pre-existing "
        "conditions, multiple interacting clauses, or 'am I covered if X and Y'\n\n"
        f"Query: {query}\n\n"
        "Reply with exactly one word: simple or complex"
    )
    try:
        from .llm_client import call_llm
        label = call_llm(
            llm, gemini_model,
            [{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        ) or "simple"
        if "complex" in label.lower():
            return "complex"
        return "simple"
    except Exception as e:
        print(f"⚠️ Query classifier failed: {e} — defaulting to simple")
        return "simple"


def route_query(query: str, llm, gemini_model=None) -> QueryType:
    """
    Main router: keyword fast-path first, LLM fallback for uncertain cases.
    """
    fast = classify_query_fast(query)
    if fast is not None:
        print(f"🔀 Router (keyword): '{query[:60]}' → {fast}")
        return fast

    label = classify_query_llm(query, llm, gemini_model)
    print(f"🔀 Router (LLM): '{query[:60]}' → {label}")
    return label
