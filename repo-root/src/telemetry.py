"""
telemetry.py — Run logging and output storage for the Insurance RAG API.

Two things are written per request:
  1. sample_outputs/<run_id>.json  — clean API response (what Swagger returned)
  2. artifacts/telemetry/<run_id>.json — full telemetry record for perf analysis

The telemetry record schema is stable so runs can be compared over time.
"""

import json
import os
import time
import hashlib
import datetime
from typing import Any, Dict, List, Optional


SAMPLE_OUTPUTS_DIR = "sample_outputs"
TELEMETRY_DIR = os.path.join("artifacts", "telemetry")


def _ensure_dirs():
    os.makedirs(SAMPLE_OUTPUTS_DIR, exist_ok=True)
    os.makedirs(TELEMETRY_DIR, exist_ok=True)


def _run_id(filename: str, ts: str) -> str:
    """Stable run ID: filename_stem + timestamp, e.g. policy_20260622_143201"""
    stem = os.path.splitext(filename)[0][:32].replace(" ", "_")
    return f"{stem}_{ts}"


def _safe_dump(path: str, data: dict):
    """Write JSON, skipping any non-serialisable values."""
    def default(obj):
        try:
            return str(obj)
        except Exception:
            return "<unserializable>"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=default)


def save_run(
    *,
    filename: str,
    file_hash: str,
    questions: List[str],
    answers: List[Dict],
    timing: Dict[str, Any],
    cache_hit: bool,
    endpoint: str = "/hackrx/upload",
    per_query_timings: Optional[List[float]] = None,
    chunk_count: int = 0,
    model_info: Optional[Dict] = None,
) -> str:
    """
    Persist one run. Returns the run_id so the caller can include it in the response.

    sample_outputs/<run_id>.json  — stripped response (decisions + answers only)
    artifacts/telemetry/<run_id>.json — full telemetry for perf comparison
    """
    _ensure_dirs()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = _run_id(filename, ts)

    # ── 1. Clean API output ──────────────────────────────────────────────────
    clean_output = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "file": filename,
        "questions": questions,
        "answers": answers,
        "timing": timing,
    }
    _safe_dump(os.path.join(SAMPLE_OUTPUTS_DIR, f"{run_id}.json"), clean_output)

    # ── 2. Full telemetry record ─────────────────────────────────────────────
    per_q = per_query_timings or []
    telemetry = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "endpoint": endpoint,

        # Request metadata
        "request": {
            "filename": filename,
            "file_hash": file_hash,
            "question_count": len(questions),
            "questions": questions,
            "cache_hit": cache_hit,
            "chunk_count": chunk_count,
        },

        # Top-level timing (seconds)
        "timing": {
            "total_s": timing.get("total_seconds", 0),
            "index_s": timing.get("index_seconds", 0),
            "embed_s": timing.get("embed_seconds", 0),
            "query_s": timing.get("query_seconds", 0),
            "cache_hit": cache_hit,
        },

        # Per-query breakdown
        "per_query": [
            {
                "index": i,
                "question": questions[i] if i < len(questions) else "",
                "time_s": per_q[i] if i < len(per_q) else None,
                "decision": answers[i].get("decision") if i < len(answers) else None,
                "confidence": answers[i].get("confidence") if i < len(answers) else None,
            }
            for i in range(len(questions))
        ],

        # Throughput
        "throughput": {
            "questions_per_second": round(
                len(questions) / timing.get("query_seconds", 1), 3
            ) if timing.get("query_seconds", 0) > 0 else None,
            "chunks_per_second": round(
                chunk_count / timing.get("index_seconds", 1), 1
            ) if timing.get("index_seconds", 0) > 0 and not cache_hit else None,
        },

        # Model info
        "model": model_info or {
            "embedding": "multilingual-e5-large",
            "llm": "gemini-2.5-flash",
            "embedding_dims": 1024,
        },

        # Answer quality signals
        "quality": {
            "decisions": [a.get("decision") for a in answers],
            "confidences": [a.get("confidence") for a in answers],
            "mean_confidence": round(
                sum(a.get("confidence", 0) for a in answers) / len(answers), 3
            ) if answers else None,
            "unclear_count": sum(1 for a in answers if a.get("decision") == "unclear"),
        },
    }
    _safe_dump(os.path.join(TELEMETRY_DIR, f"{run_id}.json"), telemetry)

    print(f"📝 Run saved → sample_outputs/{run_id}.json | artifacts/telemetry/{run_id}.json")
    return run_id


def load_telemetry_summary() -> List[Dict]:
    """
    Load all telemetry records and return a list of summary rows,
    sorted by timestamp descending. Used by the /hackrx/runs endpoint.
    """
    _ensure_dirs()
    records = []
    for fname in os.listdir(TELEMETRY_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(TELEMETRY_DIR, fname), encoding="utf-8") as f:
                t = json.load(f)
            records.append({
                "run_id": t.get("run_id"),
                "timestamp": t.get("timestamp"),
                "file": t["request"].get("filename"),
                "questions": t["request"].get("question_count"),
                "cache_hit": t["request"].get("cache_hit"),
                "total_s": t["timing"].get("total_s"),
                "index_s": t["timing"].get("index_s"),
                "query_s": t["timing"].get("query_s"),
                "mean_confidence": t["quality"].get("mean_confidence"),
                "unclear_count": t["quality"].get("unclear_count"),
                "decisions": t["quality"].get("decisions"),
            })
        except Exception as e:
            print(f"⚠️ Could not read telemetry file {fname}: {e}")
    records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return records
