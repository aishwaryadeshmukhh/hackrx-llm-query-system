"""
api/main.py — FastAPI entry point for the HackRx Insurance RAG backend.

Run from the backend/ directory:
    uvicorn api.main:app --reload --port 8000
"""

import json
import os
import queue
import re
import sys
import tempfile
import threading

# Ensure backend/ root is on the path so `src.*` imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Import the existing app and its helpers so all existing routes are included
from src.backend import app, _chunk_cache, _hash_file  # noqa: F401  (re-exports existing routes)

# ── Streaming endpoint ────────────────────────────────────────────────────────

@app.post("/hackrx/stream", tags=["Insurance RAG"], summary="Stream ReAct reasoning steps as SSE")
async def stream_query(request: Request):
    """
    Upload a PDF + one question via multipart/form-data.
    Fields: file (PDF), question (string)

    Returns a Server-Sent Events stream. Each event is JSON with a `type` field:
      {"type": "status",      "message": "..."}
      {"type": "thought",     "step": N, "thought": "...", "action": "...", "args": {...}}
      {"type": "observation", "step": N, "observation": "..."}
      {"type": "answer",      "decision": "...", "confidence": 0.0, "answer": "...",
                               "justification": "...", "relevant_clauses": [...]}
      {"type": "error",       "message": "..."}
      {"type": "done"}
    """
    import asyncio

    try:
        form = await request.form()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse form: {e}")

    file = form.get("file")
    if file is None or not hasattr(file, "filename"):
        raise HTTPException(status_code=400, detail="Missing 'file' field")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    question_raw = form.get("question") or form.get("questions") or ""
    if isinstance(question_raw, bytes):
        question_raw = question_raw.decode("utf-8")
    question = str(question_raw).split("\n")[0].strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' field")

    contents = await file.read()
    tmpdir   = tempfile.mkdtemp()
    pdf_path = os.path.join(tmpdir, file.filename)
    with open(pdf_path, "wb") as f:
        f.write(contents)

    q: queue.Queue = queue.Queue()

    def _sse(obj: dict) -> str:
        return f"data: {json.dumps(obj)}\n\n"

    def _run():
        try:
            pinecone_key = os.getenv("PINECONE_API_KEY")
            groq_key     = os.getenv("GROQ_API_KEY") or ""
            gemini_key   = os.getenv("GEMINI_API_KEY") or ""

            # ── Index PDF ────────────────────────────────────────────────
            q.put({"type": "status", "message": "Parsing and indexing PDF…"})

            file_hash = _hash_file(pdf_path)
            cached    = _chunk_cache.get(file_hash)

            if cached:
                chunks = cached["chunks"]
                q.put({"type": "status", "message": "Cache hit — re-indexing to Pinecone…"})
            else:
                from src.parse_documents import load_and_parse_documents
                from src.chunk_documents_optimized import chunk_documents_optimized
                parsed = load_and_parse_documents([pdf_path])
                transformed = [{
                    "document_name":   d.get("document_name", ""),
                    "content":         d.get("parsed_output", {}).get("content", ""),
                    "ordered_content": d.get("parsed_output", {}).get("ordered_content", []),
                } for d in parsed]
                chunks = chunk_documents_optimized(transformed)
                _chunk_cache[file_hash] = {"chunks": chunks}

            from src.embed_and_index import index_chunks_in_pinecone
            index_chunks_in_pinecone(
                chunks=chunks,
                pinecone_api_key=pinecone_key,
                pinecone_env="us-east-1",
                index_name="policy-index",
            )
            q.put({"type": "status", "message": f"Indexed {len(chunks)} chunks. Starting analysis…"})

            # ── Build processor ──────────────────────────────────────────
            from src.query_processor import QueryProcessor
            processor = QueryProcessor(
                pinecone_api_key=pinecone_key,
                groq_api_key=groq_key,
                gemini_api_key=gemini_key,
                index_name="policy-index",
            )
            processor.populate_chunk_cache(chunks)

            # ── Route query ──────────────────────────────────────────────
            from src.query_router import route_query
            query_type = route_query(question, processor._groq_client, processor._gemini_model)
            q.put({"type": "status", "message": f"Query classified as '{query_type}'. Reasoning…"})

            if query_type == "simple":
                result     = processor.process_query_routed_sync(question)
                evaluation = result.get("evaluation", {})
                q.put({
                    "type":             "answer",
                    "decision":         evaluation.get("decision", "unclear"),
                    "confidence":       evaluation.get("confidence", 0.0),
                    "answer":           evaluation.get("answer", ""),
                    "justification":    evaluation.get("justification", ""),
                    "relevant_clauses": evaluation.get("relevant_clauses", []),
                    "query_type":       "simple",
                })
            else:
                from src.agent_tools import ToolExecutor
                from src.react_agent import run_react_loop

                executor = ToolExecutor(processor)

                def on_step(event: dict):
                    event["query_type"] = "complex"
                    q.put(event)

                loop_result = run_react_loop(
                    question=question,
                    llm=processor._groq_client,
                    gemini_model=processor._gemini_model,
                    tool_executor=executor,
                    on_step=on_step,
                )

                answer      = loop_result.get("answer", {})
                raw_scores  = []
                for step in loop_result.get("reasoning_trace", []):
                    raw_scores += [float(s) for s in re.findall(r"score=([\d.]+)", step.get("observation", "") or "")]
                pseudo_vecs = [{"score": float(s)} for s in raw_scores[:5]]
                answer["confidence"] = processor._calibrate_confidence(
                    float(answer.get("confidence", 0.5)), pseudo_vecs
                )

                q.put({
                    "type":             "answer",
                    "decision":         answer.get("decision", "unclear"),
                    "confidence":       answer.get("confidence", 0.5),
                    "answer":           answer.get("answer", ""),
                    "justification":    answer.get("justification", ""),
                    "relevant_clauses": answer.get("relevant_clauses", []),
                    "query_type":       "complex",
                })

        except Exception as e:
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put({"type": "done"})

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    async def event_generator():
        loop = asyncio.get_event_loop()
        while True:
            try:
                event = await loop.run_in_executor(None, lambda: q.get(timeout=120))
                yield _sse(event)
                if event.get("type") in ("done", "error"):
                    break
            except Exception:
                yield _sse({"type": "error", "message": "Stream timeout"})
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
