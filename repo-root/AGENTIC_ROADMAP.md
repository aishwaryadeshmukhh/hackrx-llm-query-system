# Agentic RAG — System State & Roadmap

> Last updated: June 2026. This document reflects what is **actually built and running**, what design decisions were made and why, and what remains.

---

## What Was Built (Starting Point)

A linear RAG pipeline for HackRx 2024:
- PDF → parse → chunk → embed (Pinecone multilingual-e5-large) → index → single LLM call → answer
- FastAPI backend + Streamlit UI
- No structured output schema, no tool use, no reasoning trace

---

## What Is Built Now

### 1. Agentic ReAct Loop (`src/react_agent.py`)

Replaced the single LLM call with a **ReAct (Reason + Act) loop**. The LLM now decides what to retrieve at each step rather than receiving a large context blob.

**How it works:**
- Each iteration: LLM outputs `Thought` + `Action` OR `Final Answer`
- Actions dispatch to one of 5 retrieval tools
- Observations (retrieved chunks) are appended to context for the next step
- Loop exits on `Final Answer` or after `MAX_STEPS = 4`
- If max steps hit, a forced-answer prompt extracts the best conclusion from accumulated evidence

**Why ReAct over single-shot:**
- Single-shot RAG gives the LLM one chance with whatever chunks were retrieved. If the top chunks don't contain the exclusion clause, the answer is wrong with no recovery.
- ReAct lets the agent call `lookup_exclusions` after `search_policy` — it can self-correct mid-loop if the first retrieval is insufficient.
- The reasoning trace is logged per step (thought, action, args, observation) and returned in the API response — observable, not a black box.

**Parser robustness:**
Llama 3.3 outputs tool calls in 3 non-standard formats. `_parse_llm_step()` handles all three:
- `Args: {"key": "val"}` (preferred)
- `Action: tool_name({"key": "val"})` (inline)
- `key="value"` after Action line (key=value style)

---

### 2. Query Router (`src/query_router.py`)

Before hitting the agent loop, every query is classified as `simple` or `complex`.

**Why this matters:**
- Simple queries (definitions, limits, "what is covered") don't need multi-step reasoning — they need one retrieval + one LLM call (~2s)
- Complex queries (waiting periods, pre-existing + policy age, conditional coverage) need the agent loop (~15s)
- Running ReAct on a simple query wastes 3-4 LLM calls and 12 extra seconds

**Classification strategy (two-tier):**
1. **Keyword fast-path** — regex patterns for known complex signals (`waiting period`, `inception`, `pre-existing`, `first year`, `X and Y`) and known simple signals (`what is`, `define`, `sum insured`). Returns immediately without an LLM call.
2. **LLM classifier fallback** — for ambiguous queries, a single Groq call with `max_tokens=10` returns `simple` or `complex`. Cost: ~50 tokens.

---

### 3. Tool Layer (`src/agent_tools.py`)

Five focused retrieval tools the agent can call:

| Tool | What it does |
|---|---|
| `search_policy(query, policy_name)` | General semantic search across the index |
| `lookup_exclusions(procedure_or_condition)` | Targeted search biased toward exclusion sections |
| `check_waiting_period(benefit_type)` | Targeted search biased toward waiting period clauses |
| `get_definitions(term)` | Targeted search biased toward definition sections |
| `compare_policies(query)` | Runs search and groups results by source document |

Each tool returns the top-k chunks with document name, page number, and similarity score. The agent sees these as observations and decides whether to call another tool or conclude.

---

### 4. Calibrated Confidence Scores (`src/query_processor.py`)

Previously: LLM self-reported confidence (made-up number, no grounding).

Now: **retrieval-anchored blending**

```
retrieval_score = mean cosine similarity of top-3 retrieved chunks (0–1, from Pinecone)
blended = 0.4 × retrieval_score + 0.6 × llm_confidence
```

**Why this formula:**
- LLM self-confidence is not calibrated — it will say 0.95 even when the best chunk scored 0.55
- Pure retrieval score ignores whether the LLM actually found the right clause in the text
- 60/40 split keeps LLM judgment dominant (it has clause context) but retrieval quality pulls it down when evidence is weak
- Result is capped to [0.0, 1.0] and rounded to 2dp

For the ReAct path: scores are parsed from observation strings (`score=X.XXX`) in the reasoning trace since the tool calls don't return a unified vector list.

---

### 5. Direct Semantic Search for Simple Path

Previously: simple queries triggered `advanced_search_pinecone()` — a 3-stage pipeline:
- Stage 1: direct semantic search
- Stage 2: query expansion (8 synonym queries)
- Stage 3: context-aware keyword term search

This made 14+ Pinecone embed calls per simple query and was the source of the token explosion (292k chars of context, ~74k tokens per call).

Now: simple path calls `semantic_search_with_similarity()` directly — **1 embed call**, top-5 results, 5 adjacent chunks each side (2+main+2), ~5-6k tokens total.

The advanced multi-stage search still exists in the codebase for future use but is no longer called in any hot path.

---

### 6. Adjacent Chunk Context (Optimised)

Previously: `_get_adjacent_chunks_extended(doc, idx, 25, 25)` — 50 adjacent chunks per vector × 5 vectors = 255 chunks, ~74k tokens.

Now: `adjacent=2` — 2 before + main + 2 after per vector × 5 vectors = 25 chunks, ~5-6k tokens.

**Why 2+2 is enough:**
- Insurance clauses are typically 2-4 paragraphs. A 5-chunk window (±2) captures the full clause plus its heading and the clause before/after.
- The ReAct path doesn't use adjacent chunks at all — each tool call fetches fresh focused vectors.

---

### 7. Groq LLM + Gemini Fallback (`src/llm_client.py`)

**Primary: Groq** (`llama-3.3-70b-versatile`)
- OpenAI-compatible API, fast inference
- 100k tokens/day on free tier

**Fallback: Gemini** (`gemini-3.5-flash`)
- Triggered automatically on any Groq 429 / rate limit
- Uses new `google-genai` SDK (old `google.generativeai` is deprecated)
- Parses `retry_delay { seconds: N }` from the error body to wait the exact time the API specifies — avoids wasting daily quota on premature retries
- Daily quota exhaustion (`per day` in error) stops retrying immediately — no point burning remaining quota

**All 4 LLM call sites use this fallback:**
- Simple path evaluation (`_make_llm_request_with_retry`)
- ReAct loop step (`run_react_loop`)
- ReAct forced final answer
- Query router LLM classifier (`classify_query_llm`)

---

### 8. Cache-Hit Pinecone Re-Indexing (`src/backend.py`)

Previously: cache hits skipped indexing entirely — the ReAct tools queried an empty Pinecone index and returned 0 chunks.

Now: cache hits re-upsert all chunks to Pinecone before running queries. The module-level chunk cache (`_chunk_cache: dict`) stores parsed chunks by MD5 file hash. On a cache hit, re-indexing takes ~30s instead of ~380s (no re-parsing, no re-chunking, just embedding + upsert).

---

### 9. Evaluation Harness (`eval/run_eval.py`)

15 ground-truth Q&A pairs derived from the actual policy text (`BAJHLIP23020V012223.pdf`), covering:
- 6 complex queries (waiting periods, policy age conditions, conditional coverage)
- 9 simple queries (definitions, limits, exclusions)
- Mix of `covered`, `not_covered`, `partial` expected decisions

**Metrics:**
- Decision accuracy: exact match on `covered/not_covered/partial/unclear`
- Clause recall: do expected keywords appear in `answer + justification`?
- Mean confidence: correct vs incorrect answers (validates calibration — correct answers should score higher)
- Simple vs complex accuracy breakdown

**Design:**
- Indexes PDF once, then runs 15 questions sequentially with a 3s gap between calls
- Calls `process_query_routed_sync()` directly — no HTTP overhead, no server needed
- Saves timestamped JSON to `eval/results/`

---

## What Remains

### Phase 6 — Streamlit Reasoning Trace UI (1 day)

The reasoning trace is already in the API response (`reasoning_trace` list). It just needs to be surfaced in `app.py`:

```
Step 1 — Thought: "Need to check waiting period for surgical procedures"
         Action: check_waiting_period("surgical procedures")
         Found: 5 chunks — Excl02 clause, 24-month waiting period

Step 2 — Thought: "Knee replacement is listed — surgery is 2 months in, clearly not covered"
         Final Answer: NOT COVERED (confidence: 0.81)
```

This is the highest-impact remaining feature for demo purposes — makes the reasoning visible instead of just showing a JSON answer.

---

### Phase 7 — Deployment

**Current state:**
- In-memory chunk cache (`_chunk_cache: dict` in `backend.py`) — survives across requests within one process, resets on restart
- Pinecone index cleared after each run (`delete(delete_all=True)`)
- Run telemetry in local JSON files (`artifacts/telemetry/`)

**Production deployment plan:**

| Layer | Local (now) | Cloud |
|---|---|---|
| **Chunk cache** | Module-level Python dict | Redis (Upstash free tier) — key: MD5 file hash, value: serialised chunks |
| **Pinecone index** | Already cloud-hosted | No change |
| **Run telemetry** | Local JSON in `artifacts/` | S3/GCS or Supabase |
| **PDF storage** | Temp files on disk | S3 presigned URL → parse from URL |
| **Server** | Local uvicorn | Render / Railway / GCP Cloud Run (containerised) |

The cache swap is a single function change — replace dict read/write in two places with Redis `GET`/`SET`. Nothing else changes.

**Rate limit strategy for production:**
- Groq 100k TPD is insufficient for sustained load — upgrade to Dev tier ($) or switch primary to Gemini 1M TPD
- Add a per-request token budget check before dispatching to the ReAct loop
- Queue concurrent requests rather than running them all in parallel

---

## Architecture Decision Log (AI Engineering Perspective)

| Decision | Why |
|---|---|
| ReAct over single-shot | Observable reasoning, self-correcting retrieval, handles multi-hop insurance queries that require checking both coverage and exclusions |
| Query router before agent | Prevents 3-4 unnecessary LLM calls on simple factual lookups — saves ~12s and ~3k tokens per simple query |
| Retrieval-anchored confidence | LLM self-confidence is uncalibrated. Blending with cosine similarity grounds the score in actual retrieval quality |
| Adjacent chunk context (2+2 not 25+25) | 25+25 adjacent chunks = 255 chunks = 74k tokens per call = guaranteed rate limit. 2+2 captures full clause context at 5-6k tokens |
| Direct search for simple path | The 3-stage advanced search with 8 expansion queries made sense for complex queries but was overkill for "what is the sum insured" |
| Groq primary, Gemini fallback | Groq is faster and cheaper per call. Gemini has 10x higher daily token budget. Fallback is automatic, transparent to the caller |
| Sequential eval with 3s gaps | Running 15 questions in parallel burns the daily TPD in seconds. Sequential with gaps stays within RPM limits |
| Module-level chunk cache | Avoids re-parsing 380s PDF processing on repeated queries during a demo. Redis swap is a one-function change when deploying |
