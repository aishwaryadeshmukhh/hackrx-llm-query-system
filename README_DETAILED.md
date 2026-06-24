# PolicyMind — Detailed Technical Documentation

> Insurance policy query answering system built for HackRx. Accepts a PDF policy document and natural language questions, and returns structured coverage decisions with cited clauses and a full chain-of-thought reasoning trace.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Request Lifecycle — End to End](#2-request-lifecycle--end-to-end)
3. [Document Processing Pipeline](#3-document-processing-pipeline)
4. [Vector Search — Pinecone](#4-vector-search--pinecone)
5. [Caching Layer — Redis + In-Memory Fallback](#5-caching-layer--redis--in-memory-fallback)
6. [Query Routing](#6-query-routing)
7. [ReAct Agent — Chain of Thought](#7-react-agent--chain-of-thought)
8. [LLM Stack — Groq + Gemini Fallback](#8-llm-stack--groq--gemini-fallback)
9. [Token Optimisation](#9-token-optimisation)
10. [Decision Post-Processing](#10-decision-post-processing)
11. [API Endpoints](#11-api-endpoints)
12. [AI Engineering Challenges and How We Solved Them](#12-ai-engineering-challenges-and-how-we-solved-them)
13. [Configuration and Environment](#13-configuration-and-environment)
14. [Deployment Notes](#14-deployment-notes)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                    │
│  PDF upload + question → POST /hackrx/stream (SSE)          │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    FastAPI Backend                           │
│                                                              │
│  api/main.py          src/backend.py                        │
│  /hackrx/stream  ←→  /hackrx/upload                        │
│       │                    │                                 │
│       └────────┬───────────┘                                │
│                │                                             │
│         src/cache.py  ←→  Upstash Redis (HTTP)             │
│                │                                             │
│         src/query_router.py                                 │
│                │                                             │
│         ┌──────┴──────┐                                     │
│      simple         complex                                  │
│         │               │                                   │
│   direct LLM      src/react_agent.py                       │
│   answer              │                                     │
│                  src/agent_tools.py                         │
│                        │                                     │
│                 src/query_processor.py                      │
│                        │                                     │
│                   Pinecone Index                            │
│              (multilingual-e5-large, 1024d)                 │
└─────────────────────────────────────────────────────────────┘
```

**Why this stack:**
- **FastAPI** for async request handling and Server-Sent Events (SSE) streaming — lets the frontend show reasoning steps as they happen rather than waiting for a full response.
- **Pinecone** managed vector database — handles cosine similarity search across 223 policy chunks without maintaining local FAISS infrastructure.
- **Upstash Redis over HTTP** — serverless Redis with no persistent TCP connection required, compatible with Railway/Vercel deployment where long-lived connections are not guaranteed.
- **Groq** for fast LLM inference (llama-3.3-70b-versatile) — significantly lower latency than OpenAI for the same model class.
- **Gemini** (gemini-3.5-flash) as fallback — separate provider ensures the system degrades gracefully if Groq rate-limits.

---

## 2. Request Lifecycle — End to End

When a user uploads a PDF and asks a question, the following sequence runs:

### Step 1 — PDF hash
```python
file_hash = md5(pdf_bytes)  # e.g. "a3f8c2..."
```
The MD5 hash of the raw file bytes is the universal key for all caching. Same PDF bytes → same hash, regardless of filename. This means re-uploading the same document never re-processes it.

### Step 2 — Query result cache check
```
Redis: GET qr:{file_hash}:{normalised(question)}
```
If this exact PDF + exact question was answered before, the cached answer is returned immediately (~100ms). The agent does not run. The reasoning trace from the original run is included in the cached response so the frontend can still show chain-of-thought.

**Why cache query results:** The ReAct agent loop makes 3-5 LLM calls and 3-5 Pinecone searches per question, taking 30-120 seconds. For a demo or production system where the same policy is queried repeatedly, this is unacceptable. Caching the full answer dict with a 24-hour TTL eliminates this cost entirely on repeat queries.

### Step 3 — Chunk cache check
```
Redis: GET chunks:{file_hash}
→ {"chunks": [...223 items...], "indexed_in_pinecone": true}
```
If the PDF was parsed before, the 223 chunks are loaded from Redis, skipping the ~15-second parse+chunk pipeline. `indexed_in_pinecone` tracks whether those chunks have already been uploaded to Pinecone.

### Step 4 — Pinecone vector count verification
Even if Redis says `indexed_in_pinecone=True`, Pinecone is verified:
```python
stats = pc.Index("policy-index").describe_index_stats()
total_vectors = stats.total_vector_count  # Pydantic attribute, not dict key
if total_vectors == 0:
    # Redis flag is stale — re-index
```
**Why this check exists:** Pinecone's index can be wiped independently of Redis (expired free tier, manual deletion, or — the bug we hit — `delete_all=True` running after every request). Without this check, every post-wipe request would get 0 search results and the LLM would hallucinate answers from no evidence.

### Step 5 — Indexing (only if needed)
If chunks are not yet in Pinecone:
- Batch-embed all chunks via `pc.inference.embed(model="multilingual-e5-large")` in batches of 96
- Upsert vectors with metadata (`chunk_id`, `text`, `section`, `page_number`, `document_name`)
- Mark `indexed_in_pinecone=True` in Redis

### Step 6 — Query routing
The question is classified as `simple` or `complex` by a keyword router. Simple questions (waiting period, coverage limit lookups) go to a direct LLM call. Complex questions (multi-condition eligibility, exclusion-with-exception patterns) go to the ReAct agent.

### Step 7 — ReAct agent or direct answer
For complex queries, the ReAct loop runs (see section 7). For simple queries, a single Pinecone search + LLM call produces the answer.

### Step 8 — Decision post-processing
The LLM's `decision` field is checked against the answer text. If the answer mentions sub-limit language ("up to six weeks", "co-payment", "capped at") but the decision is `covered`, it is deterministically corrected to `partial`.

### Step 9 — Cache and return
The answer dict is stored in Redis (`qr:` key, 24h TTL) and returned to the frontend.

---

## 3. Document Processing Pipeline

### PDF Parsing — `src/parse_documents.py`
Uses **PyMuPDF (fitz)** to extract text page-by-page. For each page:
- `page.get_text()` extracts raw text
- `page.find_tables()` detects tabular content and converts to markdown
- `ordered_content` list preserves page order for downstream chunking

Fallback to simple extraction if enhanced parsing fails on a page.

### Chunking — `src/chunk_documents_optimized.py`
`OptimizedTextChunker` splits documents into chunks with:
- `chunk_size=800` characters
- `chunk_overlap=150` characters
- Paragraph-first splitting, then sentence-boundary splitting, then character fallback

Each chunk carries metadata: `chunk_id`, `document_name`, `chunk_index`, `section`, `page_number`, `content_type`.

**Why 800 chars:** Large enough to contain a full policy clause with context (most exclusion clauses are 200-600 chars), small enough that a top-k=6 retrieval returns focused evidence rather than entire sections. Smaller chunks improve retrieval precision at the cost of context; 800 is a practical midpoint for insurance policy text.

The Bajaj Allianz Global Health Care policy produces **223 chunks** from the PDF.

---

## 4. Vector Search — Pinecone

**Index:** `policy-index`, region `us-east-1`  
**Model:** `multilingual-e5-large` (1024 dimensions, cosine similarity)  
**Rationale for multilingual-e5-large:** Insurance policy text is domain-specific with technical terminology. multilingual-e5-large outperforms smaller models on domain-specific retrieval. The 1024-dim space provides enough separation between semantically similar clauses (e.g. two different exclusion types).

Each tool call in the ReAct agent embeds a targeted sub-query and searches Pinecone:
```python
pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[sub_query],
    parameters={"input_type": "query", "truncate": "END"}
)
index.query(vector=embedding, top_k=6, include_metadata=True)
```

`top_k=6` is a deliberate choice — returning more chunks increases LLM context (and cost/latency), while fewer risks missing the relevant clause when scores are tightly clustered (which they are on this policy: typical scores 0.82-0.87 across all chunks for insurance queries).

---

## 5. Caching Layer — Redis + In-Memory Fallback

**File:** `src/cache.py`

### Two cache types

| Key pattern | Contents | TTL | Purpose |
|---|---|---|---|
| `chunks:{file_hash}` | Parsed chunks + `indexed_in_pinecone` flag | 7 days | Skip re-parse on same PDF |
| `qr:{file_hash}:{normalised_question}` | Full answer dict including trace | 24 hours | Skip agent on repeat question |

### Normalisation
Question text is normalised before use as a cache key:
```python
def _normalise(text):
    return re.sub(r"\s+", " ", text.lower().strip())
```
This ensures "Is bariatric surgery covered?" and "is bariatric surgery covered ?" both hit the same cache entry.

### Redis client — Upstash
```python
from upstash_redis import Redis
_redis = Redis(url=UPSTASH_REDIS_REST_URL, token=UPSTASH_REDIS_REST_TOKEN)
```
Upstash Redis communicates over HTTP REST, not TCP. This matters for serverless deployments (Railway, Vercel) where persistent connections are not guaranteed between requests.

### In-memory fallback
If `UPSTASH_REDIS_REST_URL` is not set, all cache operations fall back to a module-level `dict`. This means local development works with zero setup, but the cache does not persist across server restarts.

### Cache flush endpoint
```
POST /hackrx/cache/flush-queries
```
Deletes all `qr:*` keys from Redis. Necessary when a bug caused incorrect answers to be cached (which happened twice during development — see section 12).

---

## 6. Query Routing

**File:** `src/query_router.py`

Questions are classified as `simple` or `complex` using keyword heuristics before touching the LLM:

- **Simple:** waiting period lookups, single-condition coverage checks, definition queries
- **Complex:** multi-condition eligibility (BMI + comorbidity + policy duration), exclusion-with-exception patterns, plan tier comparisons, geographic coverage scenarios

**Why route at all:** Running the full ReAct agent on every query is wasteful. A question like "What is the waiting period for pre-existing diseases?" has a direct answer in one chunk. The agent adds latency, cost, and a chance of reasoning error for no benefit. Simple routing captures ~30% of queries and answers them in under 3 seconds vs 30-90 seconds for the agent.

---

## 7. ReAct Agent — Chain of Thought

**File:** `src/react_agent.py`

The ReAct (Reason + Act) pattern interleaves reasoning steps with tool calls. Each step produces a `Thought` (the LLM's current understanding) and either an `Action` (tool call) or a `Final Answer`.

### Available tools — `src/agent_tools.py`

| Tool | Purpose | Pinecone query |
|---|---|---|
| `search_policy(query)` | General semantic search | Embedded query |
| `lookup_exclusions(procedure_or_condition)` | Find exclusion clauses | "{condition} exclusion not covered" |
| `check_waiting_period(condition)` | Find waiting period clauses | "{condition} waiting period months" |
| `final_answer(decision, confidence, answer, justification, relevant_clauses)` | End the loop | — |

Each tool embeds a targeted sub-query (not the original question) to improve retrieval precision. A question about bariatric surgery coverage for a specific patient would generate sub-queries like "bariatric surgery exclusion obesity BMI conditions" and "bariatric surgery waiting period months" — these retrieve more relevant chunks than embedding the full question.

### Step prompt structure
```
[System prompt with domain rules]

Question: {question}

[Step 1 thought + action + observation]
[Step 2 thought + action + observation — compressed to ~80 tokens]
...
[Most recent observation — full text, up to 1200 chars per top chunk]

Next step:
```

**Why compress history:** Each completed step's observation is compressed to ~80 tokens (section names + first key clause sentence). The most recent observation is kept full. This reduces the context sent to the LLM by ~65% on a 4-step query, directly cutting Groq token consumption and rate limit pressure.

### Observation formatting — `_format_chunks_for_observation`
- Top 2 chunks: full text, up to 1200 chars each
- Chunks 3-6: section header + first 120 chars only

**Rationale:** The LLM needs full text for the most relevant chunks to reason accurately. Chunks 3-6 are lower-relevance but their section headers serve as evidence that the topic was searched, preventing the agent from re-searching the same area.

### Repeat detection
```python
_prev_calls: set  # (tool_name, json.dumps(args, sort_keys=True))
```
If the agent tries to call the same tool with the same arguments twice, it is redirected rather than blocked:
- If `steps_remaining >= 2`: redirect to `search_policy` with a contextual suggestion
- If `steps_remaining < 2`: force `final_answer` with current evidence

**Why redirect rather than block:** Simply blocking a repeat call leaves the agent with no action and causes a malformed step. Redirecting gives it a productive alternative tool call. The redirect message is constructed using the previous call's args so it suggests a relevant complementary query rather than a hardcoded example.

### Max steps
`MAX_STEPS = 6`. If the loop reaches step 6 without a `final_answer`, the LLM is called one final time with all observations and forced to output an answer.

**Why 6 and not 4:** With 4 steps, a 3-condition query (exclusion check → waiting period check → BMI criteria check) exhausts the budget before synthesis. Step 2 is often consumed by a repeat redirect, leaving only 2 real tool calls. 6 steps allows: exclusion check → redirect to benefit clause → waiting period check → synthesize → final answer, with budget to spare.

### Step streaming via SSE
Each step is emitted as a Server-Sent Event as it completes:
```json
{"type": "thought", "step": 2, "thought": "...", "action": "search_policy", "args": {...}}
{"type": "observation", "step": 2, "observation": "..."}
```
This lets the frontend display reasoning steps progressively rather than waiting for the full answer.

---

## 8. LLM Stack — Groq + Gemini Fallback

**Primary:** Groq `llama-3.3-70b-versatile`  
**Fallback:** Google Gemini `gemini-3.5-flash`

The fallback activates automatically if a Groq call raises an exception (rate limit, timeout, API error):
```python
try:
    response = groq_client.chat.completions.create(...)
except Exception:
    response = gemini_model.generate_content(...)
```

**Why Groq as primary:** Groq's LPU inference hardware delivers ~10x lower latency than standard GPU inference for the same Llama model. A ReAct step that takes ~8s on OpenAI takes ~1-2s on Groq. Given that a complex query runs 4-6 LLM calls, this difference is the gap between a 15-second and 60-second response.

**Why Gemini as fallback and not another Groq model:** Groq rate limits are per-account across all models. If the primary model is rate-limited, switching to another Groq model hits the same limit. Gemini is a completely separate provider.

---

## 9. Token Optimisation

Groq's free tier has aggressive RPM (requests per minute) and TPM (tokens per minute) limits. Two optimisations were implemented to stay within limits:

### Option A — Observation trimming
Rather than passing all retrieved chunks in full to every step, only the top 2 chunks are passed in full. Chunks 3-6 are truncated to 120 chars. Implemented in `_format_chunks_for_observation`.

**Estimated saving:** ~600 tokens per tool call observation (from ~1800 to ~1200).

### Option B — History compression
Completed step observations are compressed to ~80 tokens before being appended to the prompt history. Only the current step's observation is full. Implemented in `_compress_observation`.

**Estimated saving:** On a 4-step query, steps 1-3 history goes from ~5400 tokens to ~240 tokens. Combined A+B saving: ~65% reduction in context tokens per query.

---

## 10. Decision Post-Processing

**Function:** `_correct_decision` in `src/react_agent.py`

The LLM frequently returns `decision=covered` when the correct answer is `decision=partial`. This happens because the LLM focuses on whether the claim is *payable* rather than whether a cap *reduces* the payout.

A deterministic correction runs after every final answer:
```python
_PARTIAL_INDICATORS = [
    "up to six weeks", "up to 6 weeks", "six weeks per trip",
    "sub-limit", "co-payment", "20% co-payment",
    "proportionate deduction", "80% payable",
    "capped at", "maximum benefit amount",
]

if answer["decision"] == "covered":
    text = answer["answer"] + " " + answer["justification"]
    if any(indicator in text.lower() for indicator in _PARTIAL_INDICATORS):
        answer["decision"] = "partial"
```

**Why deterministic rather than asking the LLM:** The LLM already wrote the answer text correctly — it described the 6-week cap, the co-payment, the sub-limit. It just chose the wrong badge. A string check on its own output is more reliable than asking it to reconsider, and adds zero latency.

---

## 11. API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/hackrx/upload` | Multipart: `file` (PDF) + `questions` (newline-separated). Returns JSON with all answers. Used by Swagger and legacy clients. |
| `POST` | `/hackrx/stream` | Multipart: `file` (PDF) + `question` (single). Returns SSE stream with step-by-step events. Used by the Next.js frontend. |
| `POST` | `/hackrx/cache/flush-queries` | Deletes all cached query results from Redis. Use after a model/prompt change to force re-evaluation. |
| `GET` | `/docs` | Swagger UI |

---

## 12. AI Engineering Challenges and How We Solved Them

This section documents every significant problem encountered during development, its root cause, and the fix applied.

---

### Problem 1 — Agent giving inconsistent answers on the same question

**Symptom:** The same question asked twice would sometimes return `covered` and sometimes `not_covered` with different justifications, even with the same PDF.

**Root cause:** Pinecone returns chunks ranked by cosine similarity. For insurance policy text, many chunks score within 0.01 of each other (e.g. 0.827, 0.826, 0.826, 0.825, 0.825). The chunk that appears first in the observation changes between calls due to floating-point non-determinism in Pinecone's ANN index. The LLM would reason differently based on which chunk it saw first.

**Fix:** Improved the system prompt's domain knowledge so the LLM could reason correctly from any chunk ordering. Also added explicit handling for the most common confusion patterns (Section 12 benefit vs exclusion sub-clause, bariatric surgery exclusion-with-exception, etc.).

**Why this is fundamentally hard:** Vector similarity search is approximate by design. Unless scores have a clear gap (>0.05), the "most relevant" chunk is ambiguous. The real fix is better reasoning, not better retrieval ordering.

---

### Problem 2 — `delete_all=True` wiping Pinecone after every request

**Symptom:** Back-to-back queries on the same PDF would always re-index. The second query would get 0 search results mid-agent-loop (after indexing finished for the first query), causing complete hallucination.

**Root cause:** Three separate `delete(delete_all=True)` calls existed in the codebase — in `backend.py` (Step 5 cleanup), in the old batch query path, and as a utility function. The intent was to keep the Pinecone index clean between documents. The unintended consequence was destroying vectors mid-request when two requests overlapped.

**Fix:** Removed all `delete_all=True` calls from the request path. Pinecone vectors now persist across requests. The vector count check at the start of each request handles the edge case of a genuinely empty index.

**Lesson:** "Clean up after yourself" is a reasonable principle for temporary resources, but a shared vector index used across concurrent requests is not a temporary resource.

---

### Problem 3 — Stale bad answers getting cached

**Symptom:** After fixing the `delete_all` bug, queries still returned wrong answers instantly. Backend showed no logs because the answer came from Redis cache.

**Root cause:** The bad answers (generated when Pinecone had 0 vectors) were cached in Redis before we fixed the underlying bug. The cache TTL is 24 hours, so the wrong answer would persist until the next day.

**Fix:** Added `POST /hackrx/cache/flush-queries` endpoint to manually invalidate all query result cache entries. Used this immediately after fixing the `delete_all` bug to force fresh evaluation.

**Lesson:** A cache that stores results from a buggy state is worse than no cache. Any system with a query result cache needs a manual invalidation path.

---

### Problem 4 — Redis `indexed_in_pinecone` flag desynchronised from actual Pinecone state

**Symptom:** Server restart → Redis says `indexed=True` → code skips indexing → agent gets 0 chunks.

**Root cause:** Redis and Pinecone are independent services. Redis stores a flag that says "we indexed this PDF", but Pinecone's index can be wiped independently (free tier expiry, manual deletion, the `delete_all` bug). There was no cross-check.

**Fix:** Added a `describe_index_stats()` call at the start of every request to get the actual vector count from Pinecone. If `total_vector_count == 0`, the Redis flag is treated as stale and re-indexing is forced.

**Note on SDK compatibility:** The newer Pinecone SDK returns a Pydantic model from `describe_index_stats()`, not a plain dict. `.get("total_vector_count")` silently returns `None`. The fix uses `getattr(stats, "total_vector_count", None)` with a dict fallback.

---

### Problem 5 — Agent stopping after 1-2 steps with wrong answer

**Symptom:** The bariatric surgery query (BMI 37 + uncontrolled T2D + 3 years policy) returned `not_covered` after 2 steps. The agent found the exclusion clause and stopped without checking whether the exclusion had an exception or whether the waiting period was satisfied.

**Root cause A — MAX_STEPS = 4 was too low.** A 3-condition query (exclusion check + exception conditions check + waiting period check + synthesis) requires at minimum 4 real tool calls. With a repeat redirect consuming one step, the agent was forced to answer before completing its evidence gathering.

**Root cause B — Repeat redirect used a hardcoded example.** The redirect message said "try `search_policy(query='emergency treatment outside area of cover benefit limit duration')`" — this is an emergency treatment example hardcoded into the agent. When the query was about bariatric surgery, this confused the agent into searching for emergency treatment instead.

**Fix A:** Increased `MAX_STEPS` from 4 to 6.

**Fix B:** Made the redirect message dynamic — it extracts the actual search term from the previous call's args and suggests a relevant complementary query: `search_policy(query="{prev_query} benefit conditions exceptions covered")`.

---

### Problem 6 — `covered` returned when `partial` is correct

**Symptom:** USA heart attack query on Imperial Plus Excluding USA plan returned `covered 89%`. The answer text correctly described the 6-week cap and Sum Insured limit, but the decision badge was `covered` instead of `partial`.

**Root cause:** The LLM's instruction is to determine whether the claim is payable. It correctly determined it is payable. But "payable with a sub-limit cap" is `partial` by definition, and the LLM conflated "payable" with "fully covered".

**Fix:** Post-processing function `_correct_decision` scans the answer text for sub-limit indicator phrases and upgrades `covered` → `partial` deterministically. This is more reliable than re-prompting the LLM because the LLM already wrote the correct evidence — it just chose the wrong badge.

---

### Problem 7 — Query result cache only wired into `/hackrx/stream`, not `/hackrx/upload`

**Symptom:** Swagger testing (which uses `/hackrx/upload`) always re-ran the agent even for repeated questions, while the frontend (which uses `/hackrx/stream`) correctly used the cache.

**Root cause:** The query result cache (`cache.get_query_result` / `cache.set_query_result`) was added to `api/main.py` which handles `/hackrx/stream`, but `src/backend.py` which handles `/hackrx/upload` had its own separate query loop with no cache integration.

**Fix:** Added cache check and cache write inside the `run_query` async function in `_process_pdf_and_answer` in `backend.py`. Each query now checks Redis before running the agent and stores the result after.

---

### Problem 8 — Groq rate limits hit on multi-step queries

**Symptom:** Terminal showed Groq 429 errors mid-agent-loop, falling back to Gemini. Overall query time increased significantly because Gemini is slower.

**Root cause:** A 6-step ReAct query with full observations was sending ~8,000-12,000 tokens per step to Groq, consuming the TPM budget in 2-3 queries.

**Fix:** Two-pronged token reduction:
1. Observation trimming — top 2 chunks full, chunks 3-6 header-only
2. History compression — completed steps compressed to ~80 tokens each

Combined: ~65% reduction in tokens per query, keeping most queries within Groq's free tier limits.

---

### Problem 9 — `NameError: name 'cached' is not defined`

**Symptom:** 500 error on second request to `/hackrx/upload` after Redis integration.

**Root cause:** During the Redis cache refactor, the variable `cached` was renamed to `cached_doc` throughout `backend.py`. Two references were missed at lines 516 and 531.

**Fix:** Simple variable rename. The lesson is that large refactors on long functions should use `replace_all` to catch every reference rather than manual search.

---

### Problem 10 — `index_time` undefined on cache hit path

**Symptom:** 500 error on the first request when the PDF was already in Redis cache.

**Root cause:** `index_time` was initialised inside the `else` branch (first-time parse path). The cache hit path set `already_indexed=True` and skipped the `else` branch entirely, leaving `index_time` undefined when it was referenced in the timing dict.

**Fix:** Added `index_time = 0.0` before the cache/parse branch so it is always defined regardless of which path executes.

---

## 13. Configuration and Environment

### Backend — `backend/.env`
```
PINECONE_API_KEY=...          # Pinecone serverless index
GEMINI_API_KEY=...            # Google Gemini fallback LLM
GROQ_API_KEY=...              # Groq primary LLM
UPSTASH_REDIS_REST_URL=...    # Upstash Redis HTTP endpoint
UPSTASH_REDIS_REST_TOKEN=...  # Upstash Redis auth token
CORS_ORIGINS=http://localhost:3000  # comma-separated for multiple origins
```

### Frontend — `frontend/.env.local`
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Key constants
| Constant | Value | File |
|---|---|---|
| `MAX_STEPS` | 6 | `react_agent.py` |
| `CHUNK_TTL` | 7 days | `cache.py` |
| `QUERY_TTL` | 24 hours | `cache.py` |
| Embedding model | `multilingual-e5-large` | `embed_and_index.py` |
| Embedding dims | 1024 | `embed_and_index.py` |
| Pinecone index | `policy-index` | throughout |
| Primary LLM | `llama-3.3-70b-versatile` | `query_processor.py` |
| Fallback LLM | `gemini-3.5-flash` | `query_processor.py` |
| Chunk size | 800 chars | `chunk_documents_optimized.py` |
| Chunk overlap | 150 chars | `chunk_documents_optimized.py` |

---

## 14. Deployment Notes

### Backend — Railway
- Run command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- Working directory: `backend/`
- All env vars set in Railway dashboard
- Pinecone and Upstash are external managed services — no infrastructure to deploy

### Frontend — Vercel
- `NEXT_PUBLIC_API_URL` must point to the Railway backend URL
- Set `CORS_ORIGINS` on the backend to the Vercel deployment URL

### Redis setup — Upstash
1. Create a free database at console.upstash.io
2. Copy `REST URL` and `REST Token` to backend env vars
3. No further configuration needed — the client verifies connection on startup with a ping

### Why not Docker
The system uses three external managed services (Pinecone, Upstash, Groq/Gemini). Dockerising the backend adds operational complexity (image builds, container registry, port mapping) with no benefit since there is no local state to containerise. Railway deploys directly from the git repo in ~90 seconds and handles scaling, restarts, and HTTPS termination.
