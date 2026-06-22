# Agentic RAG — Implementation Roadmap

> This document captures the current honest state of the system and a concrete plan to evolve it into a genuinely agentic RAG pipeline suitable for an AI engineering portfolio.

---

## Current State of the Project

### What it is

A **linear RAG pipeline** built for HackRx 2024. A user submits a question, the system retrieves document chunks from Pinecone, and passes them to Gemini with a single prompt. It has two interfaces: a Streamlit UI (`app.py`) and a FastAPI backend (`backend.py`) that accepts a PDF URL + list of questions and returns answers in bulk.

### What works well

- End-to-end pipeline is functional: PDF → parse → chunk → embed (Pinecone multilingual-e5-large) → index → query → Gemini answer
- FastAPI endpoint parallelises multiple queries against a freshly uploaded PDF, with proper async task management
- Document registry prevents reprocessing unchanged files
- Graceful fallback when Gemini quota is exceeded
- Async wrappers throughout the pipeline mean it is non-blocking under load

### Honest gaps (RAG layer)

| Area | Issue |
|---|---|
| **Ingestion** | `detect_table_structures()` is fully written in `parse_documents.py` but never called — the actual parse function does `page.get_text()` in a plain loop, discarding all table data |
| **Chunking** | Pure character-count chunking with paragraph/sentence heuristics. No section-boundary awareness — a clause about exclusions and a clause about benefits can end up in the same chunk |
| **Chunk metadata** | Page numbers are extracted during parsing but dropped before the chunk is written to Pinecone |
| **Adjacent-chunk lookup** | `_get_adjacent_chunks_extended()` queries Pinecone with a `[0.0] * 1024` zero vector to find neighbours by document name — this is not guaranteed to return all chunks, is slow, and burns API quota |
| **Hybrid search** | Described as "hybrid" but is actually dense vector search with post-retrieval keyword score boosting on a hardcoded synonym dict — no sparse index, no BM25 |
| **Metadata filtering** | All queries scan the full index regardless of which policy is being asked about |
| **Evaluation schema** | `app.py` renders `decision`, `confidence`, `justification` fields but `_llm_evaluation_with_comprehensive_context()` only returns `answer`, `source_document`, `relevant_sections` — the coverage verdict fields are never actually populated by the LLM |
| **No eval harness** | No test set, no accuracy measurement, no retrieval recall metric |

### Honest gaps (agentic layer)

The system has no agency at all. The LLM is called exactly once per query, given a large context blob, and asked to produce a JSON answer. There is no:
- Query decomposition
- Tool selection
- Iterative retrieval
- Self-verification
- Visible reasoning trace

---

## Target Architecture: Agentic RAG

The goal is to replace the single `process_query()` call with an **agent loop** where the LLM controls what to retrieve, decides when it has enough information, and verifies its own answer before returning it.

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│         Planner Agent           │  ← Decomposes query into sub-questions
│  "What do I need to answer      │    and decides which tools to call first
│   this completely?"             │
└────────────┬────────────────────┘
             │  sub-questions + tool plan
             ▼
┌─────────────────────────────────┐
│       Tool Execution Layer      │  ← Each tool is a focused retrieval call
│                                 │
│  search_policy(q, policy_name)  │  Dense vector search, optional filter
│  lookup_exclusions(procedure)   │  Filtered search on exclusion sections
│  check_waiting_period(benefit)  │  Filtered search on waiting period clauses
│  compare_policies(q)            │  Runs search across all loaded policies
│  get_clause_by_section(ref)     │  Exact section lookup by heading
└────────────┬────────────────────┘
             │  retrieved evidence per tool call
             ▼
┌─────────────────────────────────┐
│        Reasoning Agent          │  ← ReAct loop: Thought → Action → Observation
│                                 │    Runs until confident or max steps hit
│  Thought: "I have coverage      │
│  info but haven't checked       │
│  exclusions yet"                │
│  Action: lookup_exclusions(...)  │
│  Observation: [chunk text]      │
│  Thought: "No exclusion found,  │
│  ready to answer"               │
└────────────┬────────────────────┘
             │  reasoning trace + evidence
             ▼
┌─────────────────────────────────┐
│         Critic Agent            │  ← Single focused check:
│  "Does this answer contradict   │    "Is there any exclusion or waiting
│   any exclusion clause?"        │     period that invalidates this answer?"
└────────────┬────────────────────┘
             │  verified answer or revision request
             ▼
┌─────────────────────────────────┐
│          Synthesizer            │  ← Formats final response:
│                                 │    decision + confidence + clause citations
│  decision: covered/not_covered  │    + full reasoning trace for UI display
│  confidence: 0.0 – 1.0         │
│  relevant_clauses: [...]        │
│  reasoning_trace: [steps]       │
└─────────────────────────────────┘
```

---

## Implementation Plan

### Phase 0 — Fix the Existing Bugs First (1–2 days)

These are quick fixes that unblock everything downstream. Do these before any agentic work.

**0.1 Wire up table extraction**

In `parse_documents.py`, `parse_document_enhanced_pymupdf()` ignores the `detect_table_structures()` function that is already written. Add one call per page and append table content to `ordered_content` with `type: 'table'`.

```python
# In parse_document_enhanced_pymupdf(), inside the page loop:
tables = detect_table_structures(page)
for table in tables:
    ordered_content.append({
        'content': table['content'],
        'type': 'table',
        'page': page_num + 1,
        'source': 'pymupdf_table'
    })
```

**0.2 Preserve page numbers in chunk metadata**

In `chunk_documents_optimized.py`, add `page_number` to the structured chunk dict. The parser already produces page numbers in `ordered_content` — they just need to be threaded through.

**0.3 Fix the evaluation schema**

In `query_processor.py`, `_llm_evaluation_with_comprehensive_context()` prompt should return `decision`, `confidence`, `justification`, and `relevant_clauses` — not just `answer`. Update the prompt and the JSON schema it expects to match what `app.py` actually renders.

**0.4 Replace zero-vector adjacent chunk lookup**

Replace `_get_adjacent_chunks_extended()` with a local in-memory lookup. When chunks are loaded into Pinecone, also write them to a `dict[tuple[doc_name, chunk_index], text]` stored on the `QueryProcessor` instance. Neighbour lookup then becomes a dict access rather than a Pinecone query.

---

### Phase 1 — Tool Layer (2–3 days)

Define the tools the agent can call. Each tool is a thin wrapper around the existing retrieval code with a clear schema.

**File: `src/agent_tools.py`** (new file)

```python
TOOLS = [
    {
        "name": "search_policy",
        "description": "Search the full policy index for text relevant to a question.",
        "parameters": {
            "query": "str — the sub-question to search",
            "policy_name": "str | None — filter to a specific PDF filename, or None for all"
        }
    },
    {
        "name": "lookup_exclusions",
        "description": "Search specifically in exclusion clauses. Use when checking if something is explicitly excluded.",
        "parameters": {
            "procedure_or_condition": "str — the thing to check for exclusion"
        }
    },
    {
        "name": "check_waiting_period",
        "description": "Search specifically for waiting period clauses for a given benefit type.",
        "parameters": {
            "benefit_type": "str — e.g. 'maternity', 'dental', 'pre-existing conditions'"
        }
    },
    {
        "name": "compare_policies",
        "description": "Run a query across all loaded policies and return results from each.",
        "parameters": {
            "query": "str — the question to compare across policies"
        }
    }
]
```

Each tool maps to a method on `QueryProcessor` that does a targeted Pinecone query. `lookup_exclusions` and `check_waiting_period` use Pinecone metadata filters once section labels are stored in chunk metadata (Phase 0 + Phase 2).

---

### Phase 2 — Section-Aware Chunking (2 days)

Replace character-count chunking with section-boundary chunking so each chunk maps to one logical clause.

**In `chunk_documents_optimized.py`**, add a section detector that runs before the current chunker:

```python
SECTION_HEADER_PATTERN = re.compile(
    r'^(?:Section|Clause|Article|Part|Schedule|Annexure)\s+[\dA-Z]+[\.\-]?\s+\w+',
    re.MULTILINE | re.IGNORECASE
)
```

Split the document on these headers first. Each section becomes a chunk (or is further split at sentence boundaries only if it exceeds `max_chunk_size`). Store `section_title` in chunk metadata — this is what enables the targeted tool searches in Phase 1.

---

### Phase 3 — ReAct Agent Loop (3–4 days)

This is the core of the agentic upgrade. Replace `process_query()` in `query_processor.py` with an agent loop.

**The loop (pseudocode):**

```
max_steps = 6
steps = []

for step in range(max_steps):
    prompt = build_react_prompt(original_query, steps, TOOLS)
    response = llm.generate(prompt)          # returns Thought + Action or Final Answer
    
    if response.is_final_answer:
        break
    
    tool_result = execute_tool(response.action)
    steps.append({
        "thought": response.thought,
        "action": response.action,
        "observation": tool_result
    })
```

**The ReAct prompt structure:**

```
You are an insurance policy analyst. Answer the question by using the available tools.
For each step, output:
  Thought: your reasoning about what to do next
  Action: tool_name({"param": "value"})
When you have enough information, output:
  Final Answer: {"decision": ..., "confidence": ..., "justification": ..., "relevant_clauses": [...]}

Question: {original_query}

{previous_steps}
```

The agent decides when it has enough evidence. It can call `lookup_exclusions` after `search_policy` if the initial results mention exclusions. It can call `check_waiting_period` if the policy duration is relevant to the query. This is what makes it genuinely agentic — the retrieval path is not hardcoded.

---

### Phase 4 — Critic Agent (1–2 days)

After the ReAct loop produces an answer, run a single focused verification pass.

```python
critic_prompt = f"""
You are reviewing an insurance coverage decision for correctness.

Original question: {query}
Proposed answer: {react_answer}
All retrieved evidence: {all_retrieved_chunks}

Your task: Check ONLY whether the proposed answer missed any exclusion clause or 
waiting period that would change the decision. 

If you find a contradiction, return:
{{"verdict": "revise", "reason": "...", "missed_clause": "..."}}

If the answer is consistent with all evidence, return:
{{"verdict": "confirmed"}}
"""
```

If the critic returns `"revise"`, the synthesizer adjusts the confidence score downward and appends the missed clause to `relevant_clauses`. This single extra LLM call is cheap and catches the most common failure mode in insurance RAG — silently ignoring exclusions.

---

### Phase 5 — Evaluation Harness (1 day)

Add `eval/run_eval.py` with 10–15 hardcoded question/answer pairs derived from a known policy document. The script runs each question through the full pipeline and reports:

- Exact match on `decision` (covered / not_covered / partial)
- Clause citation recall (did the right clause appear in `relevant_clauses`?)
- Mean confidence score for correct vs incorrect answers

This is the thing that transforms "I built a demo" into "I built a system" in an interview.

```
python eval/run_eval.py --policy docs/sample_policy.pdf

Results:
  Decision accuracy:     11/15  (73%)
  Clause recall:         9/15   (60%)
  Mean confidence (correct):  0.82
  Mean confidence (incorrect): 0.51
```

---

### Phase 6 — UI: Visible Reasoning Trace (1 day)

In `app.py`, display the agent's reasoning steps in a Streamlit expander. This is the feature that makes the demo visually impressive — the user can see the agent thinking.

```
Step 1 — Thought: "I need to check if dental surgery is in the base coverage first"
         Action: search_policy("dental surgery coverage")
         Found: 2 chunks from policy.pdf

Step 2 — Thought: "Coverage found, but policy is only 3 months old — checking waiting periods"
         Action: check_waiting_period("dental")
         Found: "Dental procedures: 12-month waiting period applies"

Step 3 — Thought: "Waiting period exceeds policy duration. Answer is not covered."
         Final Answer: NOT COVERED (confidence: 0.91)
```

---

## File Change Summary

| File | Change |
|---|---|
| `src/parse_documents.py` | Wire up `detect_table_structures()` into the main parse loop |
| `src/chunk_documents_optimized.py` | Add section-header detection, store `section_title` and `page_number` in chunk metadata |
| `src/agent_tools.py` | New file — tool definitions and execution wrappers |
| `src/query_processor.py` | Replace `process_query()` with ReAct agent loop + critic pass; fix evaluation JSON schema |
| `src/pipeline.py` | Thread `section_title`/`page_number` metadata through to Pinecone upsert |
| `app.py` | Add reasoning trace display in query results |
| `eval/run_eval.py` | New file — offline evaluation script |
| `requirements.txt` | No new dependencies needed — Gemini function calling handles tool dispatch |

---

## Estimated Timeline

| Phase | Work | Days |
|---|---|---|
| 0 | Fix existing bugs | 1–2 |
| 1 | Tool layer | 2–3 |
| 2 | Section-aware chunking | 2 |
| 3 | ReAct agent loop | 3–4 |
| 4 | Critic agent | 1–2 |
| 5 | Evaluation harness | 1 |
| 6 | UI reasoning trace | 1 |
| **Total** | | **~2 weeks** |

Phases 0 and 5 can be done independently of each other. Phase 3 depends on Phases 1 and 2 being complete. Phase 4 depends on Phase 3.

---

## Deployment & Caching Strategy

The current chunk cache (`_chunk_cache: dict = {}` in `backend.py`) is in-memory only — it survives across API requests within the same server process but resets on every restart. This is fine for local development and hackathon demos.

When deploying to a cloud environment, the cache should be moved out of process memory:

| Layer | Local (now) | Cloud (production) |
|---|---|---|
| **Chunk cache** | Module-level Python dict | Redis (e.g. Upstash, Redis Cloud) with `file_hash` as key, chunks as JSON value |
| **Pinecone index** | Already cloud-hosted | No change needed |
| **Run telemetry** | Local JSON files in `artifacts/` | Object storage (S3/GCS bucket) or a lightweight DB (PlanetScale, Supabase) |
| **PDF storage** | Temp files on disk | S3/GCS upload before processing; URL passed to parser |

The swap is a single function change — replace the dict read/write in `_cache_get()` / `_cache_put()` with Redis `GET`/`SET` calls. Everything above those functions stays the same.

For the HackRx submission or a portfolio demo, the in-memory cache is sufficient. Add Redis when you move to a persistent deployment (e.g. Render, Railway, or a cloud VM).

---

## Why This Architecture Is Interesting for AI Engineering Roles

Standard RAG is now table stakes. What distinguishes this project after these changes:

1. **Observable reasoning** — the agent's tool calls and thoughts are visible, not hidden inside one black-box prompt
2. **Tool-grounded retrieval** — the LLM controls what gets retrieved based on what it finds, not a hardcoded retrieval pipeline
3. **Self-verification** — the critic pass models a real quality-control pattern used in production LLM systems
4. **Measurable** — the eval harness means you can quote accuracy numbers, which is what separates engineering from hacking
5. **Honest about limits** — confidence scores are calibrated by the critic, not just a number the LLM makes up
