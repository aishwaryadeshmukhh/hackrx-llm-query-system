# RAG Pipeline — Issues, Root Causes, and Fixes

## 1. Emoji UnicodeEncodeError → random vectors → 0 Pinecone matches

**Symptom:** Every tool call returned 0 chunks. Backend said `query embedding OK (1024 dims)` but Pinecone returned no matches.

**Root cause:** `generate_query_embedding_pinecone()` in `embed_and_index.py` had emoji `print()` statements (`✅`, `❌`, `📦`) inside the `try` block. On Windows with cp1252 terminal encoding, these threw `UnicodeEncodeError`, which was caught by `except Exception`. The except block returned a random near-zero vector `[random.uniform(-0.01, 0.01) for _ in range(1024)]` instead of the real embedding. Cosine similarity against a near-zero vector is undefined — Pinecone returns 0 matches.

**Fix:** Remove all emoji from `print()` statements in `embed_and_index.py`. Replace with ASCII equivalents (`[embed] query embedding OK`, `[embed] ERROR`, etc.).

**Key lesson:** A function that silently returns a fallback value on exception can mask critical failures. The `[embed] query embedding OK` print was inside the `try` block *after* the return — so it only printed when the embedding succeeded, but the except branch also returned without printing an error because the emoji itself was the error.

---

## 2. Stale Pinecone index — vectors indexed with broken embeddings

**Symptom:** After fixing the emoji bug, queries still returned 0 chunks. The `DocumentRegistry` marked the PDF as already indexed, so re-uploading via the frontend did nothing.

**Root cause:** The 223 vectors in Pinecone were indexed using the broken (random near-zero) embedding function. The index existed and had the right vector count, but every stored vector was meaningless. `smart_index_documents` checks a local `DocumentRegistry` file — since the PDF hash hadn't changed, it skipped re-indexing.

**Fix:** Use `force_reindex_all()` which clears the registry before indexing. Created `reindex.py` script to call this directly. Also fixed `clear_pinecone_index()` to handle the "namespace not found" 404 error that Pinecone throws when the index is empty.

**Command:**
```
cd backend
python reindex.py
```

---

## 3. `policy_name` filter matching nothing in Pinecone

**Symptom:** `search_policy('...', policy_name='Imperial Plus')` → 0 chunks. `lookup_exclusions` on the same query returned 8 chunks fine.

**Root cause:** `index_chunks_in_pinecone` stores metadata with `document_name = "BAJHLIP23020V012223.pdf"` (the actual filename). When the agent passed `policy_name="Imperial Plus"`, `_search_policy` turned this into a Pinecone filter `{"document_name": {"$eq": "Imperial Plus"}}` — which matches nothing because no chunk has that document_name.

**Fix:** Only apply the `document_name` filter when `policy_name` ends in `.pdf`:
```python
if policy_name and policy_name.endswith(".pdf"):
    filter_dict = {"document_name": {"$eq": policy_name}}
else:
    filter_dict = None
```

---

## 4. LLM passing wrong argument names to tools (`{"key": "value"}`)

**Symptom:** Terminal showed `check_waiting_period` called with `{'key': 'emergency treatment outside area of cover'}` — literally `"key"` as the argument name.

**Root cause:** The system prompt had a generic `Args: {"key": "value"}` placeholder. The LLM copied `"key"` literally instead of using the real parameter name.

**Fix:** Replace the generic placeholder with concrete per-tool examples in the system prompt:
```
- check_waiting_period(benefit_type): Search waiting period clauses
  Args example: {"benefit_type": "pre-existing disease"}
```

---

## 5. Agent looping — same tool called 3+ times

**Symptom:** Steps 3–5 all called `check_waiting_period('medical emergency outside usa')` identically. 7 steps consumed, max reached, forced answer with low confidence.

**Root cause:** The system prompt said "do NOT repeat a tool call" but the LLM ignored it when stuck (no good chunks from `search_policy`, kept trying `check_waiting_period` as the only tool returning results).

**Fix:** Added hard repeat-call detection in `react_agent.py`. Before executing a tool, scan `steps_so_far` for identical previous calls. If found, inject a skip notice into the context instead:
```python
prev_calls = re.findall(r"Action: (\w+)\((\{[^)]*?\})\)", steps_so_far)
repeat_count = sum(1 for t, a in prev_calls if t == tool_name and a == json.dumps(args))
if repeat_count >= 1:
    # inject skip message, continue loop
```

---

## 6. Query terminology mismatch — colloquial vs policy language

**Symptom:** `search_policy('medical emergency coverage outside usa')` → 0 chunks. The index had the right data (sanity check showed score 0.86 for "emergency treatment outside area of cover").

**Root cause:** The policy document uses formal language: "emergency treatment outside area of cover", "area of cover", "sum insured", "hospitalisation". The LLM generated queries using colloquial terms: "outside usa", "coverage limit", "hospital stay". These are semantically different enough that cosine similarity dropped below threshold.

**Fix:** Added `_normalise_query()` in `ToolExecutor` that rewrites colloquial phrases to policy-domain terms before embedding:
```python
_QUERY_REPHRASE = [
    (re.compile(r'\boutside\s+usa\b', re.I), 'outside area of cover'),
    (re.compile(r'\bexcluding\s+usa\b', re.I), 'outside area of cover'),
    ...
]
```

---

## 7. `_process_search_results` score threshold too high

**Symptom:** Pinecone returned matches but `_process_search_results` filtered all of them out.

**Root cause:** `min_score=0.03` was set, but some valid queries (especially after the terminology mismatch) returned scores of 0.01–0.02 for real relevant chunks.

**Fix:** Lowered `min_score` from `0.03` → `0.01` across all tool methods. Added a hard fallback in `_search_policy`: if score filter drops everything but Pinecone did return matches, return top-5 unfiltered so the agent always gets something.

---

## 8. Query router misclassifying coverage questions as "simple"

**Symptom:** "A patient with schizophrenia... is inpatient treatment covered?" classified as `simple` → went through single-retrieval path → vague answer with no clause citation.

**Root cause:** The keyword patterns in `_COMPLEX_PATTERNS` only caught waiting periods, pre-existing conditions, and policy duration. Any direct coverage question ("is X covered?") that didn't mention those keywords fell through to the LLM classifier, which often said `simple` for questions that looked like direct lookups.

**Fix:** Expanded `_COMPLEX_PATTERNS` with broad coverage question patterns:
```python
r"\b(mental illness|psychiatric|schizophrenia|...)\b",
r"\b(admitted|admission|inpatient|hospitali[sz]ed)\b",
r"\bis .{3,60} covered\b",
r"\bcovered (for|under|by)\b",
r"\b(excluded|exclusion|not covered|limitation)\b",
```
These patterns catch almost every coverage question and route it to ReAct.

---

## 9. Observation text truncated — missing key clause details

**Symptom:** Answer said "covered" but didn't mention "up to six weeks per trip" or "treatment must start within 24 hours". The justification referenced the right section but the detail was missing.

**Root cause:** `_format_chunks_for_observation()` truncated chunk text at 300 characters. The Section I-12 clause is ~500 characters — the "six weeks per trip" condition was after the 300-char cutoff.

**Fix:** Increased truncation limit from 300 → 600 characters in `react_agent.py`.

---

## 10. MAX_STEPS too high → rate limit exhaustion

**Symptom:** Agent took 6–7 steps on every query, hitting Groq TPM limits and falling back to Gemini which had daily quota exhausted.

**Root cause:** `MAX_STEPS=7` allowed the agent to make 6 tool calls before being forced to answer. With `top_k=15` and 600-char chunks, each step was ~800–1000 tokens of context. 6 steps × ~1000 tokens = ~6000 tokens per query, hitting Groq free tier TPM limits.

**Fix:** Reduced `MAX_STEPS` from 7 → 4. Updated system prompt rules to enforce answering by step 3 if coverage + waiting period clauses are found.

---

## Current Architecture Summary

```
Query → Router (keyword fast-path → LLM fallback)
           ↓ complex                    ↓ simple
    ReAct loop (MAX_STEPS=4)    Single retrieval + LLM eval
           ↓
    Tools: search_policy / lookup_exclusions / check_waiting_period / get_definitions
           ↓
    Pinecone (multilingual-e5-large, 1024-dim, cosine)
           ↓
    LLM: Groq llama-3.3-70b-versatile (primary) → Gemini gemini-3.5-flash (fallback)
           ↓
    Final Answer: {decision, confidence, answer, justification, relevant_clauses}
```

## Remaining Improvements (to do)

- **Groq rate limits**: Switch to `llama-3.1-8b-instant` for lower token usage, or add per-query token budget
- **Chunk text in metadata**: Currently truncated at 1000 chars at index time (`meta['text'] = chunk['content'][:1000]`). Some long clauses may be cut. Consider increasing to 1500.
- **Section-level metadata**: Enrich Pinecone metadata with parsed section headers so `search_policy` can filter by section, not just filename.
- **Redis cache**: Cache embeddings + query results for repeated queries (Upstash free tier).
- **Deployment**: Railway (backend) + Vercel (frontend).
