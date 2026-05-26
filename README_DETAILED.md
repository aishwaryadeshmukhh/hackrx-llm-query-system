**Overview**
- **Goal:** Document the implemented RAG pipeline in this workspace (the `JBBR-Backend` snapshot) with exact code pointers, configuration, dependencies, limitations, and recommended improvements.

**Contents**
- **1. `src/query_processor.py`** — retrieval and query flow (code excerpts + explanation)
- **2. `src/embed_and_index.py`** — Pinecone indexing and embedding pipeline
- **3. LLM integration** — prompts, parsing, fallbacks
- **4. Document processing** — parsing, chunking, metadata
- **5. Streamlit app state management** — how UI maintains state and errors
- **6. Testing & validation** — existing artifacts, sample I/O, known failures

**1. `src/query_processor.py` — Retrieval & Query Flow**
- **Exact retrieval query code (direct semantic search):**

The primary synchronous vector search is implemented in `semantic_search` and `semantic_search_with_similarity`.

Example: vector-only search (from `semantic_search`) — see [JBBR-Backend/src/query_processor.py](JBBR-Backend/src/query_processor.py#L1099-L1124)

```python
query_embedding = self._encode_query(query)
response = self.index.query(
    vector=query_embedding,
    top_k=top_k,
    include_metadata=True
)
search_results = []
for m in response.matches:
    search_results.append({
        "id": m.id,
        "score": m.score,
        "text": m.metadata.get("text", ""),
        "document_name": m.metadata.get("document_name", ""),
        "page_number": m.metadata.get("page_number", 1)
    })
```

- **Advanced multi-stage search:** `advanced_search_pinecone` runs stages: direct search, expanded search (`_perform_expanded_search`), and context search (`_perform_context_search`), then deduplicates and normalizes scores. See [JBBR-Backend/src/query_processor.py](JBBR-Backend/src/query_processor.py#L515-L612).

- **Context expansion logic:** `_expand_context` and `_get_adjacent_chunks_extended` fetch adjacent chunks from the same document using a metadata-only Pinecone query (dummy vector) and then combine them with `_combine_chunks_with_context`. See [JBBR-Backend/src/query_processor.py](JBBR-Backend/src/query_processor.py#L834-L897).

```python
adjacent_chunks = self._get_adjacent_chunks_extended(doc_name, chunk_index, 25, 25)
expanded_text = self._combine_chunks_with_context(candidate.get("text", ""), adjacent_chunks, context_chars)
candidate["text"] = expanded_text
candidate["context_expanded"] = True
```

- **Confidence scoring calculation:** there are two places where confidence is derived:
  - In `analyze_with_gemini` (example flow in `example.py`) confidence is categorized from average vector score:

```python
avg_score = sum(r["score"] for r in search_results) / len(search_results)
if avg_score > 0.7:
    confidence = "high"
elif avg_score > 0.4:
    confidence = "medium"
else:
    confidence = "low"
```

  - The UI expects `evaluation.get("confidence")` (a numeric 0-1 in some code paths) — there is not a single unified numeric confidence function in `query_processor.py`; hybrid scoring uses vector similarity plus keyword boosts (`_calculate_hybrid_score`) and stores `final_score` on candidates. See [JBBR-Backend/src/query_processor.py](JBBR-Backend/src/query_processor.py#L686-L723).

- **Entity extraction implementation (LLM-based):** `_llm_entity_extraction` builds a prompt and calls LLM via `_make_llm_request_with_retry`. See [JBBR-Backend/src/query_processor.py](JBBR-Backend/src/query_processor.py#L201-L241).

Entity-extraction prompt (exact):

```text
Extract the following entities from this insurance/medical query: "{query}"

Return a JSON object with these fields (use null if not found):
- age: integer (patient age)
- gender: string (M/F/Male/Female)
- procedure: string (medical procedure/surgery)
- location: string (city/location)
- policy_duration: string (how old is the policy)
- policy_type: string (type of insurance policy)
- amount: number (any monetary amount mentioned)

Only return the JSON, no other text.
```

- **Query expansion method:** `_generate_expanded_queries` contains a synonyms dictionary for insurance domain terms and creates expanded queries replacing matched keywords with synonyms. See [JBBR-Backend/src/query_processor.py](JBBR-Backend/src/query_processor.py#L426-L460).

**2. `src/embed_and_index.py` — Pinecone operations**
- **Index schema & metadata fields (upsert shape):** when indexing, each vector is upserted as a tuple `(id, vector, metadata)` where `metadata` contains:
  - `text` (snippet, up to 1000 chars)
  - `document_name`
  - `page_number`
  - `chunk_id`

See metadata build in `index_chunks_in_pinecone` — [JBBR-Backend/src/embed_and_index.py](JBBR-Backend/src/embed_and_index.py#L128-L146).

- **Batch embedding process:** `generate_embeddings_batch` initializes `Pinecone(api_key=api_key)` and calls `pc.inference.embed(model="multilingual-e5-large", inputs=batch, parameters=...)` in batches (default batch_size 96). See [JBBR-Backend/src/embed_and_index.py](JBBR-Backend/src/embed_and_index.py#L1-L80).

```python
pc = Pinecone(api_key=api_key)
response = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=batch,
    parameters={"input_type": "passage", "truncate": "END"}
)
for embedding in response.data:
    batch_embeddings.append(embedding.values)
```

- **Index creation & dimension management:** `check_or_create_pinecone_index` checks the existing index dimension and will delete+recreate the index if it differs from required (default 1024). See [JBBR-Backend/src/embed_and_index.py](JBBR-Backend/src/embed_and_index.py#L88-L140).

- **Upsert / batching:** `index_chunks_in_pinecone` converts each chunk into `(chunk_id, embedding_list, metadata)` and upserts in batches of 100. See [JBBR-Backend/src/embed_and_index.py](JBBR-Backend/src/embed_and_index.py#L150-L220).

- **Search execution code:** queries in `query_processor.py` call `index.query(vector=..., top_k=..., include_metadata=True)`; `embed_and_index.py` also exposes index management utilities such as `delete_duplicate_vectors`, `reindex_documents`, and `get_index_stats` for administrative operations. See [JBBR-Backend/src/embed_and_index.py](JBBR-Backend/src/embed_and_index.py#L1-L300).

- **Filters / post-processing:** post-retrieval processing (filtering, deduplication, normalization, balancing content types) is implemented in `query_processor.py` via `_process_search_results`, `_deduplicate_results`, `_normalize_scores`, `_balance_content_types`.

**3. LLM Integration — Prompts & Parsing**
- **System/User prompt templates** (examples):
  - **Entity extraction prompt** (see above).
  - **Comprehensive evaluation prompt** (used in `_llm_evaluation_with_comprehensive_context`) — exact prompt assembled in `query_processor.py` and includes: QUERY, extracted entities, TOP 5 VECTORS SUMMARY, COMPREHENSIVE POLICY CONTEXT, and strict instructions to return JSON only. See [JBBR-Backend/src/query_processor.py](JBBR-Backend/src/query_processor.py#L936-L1016).
  - **Analyze-with-Gemini prompt** (in `example.py`) differs for table vs text content; text prompt requests synthesis while table prompt asks to indicate table IDs and recommend DataframeAnalysisTool. See [JBBR-Backend/example.py](JBBR-Backend/example.py#L432-L520).

- **Response parsing code:** `_extract_json_from_response` tries json.loads on entire response, otherwise searches for a `{...}` substring and parses it. See [JBBR-Backend/src/query_processor.py](JBBR-Backend/src/query_processor.py#L85-L122).

```python
try:
    return json.loads(response_text)
except json.JSONDecodeError:
    # find first { and last } and try again
    start = response_text.find('{')
    end = response_text.rfind('}') + 1
    json_str = response_text[start:end]
    return json.loads(json_str)
```

- **Fallback rule-based logic:** if Gemini/LLM is unavailable, `_generate_fallback_response` builds a simple answer from the top vector content using `_extract_relevant_answer_from_content` (a heuristic text extraction) and returns a low-confidence answer. See [JBBR-Backend/src/query_processor.py](JBBR-Backend/src/query_processor.py#L1056-L1088).

- **Hallucination detection:** No explicit hallucination-detection module is implemented. The code does use retrieval-augmented context to mitigate hallucination and uses JSON extraction checks. Recommend adding: answer grounding checks (verify numbers and quoted clauses against source metadata), and token-level traceability.

**4. Document Processing — Actual Implementation**
- **PDF parsing (edge cases):** `parse_document_enhanced_pymupdf` (in `parse_documents.py`) extracts page text with `page.get_text()`, cleans text, detects tables (via `find_tables` if available), and records `ordered_content` entries per page. It falls back to `parse_document_simple_fallback` on errors. See [JBBR-Backend/src/parse_documents.py](JBBR-Backend/src/parse_documents.py#L1-L220).

Key snippet (table detection):

```python
if hasattr(page, 'find_tables'):
    found_tables = page.find_tables()
    for table in found_tables:
        table_data = table.extract()
        # convert to DataFrame and then to markdown for storage
```

- **Chunking function & boundary logic:** `chunk_documents_optimized.chunk_documents_optimized` uses `OptimizedTextChunker` with config values: chunk_size=800, chunk_overlap=150, min_chunk_size derived, split_on_sentences=True, preserve_paragraphs=True. Chunks are created via paragraph-first, then sentence-splitting, with fallbacks to character-based chunking. See [JBBR-Backend/src/chunk_documents_optimized.py](JBBR-Backend/src/chunk_documents_optimized.py#L1-L200).

Key chunk metadata fields are populated as:

```json
{
  "chunk_id": "{document_name}_{i}_{md5}",
  "document_name": "...",
  "content": "...",
  "chunk_index": i,
  "char_count": n,
  "metadata": {
    "chunking_method": "optimized_text",
    "source_document": document_name,
    "chunk_size_config": 800,
    "overlap_config": 150
  }
}
```

- **Metadata extraction:** `parse_documents` builds `ordered_content` entries with `page` and `source`, and indexing attaches `document_name`, `page_number`, and `chunk_id` to Pinecone metadata during upsert.

- **Document registry comparison logic:** `DocumentRegistry` computes MD5 of file contents and stores per-file `hash`, `indexed_at`, `chunk_count`. `get_document_status` returns 'indexed'|'changed'|'new'|'missing' for files. See [JBBR-Backend/src/document_registry.py](JBBR-Backend/src/document_registry.py#L1-L120).

**5. Streamlit app state management**
- **Session state initialization:** `app.py` ensures keys exist in `st.session_state`:

```python
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'processing_stats' not in st.session_state:
    st.session_state.processing_stats = {}
if 'query_results' not in st.session_state:
    st.session_state.query_results = None
if 'persistent_query' not in st.session_state:
    st.session_state.persistent_query = ''
```

- **Data persistence between runs:** results are stored in `st.session_state.query_results` (so reruns keep results visible). Processing statistics are stored in `st.session_state.processing_stats` after pipeline runs.

- **Error handling:** The app catches exceptions at top-level around pipeline and query calls and displays them via `st.error()`. Pipeline functions return structured `{'success': False, 'error': ...}` objects which are rendered. See `app.py` UI handling for `single_click_pipeline` and `query_documents` sections [JBBR-Backend/app.py](JBBR-Backend/app.py#L1-L240).

**6. Testing & Validation**
- **Existing tests:** There are no formal unit tests found in the repository. The project contains `example.py` which contains illustrative example flows and basic heuristics (score→confidence mapping) used for manual testing. See [JBBR-Backend/example.py](JBBR-Backend/example.py#L1-L1200).

- **Sample input/output pairs (recommended):**
  - Input: Query: "Is dental surgery covered for a 35-year-old female?" → Expected: Decision (covered/not), confidence, justification quoting policy clause(s). The Streamlit README includes a JSON example response in `README.md`.

- **Known failures / edge cases**
  - Pinecone API missing/invalid → indexing and embedding fail (fallback is random small vectors in embed_and_index but not suitable for production).
  - Gemini quota exceeded → LLM calls return None; code falls back to heuristic `_generate_fallback_response` but result is lower quality.
  - Index recreation is destructive when dimensions mismatch — existing vectors are removed.
  - Table extraction may fail for scanned PDFs; `parse_documents` falls back to simple extraction but OCR components are not fully integrated.

- **Performance metrics & monitoring**
  - `src/performance_monitor.py` provides utilities for measuring duration, memory usage, and recommending tuning options. It also contains `estimate_indexing_time` which assumes ~0.1s embedding per chunk and ~2s per 100 upsert batch. See [JBBR-Backend/src/performance_monitor.py](JBBR-Backend/src/performance_monitor.py#L1-L140).

**Configuration values & external dependencies**
- **Important constants & defaults**
  - Embedding model / dim: `multilingual-e5-large` → 1024 dims
  - Pinecone index name: `policy-index`
  - Chunk defaults: `chunk_size=800`, `chunk_overlap=150`
  - Batch sizes: embedding batch 96, upsert batch 100

- **Dependencies (from JBBR-Backend/requirements.txt)**
  - pinecone, google-generative-ai (genai), streamlit, PyMuPDF (fitz), pandas, sentencepiece and other standard libs. See [JBBR-Backend/requirements.txt](JBBR-Backend/requirements.txt).

**Limitations & Assumptions**
- Assumes availability of Pinecone (managed) for embeddings and index storage — local fallback not enabled.
- Assumes Gemini (Google) for LLM reasoning; code has basic fallback paths but they are lower-quality.
- No formal unit test suite or CI currently present — testing is manual via `example.py` and Streamlit UI.

**Recommended Next Improvements (prioritized)**
1. Add local embedding + FAISS fallback so the system can run offline and for CI.
2. Implement numeric confidence aggregation (map `final_score` → numeric confidence 0-1) and display consistently in UI.
3. Add automated unit tests for parsing, chunking, embedding pipeline, and retrieval evaluation (golden pairs).
4. Add hallucination detection: cross-check LLM assertions against source snippets by string matching and number checking.
5. Add safe index migrations (non-destructive) and versioning for schema changes.

---
References: implementation files cited above:
- [JBBR-Backend/src/query_processor.py](JBBR-Backend/src/query_processor.py)
- [JBBR-Backend/src/embed_and_index.py](JBBR-Backend/src/embed_and_index.py)
- [JBBR-Backend/src/parse_documents.py](JBBR-Backend/src/parse_documents.py)
- [JBBR-Backend/src/chunk_documents_optimized.py](JBBR-Backend/src/chunk_documents_optimized.py)
- [JBBR-Backend/app.py](JBBR-Backend/app.py)
