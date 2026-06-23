"""
Run from backend/ directory:
    python diagnose_pinecone.py

Checks what's actually in Pinecone and tests a few queries directly.
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone
from src.embed_and_index import generate_query_embedding_pinecone

PINECONE_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME   = "policy-index"

pc    = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)

# ── 1. Index stats ────────────────────────────────────────────────────────────
stats = index.describe_index_stats()
print(f"\n=== Index stats ===")
print(f"  Total vectors : {stats.total_vector_count}")
print(f"  Namespaces    : {dict(stats.namespaces)}")

# ── 2. Fetch a random sample to see what metadata looks like ─────────────────
print(f"\n=== Sample metadata (first 3 matches on a broad query) ===")
probe_vec = generate_query_embedding_pinecone("insurance policy coverage", PINECONE_KEY)
probe = index.query(vector=probe_vec, top_k=3, include_metadata=True)
for m in probe.matches:
    content_preview = (m.metadata or {}).get("content", "")[:120].replace("\n", " ")
    print(f"  id={m.id}  score={m.score:.4f}  doc={m.metadata.get('document_name','?')}  page={m.metadata.get('page_number','?')}")
    print(f"    content: {content_preview}")

# ── 3. Direct queries that the agent is failing on ───────────────────────────
test_queries = [
    "geographical coverage excluding USA emergency treatment",
    "emergency medical treatment abroad outside area of cover",
    "Imperial Plus plan emergency six weeks outside area",
    "emergency treatment outside area of cover Imperial Plus",
]

# ── 3b. Fetch one vector to see ALL metadata keys ────────────────────────────
print(f"\n=== Full metadata of first match ===")
first_id = probe.matches[0].id
fetched = index.fetch(ids=[first_id])
vec_data = fetched.vectors.get(first_id)
if vec_data:
    print(f"  id: {first_id}")
    print(f"  metadata keys: {list(vec_data.metadata.keys())}")
    for k, v in vec_data.metadata.items():
        val_preview = str(v)[:200].replace('\n', ' ')
        print(f"    {k}: {val_preview}")

print(f"\n=== Query test showing actual text field ===")
for q in test_queries:
    vec = generate_query_embedding_pinecone(q, PINECONE_KEY)
    res = index.query(vector=vec, top_k=10, include_metadata=True)
    print(f"\nQuery: '{q}'")
    if not res.matches:
        print("  -> NO MATCHES RETURNED (Pinecone returned empty)")
    else:
        for m in res.matches[:3]:
            md = m.metadata or {}
            # show the actual text field
            text = md.get("text", md.get("content", "EMPTY"))[:120].replace("\n", " ")
            print(f"  score={m.score:.4f} | text: {text}")

# ── 4. Simulate exact _process_search_results logic ──────────────────────────
print(f"\n=== Simulating _process_search_results with min_score=0.03 ===")
vec = generate_query_embedding_pinecone("emergency treatment outside area of cover Imperial Plus", PINECONE_KEY)
res = index.query(vector=vec, top_k=15, include_metadata=True)
print(f"  Raw matches from Pinecone: {len(res.matches)}")
passed = []
for m in res.matches:
    if m.score >= 0.03 and m.metadata:
        content = m.metadata.get('content', '') or m.metadata.get('text', '')
        if content:
            passed.append((m.score, content[:100]))
print(f"  After score+content filter: {len(passed)} chunks")
for score, text in passed[:5]:
    print(f"    {score:.4f}: {text.replace(chr(10),' ')}")
