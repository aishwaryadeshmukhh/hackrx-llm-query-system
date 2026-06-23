"""
Run from backend/ directory:
    python check_index.py
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone

PINECONE_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "policy-index"

pc = Pinecone(api_key=PINECONE_KEY)

# List all indexes
print("=== All indexes ===")
for idx in pc.list_indexes():
    print(f"  {idx.name} | host: {idx.host}")

# Stats for our index
index = pc.Index(INDEX_NAME)
stats = index.describe_index_stats()
print(f"\n=== Stats for '{INDEX_NAME}' ===")
print(f"  Total vectors: {stats.total_vector_count}")
print(f"  Namespaces: {dict(stats.namespaces)}")

# Test query directly (same way the backend does it)
from src.embed_and_index import generate_query_embedding_pinecone
vec = generate_query_embedding_pinecone("emergency treatment outside area of cover", PINECONE_KEY)
res = index.query(vector=vec, top_k=3, include_metadata=True)
print(f"\n=== Direct query (no namespace) ===")
print(f"  Matches: {len(res.matches)}")
for m in res.matches:
    print(f"  score={m.score:.4f} | metadata_keys={list((m.metadata or {}).keys())}")

# Also try with namespace=""
res2 = index.query(vector=vec, top_k=3, include_metadata=True, namespace="")
print(f"\n=== Query with namespace='' ===")
print(f"  Matches: {len(res2.matches)}")
for m in res2.matches:
    print(f"  score={m.score:.4f}")
