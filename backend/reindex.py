"""
Run from backend/ directory:
    python reindex.py

Clears all vectors from policy-index and re-indexes from the docs folder.
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone

PINECONE_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "policy-index"

import time

pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)

# Step 1: Re-index from docs folder (force_reindex_all clears registry + index internally)
DOCS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
print(f"\nRe-indexing from: {DOCS_FOLDER}")
print(f"Files: {os.listdir(DOCS_FOLDER)}")

from src.embed_and_index import force_reindex_all
result = force_reindex_all(DOCS_FOLDER, PINECONE_KEY, INDEX_NAME)
print(f"\nResult: {result}")

time.sleep(5)
stats = index.describe_index_stats()
print(f"Vectors after re-index: {stats.total_vector_count}")

# Step 3: Quick sanity check
print("\nSanity check - querying 'emergency treatment outside area of cover'...")
from src.embed_and_index import generate_query_embedding_pinecone
vec = generate_query_embedding_pinecone("emergency treatment outside area of cover", PINECONE_KEY)
mag = sum(x*x for x in vec) ** 0.5
print(f"Query vector magnitude: {mag:.4f}")

res = index.query(vector=vec, top_k=3, include_metadata=True)
for m in res.matches:
    text = (m.metadata or {}).get("text", "")[:120].replace("\n", " ")
    print(f"  score={m.score:.4f} | {text}")
