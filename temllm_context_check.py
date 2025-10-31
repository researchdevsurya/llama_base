#!/usr/bin/env python3

"""
College RAG Context Retriever
-----------------------------
Loads your FAISS vector index and college CSV, embeds a query, and
prints out the most relevant database excerpts — without any text generation.
"""

import os
import faiss
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
CSV_FILE = "colleges_processed.csv"      # preprocessed CSV with combined column
FAISS_INDEX_FILE = "colleges.index"      # saved FAISS index
EMBED_MODEL_NAME = "intfloat/e5-base-v2" # must match the model used for indexing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5                                # number of results to show
SIMILARITY_THRESHOLD = 0.2               # skip low-similarity results

# -----------------------------
# Load Database & FAISS Index
# -----------------------------
if not os.path.exists(CSV_FILE) or not os.path.exists(FAISS_INDEX_FILE):
    raise FileNotFoundError("❌ Missing database or FAISS index. Build them first.")

print(f"📂 Loading database: {CSV_FILE}")
df = pd.read_csv(CSV_FILE)

print(f"📦 Loading FAISS index: {FAISS_INDEX_FILE}")
index = faiss.read_index(FAISS_INDEX_FILE)
print(f"✅ Index contains {index.ntotal} vectors")

# -----------------------------
# Load Embedding Model
# -----------------------------
print(f"🧠 Loading embedding model: {EMBED_MODEL_NAME} on {DEVICE}")
emb_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

# -----------------------------
# Search Function
# -----------------------------
def search_colleges(query: str, top_k: int = TOP_K):
    """Search the FAISS database and return top matching college entries."""
    print(f"\n🔍 Searching for: '{query}'")

    # Prepare query text (same prefix format as DB build)
    q_text = f"query: {query}"
    q_emb = emb_model.encode([q_text], convert_to_numpy=True)
    q_emb = q_emb.astype("float32")
    faiss.normalize_L2(q_emb)

    # Search
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        if score < SIMILARITY_THRESHOLD:
            continue
        row = df.iloc[idx]
        results.append((float(score), row["combined"]))
    return results

# -----------------------------
# Interactive Loop
# -----------------------------
def main():
    print("\n🎓 College Context Retriever — type 'exit' to quit.\n")
    while True:
        query = input("Enter your query: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        results = search_colleges(query)
        if not results:
            print("⚠️  No relevant results found.")
            continue

        print("\n📘 Top Contexts:")
        print("-" * 80)
        for rank, (score, text) in enumerate(results, 1):
            print(f"#{rank} (similarity={score:.3f})")
            print(text[:1200].strip())  # show truncated
            print("-" * 80)
        print()

if __name__ == "__main__":
    main()
