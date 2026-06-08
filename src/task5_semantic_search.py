"""
Task 5 - Semantic Search Module.
"""

import sys

from .task4_chunking_indexing import EMBEDDING_MODEL, INDEX_FILE, cosine_similarity, embed_query, ensure_index

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Dense retrieval over the local vector index.

    Returns list[dict] with content, score and metadata sorted descending.
    """
    if top_k <= 0:
        return []

    query_embedding = embed_query(query)
    results = []
    for item in ensure_index():
        score = cosine_similarity(query_embedding, item.get("embedding", []))
        if score <= 0:
            continue
        results.append({
            "content": item["content"],
            "score": float(score),
            "metadata": item.get("metadata", {}),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Index file: {INDEX_FILE}")
    for result in semantic_search("hinh phat ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
