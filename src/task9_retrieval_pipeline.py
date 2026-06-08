"""
Task 9 - Complete Retrieval Pipeline.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Run semantic + lexical retrieval, merge with RRF, rerank, then fallback to
    PageIndex-style vectorless retrieval if confidence is too low.
    """
    if top_k <= 0:
        return []

    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 3)
    for item in merged:
        item["source"] = "hybrid"

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    for item in final_results:
        item["source"] = "hybrid"
        item["metadata"] = item.get("metadata", {})

    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        return pageindex_search(query, top_k=top_k)

    return final_results[:top_k]


if __name__ == "__main__":
    queries = [
        "Hinh phat cho toi tang tru trai phep chat ma tuy",
        "Nghe si nao bi bat vi su dung ma tuy",
        "Luat phong chong ma tuy 2021 quy dinh gi ve cai nghien",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        for i, result in enumerate(retrieve(q, top_k=3), 1):
            print(f"  {i}. [{result['score']:.3f}] [{result['source']}] {result['content'][:80]}...")
