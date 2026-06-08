"""
Task 7 - Reranking Module.

Implements the README options:
- Jina cross-encoder reranker via API.
- Qwen cross-encoder reranker via local transformers model.
- MMR (Maximal Marginal Relevance), implemented locally.
- RRF (Reciprocal Rank Fusion), implemented locally.

The public rerank() function keeps a safe offline default so tests and the Task 9
pipeline run without API keys or large model downloads.
"""

import os
import sys
from typing import Any

import requests
from dotenv import load_dotenv

from .task4_chunking_indexing import cosine_similarity, embed_query, hashed_embedding, tokenize

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_RERANK_METHOD = os.getenv("RERANK_METHOD", "cross_encoder")
JINA_RERANK_MODEL = os.getenv("JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_QWEN_PATH = os.path.join(PROJECT_DIR, "models", "rerankings", "Qwen3-Reranker-0.6B")
QWEN_RERANK_MODEL = os.getenv(
    "QWEN_RERANK_MODEL_PATH",
    DEFAULT_QWEN_PATH if os.path.exists(DEFAULT_QWEN_PATH) else "Qwen/Qwen3-Reranker-0.6B",
)
ALLOW_RERANK_FALLBACK = os.getenv("ALLOW_RERANK_FALLBACK", "true").lower() in {"1", "true", "yes"}


def rerank_jina_api(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Cross-encoder reranking with Jina Reranker API.

    Requires JINA_API_KEY. If unavailable and ALLOW_RERANK_FALLBACK=true, falls
    back to the local lexical cross-encoder approximation.
    """
    api_key = os.getenv("JINA_API_KEY", "").strip()
    if not api_key:
        return _rerank_fallback_or_raise("Jina reranker", "missing JINA_API_KEY", query, candidates, top_k)

    try:
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": JINA_RERANK_MODEL,
                "query": query,
                "documents": [c.get("content", "") for c in candidates],
                "top_n": top_k,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json().get("results", [])
        reranked = []
        for result in data:
            item = candidates[int(result["index"])].copy()
            item["score"] = float(result.get("relevance_score", 0.0))
            item["metadata"] = item.get("metadata", {})
            item["rerank_method"] = "jina_api"
            reranked.append(item)
        return reranked[:top_k]
    except Exception as exc:
        return _rerank_fallback_or_raise("Jina reranker", str(exc), query, candidates, top_k)


def rerank_qwen_local(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Cross-encoder reranking with Qwen/Qwen3-Reranker-0.6B via transformers.

    This model is optional and relatively heavy. If transformers/model loading is
    unavailable and ALLOW_RERANK_FALLBACK=true, falls back to the local scorer.
    """
    try:
        pairs = [(query, c.get("content", "")) for c in candidates]
        scores = _score_with_qwen_cross_encoder(pairs)
        if isinstance(scores, float):
            scores = [scores]

        reranked = []
        for candidate, score in zip(candidates, scores):
            item = candidate.copy()
            item["score"] = float(score)
            item["metadata"] = item.get("metadata", {})
            item["rerank_method"] = "qwen_local"
            reranked.append(item)
        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked[:top_k]
    except Exception as exc:
        return _rerank_fallback_or_raise("Qwen reranker", str(exc), query, candidates, top_k)


def _score_with_qwen_cross_encoder(pairs: list[tuple[str, str]]) -> list[float]:
    """Score query-document pairs with the local Qwen CrossEncoder model."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(QWEN_RERANK_MODEL, trust_remote_code=True)
    scores = model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
    return [float(score) for score in scores.tolist()]


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Offline cross-encoder-style fallback.

    It approximates relevance with token overlap plus the incoming retrieval
    score, avoiding network/API requirements for tests.
    """
    query_terms = set(tokenize(query))
    rescored = []
    for candidate in candidates:
        doc_terms = set(tokenize(candidate.get("content", "")))
        overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
        base_score = float(candidate.get("score", 0.0))
        score = 0.65 * overlap + 0.35 * _normalize_score(base_score)
        item = candidate.copy()
        item["score"] = float(score)
        item["metadata"] = candidate.get("metadata", {})
        item["rerank_method"] = "local_cross_encoder_fallback"
        rescored.append(item)
    rescored.sort(key=lambda item: item["score"], reverse=True)
    return rescored[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Maximal Marginal Relevance: balance relevance and diversity."""
    if not candidates or top_k <= 0:
        return []

    prepared = []
    for candidate in candidates:
        item = candidate.copy()
        item["embedding"] = item.get("embedding") or hashed_embedding(item.get("content", ""))
        prepared.append(item)

    selected: list[int] = []
    remaining = list(range(len(prepared)))

    while remaining and len(selected) < top_k:
        best_idx = remaining[0]
        best_score = float("-inf")
        for idx in remaining:
            relevance = cosine_similarity(query_embedding, prepared[idx]["embedding"])
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    cosine_similarity(prepared[idx]["embedding"], prepared[sel]["embedding"])
                    for sel in selected
                )
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if mmr_score > best_score:
                best_idx = idx
                best_score = mmr_score
        prepared[best_idx]["score"] = float(best_score)
        prepared[best_idx]["rerank_method"] = "mmr"
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [prepared[i] for i in selected]


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion for merging ranked lists."""
    scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = _dedupe_key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in content_map or item.get("score", 0) > content_map[key].get("score", 0):
                content_map[key] = item

    fused = []
    for key, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]:
        item = content_map[key].copy()
        item["score"] = float(score)
        item["metadata"] = item.get("metadata", {})
        item["rerank_method"] = "rrf"
        fused.append(item)
    return fused


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = DEFAULT_RERANK_METHOD,
) -> list[dict]:
    """Unified reranking interface."""
    if not candidates or top_k <= 0:
        return []

    normalized = method.lower().replace("-", "_")
    if normalized in {"cross_encoder", "local_cross_encoder"}:
        return rerank_cross_encoder(query, candidates, top_k)
    if normalized in {"jina", "jina_api", "jina_reranker"}:
        return rerank_jina_api(query, candidates, top_k)
    if normalized in {"qwen", "qwen_local", "qwen3"}:
        return rerank_qwen_local(query, candidates, top_k)
    if normalized == "mmr":
        return rerank_mmr(embed_query(query), candidates, top_k)
    if normalized == "rrf":
        return rerank_rrf([candidates], top_k)
    raise ValueError(f"Unknown rerank method: {method}")


def compare_rerank_methods(query: str, candidates: list[dict], top_k: int = 3) -> dict[str, list[dict]]:
    """Run all README methods for comparison in a demo."""
    methods = ["cross_encoder", "jina_api", "qwen_local", "mmr", "rrf"]
    comparison: dict[str, list[dict]] = {}
    for method in methods:
        try:
            comparison[method] = rerank(query, candidates, top_k=top_k, method=method)
        except Exception as exc:
            comparison[method] = [{
                "content": f"{method} unavailable: {exc}",
                "score": 0.0,
                "metadata": {},
                "rerank_method": method,
            }]
    return comparison


def _rerank_fallback_or_raise(
    name: str,
    reason: str,
    query: str,
    candidates: list[dict],
    top_k: int,
) -> list[dict]:
    if not ALLOW_RERANK_FALLBACK:
        raise RuntimeError(f"{name} unavailable: {reason}")
    fallback = rerank_cross_encoder(query, candidates, top_k)
    for item in fallback:
        item["rerank_method"] = f"{name.lower().replace(' ', '_')}_fallback_local"
    return fallback


def _normalize_score(score: float) -> float:
    return max(0.0, min(1.0, score))


def _dedupe_key(item: dict) -> str:
    metadata = item.get("metadata", {})
    source = metadata.get("source", "")
    chunk_index = metadata.get("chunk_index", "")
    return f"{source}:{chunk_index}:{item.get('content', '')[:80]}"


def _demo_candidates() -> list[dict[str, Any]]:
    return [
        {"content": "Dieu 248 toi tang tru trai phep chat ma tuy", "score": 0.8, "metadata": {"source": "bo-luat-hinh-su"}},
        {"content": "Nghe si bi bat vi su dung ma tuy", "score": 0.7, "metadata": {"source": "news"}},
        {"content": "Quy dinh cai nghien bat buoc trong Luat phong chong ma tuy", "score": 0.6, "metadata": {"source": "luat-phong-chong-ma-tuy"}},
        {"content": "Python programming tutorial", "score": 0.4, "metadata": {"source": "irrelevant"}},
    ]


if __name__ == "__main__":
    query_text = "hinh phat tang tru ma tuy"
    for method_name, results in compare_rerank_methods(query_text, _demo_candidates(), top_k=2).items():
        print(f"\nMethod: {method_name}")
        for result in results:
            print(f"[{result['score']:.3f}] [{result.get('rerank_method')}] {result['content'][:100]}")
