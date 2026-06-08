"""Group Option A search service.

This module wraps the individual Task 5-10 pipeline for the Streamlit UI and
offline group evaluation.
"""

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank, rerank_rrf
from src.task10_generation import generate_with_citation

PROJECT_DIR = Path(__file__).resolve().parents[1]
GROUP_DIR = PROJECT_DIR / "group_project"
LOG_DIR = GROUP_DIR / "logs"
LOG_JSONL = LOG_DIR / "search_logs.jsonl"
LOG_CSV = LOG_DIR / "search_logs.csv"


def run_search(query: str, top_k: int = 5, rerank_method: str = "cross_encoder") -> dict:
    """Run semantic, lexical, hybrid fusion and reranking for comparison."""
    started = time.perf_counter()
    semantic = semantic_search(query, top_k=top_k * 2)
    lexical = lexical_search(query, top_k=top_k * 2, method="bm25")
    hybrid = rerank_rrf([semantic, lexical], top_k=top_k * 2)
    reranked = rerank(query, hybrid, top_k=top_k, method=rerank_method)

    for item in semantic:
        item["retrieval_stage"] = "semantic"
    for item in lexical:
        item["retrieval_stage"] = "lexical_bm25"
    for item in hybrid:
        item["retrieval_stage"] = "hybrid_rrf"
    for item in reranked:
        item["retrieval_stage"] = "reranked"

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    comparison = explain_comparison(query, semantic, lexical, hybrid, reranked)
    payload = {
        "query": query,
        "top_k": top_k,
        "rerank_method": rerank_method,
        "semantic": semantic[:top_k],
        "lexical": lexical[:top_k],
        "hybrid": hybrid[:top_k],
        "reranked": reranked,
        "comparison": comparison,
        "elapsed_ms": elapsed_ms,
    }
    write_log(payload)
    return payload


def run_generation(query: str, top_k: int = 5) -> dict:
    """Run Task 10 generation with citation and log the answer."""
    started = time.perf_counter()
    result = generate_with_citation(query, top_k=top_k)
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    write_log({
        "query": query,
        "top_k": top_k,
        "mode": "generation_with_citation",
        "answer_preview": result.get("answer", "")[:500],
        "sources": _source_names(result.get("sources", [])),
        "elapsed_ms": result["elapsed_ms"],
    })
    return result


def explain_comparison(query: str, semantic: list[dict], lexical: list[dict], hybrid: list[dict], reranked: list[dict]) -> dict:
    """Explain why one retrieval stage is better/worse for this query."""
    query_terms = set(query.lower().split())
    stages = {
        "semantic": semantic,
        "lexical_bm25": lexical,
        "hybrid_rrf": hybrid,
        "reranked": reranked,
    }
    rows = {}
    for name, results in stages.items():
        top = results[0] if results else {}
        content = top.get("content", "").lower()
        overlap = len(query_terms & set(content.split()))
        rows[name] = {
            "top_score": float(top.get("score", 0.0)) if top else 0.0,
            "source": top.get("metadata", {}).get("source", "none") if top else "none",
            "query_term_overlap": overlap,
            "reason": _stage_reason(name, overlap, bool(results)),
        }

    best_stage = max(rows, key=lambda key: (rows[key]["query_term_overlap"], rows[key]["top_score"]))
    return {
        "best_stage": best_stage,
        "stages": rows,
        "summary": (
            f"Best stage: {best_stage}. It has stronger query-term overlap and/or score. "
            "Semantic helps paraphrases, BM25 helps exact legal terms, hybrid+rerank balances both."
        ),
    }


def write_log(payload: dict) -> None:
    """Append every user query/run to JSONL and CSV logs."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with LOG_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_exists = LOG_CSV.exists()
    with LOG_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "query", "top_k", "rerank_method", "mode", "elapsed_ms", "top_source", "top_score"],
        )
        if not csv_exists:
            writer.writeheader()
        top_values = payload.get("reranked") or payload.get("sources") or [{}]
        top = top_values[0] if isinstance(top_values, list) and top_values else {}
        if isinstance(top, str):
            top_source = top
            top_score = ""
        else:
            top_source = top.get("metadata", {}).get("source", "")
            top_score = top.get("score", "")
        writer.writerow({
            "timestamp": record["timestamp"],
            "query": payload.get("query", ""),
            "top_k": payload.get("top_k", ""),
            "rerank_method": payload.get("rerank_method", ""),
            "mode": payload.get("mode", "search"),
            "elapsed_ms": payload.get("elapsed_ms", ""),
            "top_source": top_source,
            "top_score": top_score,
        })


def load_recent_logs(limit: int = 20) -> list[dict]:
    if not LOG_JSONL.exists():
        return []
    lines = LOG_JSONL.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-limit:] if line.strip()]


def _source_names(results: list[dict]) -> list[str]:
    return [item.get("metadata", {}).get("source", "unknown") for item in results]


def _stage_reason(name: str, overlap: int, has_results: bool) -> str:
    if not has_results:
        return "No result returned, so this stage is weaker for this query."
    if name == "semantic":
        return "Good for meaning/paraphrase; weaker when exact article numbers are needed." if overlap < 2 else "Good semantic match with useful term overlap."
    if name == "lexical_bm25":
        return "Strong for exact keywords, names, and law article numbers." if overlap else "Low exact overlap, so BM25 is weaker here."
    if name == "hybrid_rrf":
        return "Combines semantic and BM25 rankings, reducing single-method blind spots."
    return "Reranking reorders hybrid candidates by query relevance and usually improves top precision."
