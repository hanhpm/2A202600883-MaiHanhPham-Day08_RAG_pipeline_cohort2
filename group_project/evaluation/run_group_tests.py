"""Run group Option A search/generation scenarios and export summary."""

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from group_project.search_service import run_generation, run_search
from src.task6_lexical_search import lexical_search

SCENARIOS_PATH = Path(__file__).parent / "group_test_scenarios.json"
SUMMARY_PATH = PROJECT_DIR / "group_project" / "group-summary.md"


def load_scenarios() -> list[dict]:
    return json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))


def score_results(results: list[dict], scenario: dict) -> dict:
    if not results:
        return {
            "keyword_recall": 0.0,
            "source_match": 0.0,
            "score": 0.0,
            "top_source": "none",
        }
    text = " ".join(item.get("content", "") for item in results[:3]).lower()
    expected = [kw.lower() for kw in scenario.get("expected_keywords", [])]
    matched = sum(1 for kw in expected if kw in text)
    keyword_recall = matched / max(len(expected), 1)
    top_source = results[0].get("metadata", {}).get("source", "")
    hint = scenario.get("expected_source_hint", "").lower()
    source_match = 1.0 if hint and hint in top_source.lower() else 0.0
    total = round(0.75 * keyword_recall + 0.25 * source_match, 3)
    return {
        "keyword_recall": round(keyword_recall, 3),
        "source_match": source_match,
        "score": total,
        "top_source": top_source,
    }


def run() -> dict:
    scenarios = load_scenarios()
    rows = []
    for scenario in scenarios:
        query = scenario["question"]
        hybrid = run_search(query, top_k=5, rerank_method="cross_encoder")
        bm25 = lexical_search(query, top_k=5, method="bm25")
        hybrid_score = score_results(hybrid["reranked"], scenario)
        bm25_score = score_results(bm25, scenario)
        rows.append({
            "id": scenario["id"],
            "question": query,
            "hybrid_score": hybrid_score,
            "bm25_score": bm25_score,
            "better": "hybrid_rerank" if hybrid_score["score"] >= bm25_score["score"] else "bm25_only",
            "explanation": hybrid["comparison"]["summary"],
        })

    generation_query = "Luật Phòng chống ma túy 2021 quy định hành vi nào bị nghiêm cấm?"
    generation = run_generation(generation_query, top_k=5)

    report = {
        "rows": rows,
        "generation_query": generation_query,
        "generation_answer": generation.get("answer", ""),
        "generation_sources": [
            item.get("metadata", {}).get("source", "unknown")
            for item in generation.get("sources", [])
        ],
        "average_hybrid": round(sum(row["hybrid_score"]["score"] for row in rows) / len(rows), 3),
        "average_bm25": round(sum(row["bm25_score"]["score"] for row in rows) / len(rows), 3),
    }
    SUMMARY_PATH.write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict) -> str:
    lines = [
        "# Group Summary - Option A Search Engine",
        "",
        "## Requirement Coverage",
        "",
        "- Web UI: `group_project/app.py` using Streamlit.",
        "- Hybrid search: semantic search + BM25 + RRF fusion.",
        "- Reranking: Task 7 `cross_encoder` default, optional Jina/Qwen/MMR/RRF.",
        "- Source and relevance score: shown for every result card.",
        "- Comparison: semantic vs BM25 vs hybrid vs reranked.",
        "- Explanation: UI explains why a stage is better/worse.",
        "- Logs: every query is appended to `group_project/logs/search_logs.jsonl` and `.csv`.",
        "- Generation with citation: integrated through Task 10 in the app and tested below.",
        "",
        "## Accuracy Summary",
        "",
        f"- Average hybrid+rerank heuristic score: `{report['average_hybrid']}`",
        f"- Average BM25-only heuristic score: `{report['average_bm25']}`",
        "",
        "| ID | Better Config | Hybrid Score | BM25 Score | Top Hybrid Source | Top BM25 Source |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {id} | {better} | {hs:.3f} | {bs:.3f} | {hsrc} | {bsrc} |".format(
                id=row["id"],
                better=row["better"],
                hs=row["hybrid_score"]["score"],
                bs=row["bm25_score"]["score"],
                hsrc=row["hybrid_score"]["top_source"],
                bsrc=row["bm25_score"]["top_source"],
            )
        )

    lines.extend([
        "",
        "## Scenario Analysis",
        "",
    ])
    for row in report["rows"]:
        lines.extend([
            f"### {row['id']}",
            "",
            f"Question: {row['question']}",
            "",
            f"Why better/worse: {row['explanation']}",
            "",
        ])

    lines.extend([
        "## Generation With Citation Scenario",
        "",
        f"Question: {report['generation_query']}",
        "",
        "Answer:",
        "",
        report["generation_answer"],
        "",
        "Sources:",
        "",
    ])
    for source in report["generation_sources"]:
        lines.append(f"- {source}")

    lines.extend([
        "",
        "## Team Architecture",
        "",
        "Streamlit UI -> Search Service -> Task 5 Semantic + Task 6 BM25 -> RRF -> Task 7 Rerank -> Results",
        "",
        "Citation flow: Streamlit UI -> Task 10 Generation -> source/citation display.",
        "",
        "## Repository Push",
        "",
        "Code is prepared locally. Push to the shared repository with:",
        "",
        "```powershell",
        "git add group_project src",
        "git commit -m \"Add group search engine option A\"",
        "git push",
        "```",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    result = run()
    print(f"Wrote {SUMMARY_PATH}")
    print(json.dumps({
        "average_hybrid": result["average_hybrid"],
        "average_bm25": result["average_bm25"],
    }, ensure_ascii=False, indent=2))
