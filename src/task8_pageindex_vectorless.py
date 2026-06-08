"""
Task 8 - PageIndex Vectorless RAG.

README requires:
1. Upload documents to PageIndex.
2. Provide pageindex_search(query, top_k) for vectorless retrieval fallback.

This module uses the real PageIndex SDK when PAGEINDEX_API_KEY is available and
stores uploaded doc_ids in data/pageindex_documents.json. If PageIndex is not
available, pageindex_search falls back to a local vectorless lexical search so
automated tests still run.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .task4_chunking_indexing import chunk_documents, load_documents, tokenize

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
LEGAL_PDF_DIR = DATA_DIR / "landing" / "legal_pdf"
STANDARDIZED_DIR = DATA_DIR / "standardized"
PAGEINDEX_MANIFEST = DATA_DIR / "pageindex_documents.json"

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
PAGEINDEX_POLL_SECONDS = float(os.getenv("PAGEINDEX_POLL_SECONDS", "3"))
PAGEINDEX_POLL_ATTEMPTS = int(os.getenv("PAGEINDEX_POLL_ATTEMPTS", "20"))
ALLOW_PAGEINDEX_FALLBACK = os.getenv("ALLOW_PAGEINDEX_FALLBACK", "true").lower() in {"1", "true", "yes"}


def upload_documents(force: bool = False, limit: int | None = None) -> list[dict]:
    """
    Upload legal PDF documents to PageIndex and persist doc_ids.

    PageIndex SDK submit_document currently targets PDF files, so this uploads
    files from data/landing/legal_pdf. Existing manifest entries are reused
    unless force=True.
    """
    client = _pageindex_client()
    manifest = [] if force else _load_manifest()
    uploaded_by_path = {item.get("path"): item for item in manifest}

    pdf_files = sorted(LEGAL_PDF_DIR.glob("*.pdf"))
    if limit is not None:
        pdf_files = pdf_files[:limit]

    for pdf_file in pdf_files:
        rel_path = str(pdf_file.relative_to(PROJECT_DIR))
        if rel_path in uploaded_by_path:
            continue

        try:
            response = client.submit_document(str(pdf_file))
        except Exception as exc:
            if "LimitReached" in str(exc):
                manifest = _manifest_from_remote_documents(client)
                _save_manifest(manifest)
                return manifest
            raise
        doc_id = response.get("doc_id") or response.get("id")
        if not doc_id:
            raise RuntimeError(f"PageIndex upload did not return doc_id for {pdf_file.name}: {response}")

        item = {
            "doc_id": doc_id,
            "filename": pdf_file.name,
            "path": rel_path,
            "type": "legal_pdf",
            "size_bytes": pdf_file.stat().st_size,
            "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        manifest.append(item)
        uploaded_by_path[rel_path] = item
        print(f"Uploaded {pdf_file.name} -> {doc_id}")

    _save_manifest(manifest)
    return manifest


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex.

    If uploaded PageIndex doc_ids are available, query PageIndex. Otherwise use
    local vectorless fallback with source='pageindex'.
    """
    if top_k <= 0:
        return []

    try:
        manifest = _load_manifest()
        if not manifest:
            manifest = upload_documents()
        return _pageindex_search_remote(query, top_k, manifest)
    except Exception as exc:
        if not ALLOW_PAGEINDEX_FALLBACK:
            raise RuntimeError(f"PageIndex search unavailable: {exc}") from exc
        return _pageindex_search_local(query, top_k)


def _pageindex_search_remote(query: str, top_k: int, manifest: list[dict]) -> list[dict]:
    client = _pageindex_client()
    results = []

    for doc in manifest:
        doc_id = doc["doc_id"]
        retrieval = client.submit_query(doc_id=doc_id, query=query, thinking=False)
        retrieval_id = retrieval.get("retrieval_id") or retrieval.get("id")
        if not retrieval_id:
            continue

        payload = _poll_retrieval(client, retrieval_id)
        for item in _extract_retrieval_items(payload):
            results.append({
                "content": item.get("content", ""),
                "score": float(item.get("score", 0.0)),
                "metadata": {
                    "source": doc.get("filename"),
                    "doc_id": doc_id,
                    "retrieval_id": retrieval_id,
                    **item.get("metadata", {}),
                },
                "source": "pageindex",
            })

    results = [item for item in results if item["content"]]
    results.sort(key=lambda item: item["score"], reverse=True)
    if results:
        return results[:top_k]
    return _pageindex_chat_search_remote(query, top_k, manifest)


def _pageindex_chat_search_remote(query: str, top_k: int, manifest: list[dict]) -> list[dict]:
    client = _pageindex_client()
    doc_ids = [doc["doc_id"] for doc in manifest if doc.get("doc_id")]
    response = client.chat_completions(
        messages=[{"role": "user", "content": query}],
        doc_id=doc_ids,
        enable_citations=True,
    )
    choices = response.get("choices", [])
    content = ""
    if choices:
        content = choices[0].get("message", {}).get("content", "")
    citations = response.get("citations", [])
    citation_text = "; ".join(
        f"{item.get('document', 'unknown')} p.{item.get('page', '?')}"
        for item in citations
    )
    metadata = {
        "doc_ids": doc_ids,
        "citations": citations,
        "mode": "pageindex_chat_completions",
    }
    if citation_text:
        metadata["citation_text"] = citation_text
    return [{
        "content": content,
        "score": 1.0,
        "metadata": metadata,
        "source": "pageindex",
    }][:top_k] if content else []


def _poll_retrieval(client, retrieval_id: str) -> dict:
    last_payload = {}
    for _ in range(PAGEINDEX_POLL_ATTEMPTS):
        last_payload = client.get_retrieval(retrieval_id)
        status = str(last_payload.get("status", "")).lower()
        if status in {"completed", "complete", "succeeded", "success", "done"}:
            return last_payload
        if last_payload.get("results") or last_payload.get("retrieval_result"):
            return last_payload
        time.sleep(PAGEINDEX_POLL_SECONDS)
    return last_payload


def _extract_retrieval_items(payload: Any) -> list[dict]:
    """Normalize several possible PageIndex retrieval response shapes."""
    candidates = []
    if isinstance(payload, dict):
        for key in ["results", "retrieval_result", "result", "chunks", "nodes", "data"]:
            value = payload.get(key)
            if isinstance(value, list):
                candidates.extend(value)
            elif isinstance(value, dict):
                candidates.extend(_extract_retrieval_items(value))
        if not candidates and any(k in payload for k in ["text", "content", "markdown"]):
            candidates.append(payload)
    elif isinstance(payload, list):
        candidates.extend(payload)

    normalized = []
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            normalized.append({"content": str(item), "score": 1.0 / (index + 1), "metadata": {}})
            continue
        content = (
            item.get("content")
            or item.get("text")
            or item.get("markdown")
            or item.get("page_content")
            or item.get("node_text")
            or ""
        )
        score = item.get("score") or item.get("relevance_score") or item.get("rank_score") or 1.0 / (index + 1)
        normalized.append({
            "content": str(content),
            "score": float(score),
            "metadata": item.get("metadata", {}),
        })
    return normalized


def _pageindex_search_local(query: str, top_k: int) -> list[dict]:
    query_terms = set(tokenize(query))
    chunks = chunk_documents(load_documents())
    scored = []
    for chunk in chunks:
        terms = set(tokenize(chunk["content"]))
        if not terms:
            continue
        overlap = len(query_terms & terms)
        score = overlap / max(len(query_terms), 1)
        if score <= 0:
            continue
        scored.append({
            "content": chunk["content"],
            "score": float(score),
            "metadata": chunk.get("metadata", {}),
            "source": "pageindex",
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    if scored:
        return scored[:top_k]

    return [
        {
            "content": chunk["content"],
            "score": 0.0,
            "metadata": chunk.get("metadata", {}),
            "source": "pageindex",
        }
        for chunk in chunks[:top_k]
    ]


def _pageindex_client():
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY is missing in .env")
    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _manifest_from_remote_documents(client) -> list[dict]:
    """Build manifest from already uploaded PageIndex documents."""
    response = client.list_documents(limit=100)
    docs = response.get("documents", [])
    manifest = []
    for doc in docs:
        name = doc.get("name", "")
        local_path = LEGAL_PDF_DIR / name
        manifest.append({
            "doc_id": doc.get("id"),
            "filename": name,
            "path": str(local_path.relative_to(PROJECT_DIR)) if local_path.exists() else name,
            "type": "legal_pdf",
            "size_bytes": local_path.stat().st_size if local_path.exists() else None,
            "uploaded_at": doc.get("createdAt"),
            "status": doc.get("status"),
            "page_num": doc.get("pageNum"),
            "description": doc.get("description"),
        })
    return [item for item in manifest if item.get("doc_id")]


def _load_manifest() -> list[dict]:
    if not PAGEINDEX_MANIFEST.exists():
        return []
    return json.loads(PAGEINDEX_MANIFEST.read_text(encoding="utf-8"))


def _save_manifest(items: list[dict]) -> None:
    PAGEINDEX_MANIFEST.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    print("Uploading/checking PageIndex documents...")
    docs = upload_documents()
    print(f"Manifest documents: {len(docs)}")

    print("\nTest query:")
    for result in pageindex_search("hinh phat ma tuy", top_k=3):
        print(f"[{result['score']:.3f}] [{result['source']}] {result['content'][:100]}...")
