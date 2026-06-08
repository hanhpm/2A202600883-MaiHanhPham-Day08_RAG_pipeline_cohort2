"""Upload ingestion and hybrid retrieval for the group RAG chatbot.

The preferred storage path is PostgreSQL -> pgvector for dense retrieval, plus
Elasticsearch for lexical BM25. A small JSON store is kept as a local fallback so
the demo still runs when Docker services are not started.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from src.task4_chunking_indexing import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    chunk_documents,
    cosine_similarity,
    embed_texts,
    hashed_embedding,
)
from src.task6_lexical_search import build_bm25_index, tokenize
from src.task7_reranking import rerank_rrf

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parents[1]
GROUP_DIR = PROJECT_DIR / "group_project"
UPLOAD_DIR = GROUP_DIR / "uploads"
LOCAL_STORE_FILE = GROUP_DIR / "uploaded_vector_store.json"

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ELASTICSEARCH_UPLOAD_INDEX = os.getenv("ELASTICSEARCH_UPLOAD_INDEX", "drug_law_uploads")
JINA_API_KEY = os.getenv("JINA_API_KEY", "").strip()
JINA_EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v3")
PGVECTOR_DIM = int(os.getenv("PGVECTOR_DIM") or ("1024" if JINA_API_KEY else str(EMBEDDING_DIM)))


def ingest_upload(filename: str, content: bytes, doc_type: str = "upload") -> dict:
    """Parse, chunk, embed, and persist an uploaded document."""
    started = time.perf_counter()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(filename)
    saved_path = UPLOAD_DIR / safe_name
    saved_path.write_bytes(content)

    text = extract_text(saved_path, content)
    if not text.strip():
        return {"ok": False, "message": "File không có nội dung text đọc được."}

    document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{safe_name}:{hashlib.sha256(content).hexdigest()}"))
    chunks = chunk_documents([
        {
            "content": text,
            "metadata": {
                "document_id": document_id,
                "source": safe_name,
                "path": str(saved_path.relative_to(PROJECT_DIR)),
                "type": doc_type,
                "uploaded": True,
            },
        }
    ])
    embeddings = embed_texts_prefer_jina([chunk["content"] for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = _fit_dim(embedding)

    pg_status = persist_postgres(document_id, safe_name, doc_type, chunks)
    local_count = persist_local(document_id, safe_name, doc_type, chunks)
    es_status = sync_elasticsearch(chunks)

    return {
        "ok": True,
        "document_id": document_id,
        "filename": safe_name,
        "chunks": len(chunks),
        "postgres": pg_status,
        "elasticsearch": es_status,
        "local_fallback_chunks": local_count,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def search_uploaded_documents(query: str, top_k: int = 5) -> list[dict]:
    """Hybrid retrieval over uploaded documents."""
    dense = pgvector_search(query, top_k=top_k * 2)
    if not dense:
        dense = local_vector_search(query, top_k=top_k * 2)

    lexical = elasticsearch_upload_search(query, top_k=top_k * 2)
    if not lexical:
        lexical = local_bm25_search(query, top_k=top_k * 2)

    for item in dense:
        item["retrieval_stage"] = item.get("retrieval_stage", "pgvector_dense")
    for item in lexical:
        item["retrieval_stage"] = item.get("retrieval_stage", "elasticsearch_bm25")

    fused = rerank_rrf([dense, lexical], top_k=top_k)
    for item in fused:
        item["metadata"] = {**item.get("metadata", {}), "uploaded": True}
        item["source"] = "uploaded_hybrid"
    return fused


def storage_status() -> dict:
    """Return backend readiness for the sidebar."""
    return {
        "postgres_pgvector": check_postgres(),
        "elasticsearch": check_elasticsearch(),
        "local_upload_chunks": len(load_local_store()),
        "embedding_model": "jina" if JINA_API_KEY else EMBEDDING_MODEL,
    }


def extract_text(path: Path, content: bytes) -> str:
    """Extract text from common upload types."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".csv", ".json"}:
        return content.decode("utf-8", errors="ignore")

    try:
        from markitdown import MarkItDown

        result = MarkItDown().convert(str(path))
        return getattr(result, "text_content", "") or ""
    except Exception:
        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as temp:
            temp.write(content)
            temp.flush()
        return content.decode("utf-8", errors="ignore")


def embed_texts_prefer_jina(texts: list[str]) -> list[list[float]]:
    """Use Jina embeddings when configured, then fall back to Task 4 embeddings."""
    if not texts:
        return []
    if JINA_API_KEY:
        try:
            response = requests.post(
                "https://api.jina.ai/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {JINA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": JINA_EMBEDDING_MODEL, "input": texts},
                timeout=60,
            )
            response.raise_for_status()
            vectors = [item["embedding"] for item in response.json().get("data", [])]
            if len(vectors) == len(texts):
                return [[float(x) for x in vector] for vector in vectors]
        except Exception:
            pass

    vectors = embed_texts(texts)
    if vectors:
        return vectors
    return [hashed_embedding(text) for text in texts]


def persist_postgres(document_id: str, filename: str, doc_type: str, chunks: list[dict]) -> dict:
    """Persist chunks in PostgreSQL with pgvector when available."""
    try:
        import psycopg
        from pgvector.psycopg import register_vector

        with psycopg.connect(DATABASE_URL) as conn:
            register_vector(conn)
            ensure_postgres_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into rag_documents (id, filename, doc_type)
                    values (%s, %s, %s)
                    on conflict (id) do update set filename = excluded.filename, doc_type = excluded.doc_type
                    """,
                    (document_id, filename, doc_type),
                )
                cur.execute("delete from rag_chunks where document_id = %s", (document_id,))
                for i, chunk in enumerate(chunks):
                    metadata = chunk.get("metadata", {})
                    cur.execute(
                        """
                        insert into rag_chunks (id, document_id, chunk_index, content, metadata, embedding)
                        values (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{i}")),
                            document_id,
                            i,
                            chunk["content"],
                            json.dumps(metadata, ensure_ascii=False),
                            _fit_dim(chunk.get("embedding", [])),
                        ),
                    )
            conn.commit()
        return {"ok": True, "chunks": len(chunks)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def ensure_postgres_schema(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute("create extension if not exists vector")
        cur.execute(
            """
            create table if not exists rag_documents (
                id uuid primary key,
                filename text not null,
                doc_type text not null default 'upload',
                created_at timestamptz not null default now()
            )
            """
        )
        cur.execute(
            f"""
            create table if not exists rag_chunks (
                id uuid primary key,
                document_id uuid references rag_documents(id) on delete cascade,
                chunk_index integer not null,
                content text not null,
                metadata jsonb not null default '{{}}'::jsonb,
                embedding vector({PGVECTOR_DIM}) not null,
                created_at timestamptz not null default now()
            )
            """
        )
        cur.execute(
            "create index if not exists rag_chunks_embedding_idx on rag_chunks using ivfflat (embedding vector_cosine_ops)"
        )
        cur.execute("create index if not exists rag_chunks_metadata_idx on rag_chunks using gin (metadata)")


def pgvector_search(query: str, top_k: int = 10) -> list[dict]:
    try:
        import psycopg
        from pgvector.psycopg import register_vector

        query_embedding = _fit_dim(embed_texts_prefer_jina([query])[0])
        with psycopg.connect(DATABASE_URL) as conn:
            register_vector(conn)
            ensure_postgres_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select content, metadata, 1 - (embedding <=> %s::vector) as score
                    from rag_chunks
                    order by embedding <=> %s::vector
                    limit %s
                    """,
                    (query_embedding, query_embedding, top_k),
                )
                rows = cur.fetchall()
        return [
            {
                "content": row[0],
                "metadata": _json_dict(row[1]),
                "score": float(row[2] or 0.0),
                "method": "pgvector",
            }
            for row in rows
        ]
    except Exception:
        return []


def sync_elasticsearch(chunks: list[dict]) -> dict:
    try:
        base_url = ELASTICSEARCH_URL.rstrip("/")
        health = requests.get(base_url, timeout=5)
        health.raise_for_status()
        exists = requests.head(f"{base_url}/{ELASTICSEARCH_UPLOAD_INDEX}", timeout=5)
        if exists.status_code == 404:
            response = requests.put(
                f"{base_url}/{ELASTICSEARCH_UPLOAD_INDEX}",
                json={
                    "mappings": {
                        "properties": {
                            "content": {"type": "text"},
                            "source": {"type": "keyword"},
                            "type": {"type": "keyword"},
                            "metadata": {"type": "object", "enabled": True},
                        }
                    }
                },
                timeout=10,
            )
            response.raise_for_status()

        bulk_lines = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            doc_id = _chunk_id(metadata, chunk.get("content", ""))
            bulk_lines.append(json.dumps({"index": {"_index": ELASTICSEARCH_UPLOAD_INDEX, "_id": doc_id}}))
            bulk_lines.append(json.dumps({
                "content": chunk.get("content", ""),
                "source": metadata.get("source", ""),
                "type": metadata.get("type", ""),
                "metadata": metadata,
            }, ensure_ascii=False))
        if bulk_lines:
            response = requests.post(
                f"{base_url}/_bulk",
                data="\n".join(bulk_lines) + "\n",
                headers={"Content-Type": "application/x-ndjson"},
                timeout=60,
            )
            response.raise_for_status()
        return {"ok": True, "chunks": len(chunks)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def elasticsearch_upload_search(query: str, top_k: int = 10) -> list[dict]:
    try:
        response = requests.post(
            f"{ELASTICSEARCH_URL.rstrip('/')}/{ELASTICSEARCH_UPLOAD_INDEX}/_search",
            json={
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content^2", "source", "type"],
                    }
                },
            },
            timeout=5,
        )
        response.raise_for_status()
        hits = response.json().get("hits", {}).get("hits", [])
        return [
            {
                "content": hit.get("_source", {}).get("content", ""),
                "metadata": hit.get("_source", {}).get("metadata", {}),
                "score": float(hit.get("_score", 0.0)),
                "method": "elasticsearch_bm25",
            }
            for hit in hits
        ]
    except Exception:
        return []


def persist_local(document_id: str, filename: str, doc_type: str, chunks: list[dict]) -> int:
    LOCAL_STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = [item for item in load_local_store() if item.get("document_id") != document_id]
    for i, chunk in enumerate(chunks):
        metadata = {**chunk.get("metadata", {}), "source": filename, "type": doc_type, "uploaded": True}
        existing.append({
            "id": _chunk_id(metadata, chunk.get("content", "")),
            "document_id": document_id,
            "content": chunk.get("content", ""),
            "metadata": metadata,
            "embedding": _fit_dim(chunk.get("embedding", [])),
        })
    LOCAL_STORE_FILE.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(existing)


def load_local_store() -> list[dict]:
    if not LOCAL_STORE_FILE.exists():
        return []
    try:
        return json.loads(LOCAL_STORE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def local_vector_search(query: str, top_k: int = 10) -> list[dict]:
    corpus = load_local_store()
    if not corpus:
        return []
    query_embedding = _fit_dim(embed_texts_prefer_jina([query])[0])
    results = []
    for item in corpus:
        score = cosine_similarity(query_embedding, item.get("embedding", []))
        if score > 0:
            results.append({
                "content": item["content"],
                "metadata": item.get("metadata", {}),
                "score": float(score),
                "method": "local_vector_fallback",
            })
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def local_bm25_search(query: str, top_k: int = 10) -> list[dict]:
    corpus = load_local_store()
    if not corpus:
        return []
    bm25 = build_bm25_index(corpus)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda pair: float(pair[1]), reverse=True)
    results = []
    for idx, score in ranked[:top_k]:
        if float(score) <= 0:
            continue
        item = corpus[idx]
        results.append({
            "content": item["content"],
            "metadata": item.get("metadata", {}),
            "score": float(score),
            "method": "local_bm25_fallback",
        })
    return results


def check_postgres() -> dict:
    try:
        import psycopg

        with psycopg.connect(DATABASE_URL, connect_timeout=3) as conn:
            ensure_postgres_schema(conn)
            with conn.cursor() as cur:
                cur.execute("select count(*) from rag_chunks")
                count = cur.fetchone()[0]
        return {"ok": True, "chunks": int(count)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def check_elasticsearch() -> dict:
    try:
        response = requests.get(ELASTICSEARCH_URL.rstrip("/"), timeout=3)
        response.raise_for_status()
        return {"ok": True, "url": ELASTICSEARCH_URL}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _safe_filename(filename: str) -> str:
    stem = Path(filename).stem or "upload"
    suffix = Path(filename).suffix.lower()
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in stem).strip("-")
    return f"{clean[:80] or 'upload'}{suffix}"


def _fit_dim(vector: list[float]) -> list[float]:
    values = [float(x) for x in vector]
    if len(values) == PGVECTOR_DIM:
        return values
    if len(values) > PGVECTOR_DIM:
        return values[:PGVECTOR_DIM]
    return values + [0.0] * (PGVECTOR_DIM - len(values))


def _chunk_id(metadata: dict, content: str) -> str:
    raw = f"{metadata.get('document_id', '')}:{metadata.get('source', '')}:{metadata.get('chunk_index', '')}:{content[:120]}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def _json_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return {}
