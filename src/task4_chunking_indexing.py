"""
Task 4 - Chunking & Indexing.

This implementation indexes all markdown files from data/standardized into a
local JSON vector store. It keeps the assignment choices explicit while staying
usable offline for tests and demos.
"""

import hashlib
import json
import math
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"

# Recursive chunking is robust for mixed legal/news markdown where headings are
# not always consistent. 500 chars keeps chunks focused; 50 overlap preserves
# short legal clauses that may cross a boundary.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Choose exactly one embedding model by uncommenting one line below.
# MiniLM: 384 dimensions, light and fast.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# BGE-M3: 1024 dimensions, multilingual and strong for Vietnamese.
# DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
# OpenAI: 1536 dimensions, uses OPENAI_API_KEY from .env.
# DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# You can also override without editing the file:
#   $env:EMBEDDING_MODEL="BAAI/bge-m3"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

EMBEDDING_DIMS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-m3": 1024,
    "text-embedding-3-small": 1536,
}
EMBEDDING_DIM = EMBEDDING_DIMS.get(EMBEDDING_MODEL, 1024)
MODEL_INDEX_NAMES = {
    "sentence-transformers/all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "BAAI/bge-m3": "bge-m3",
    "text-embedding-3-small": "text-embedding-3-small",
}
INDEX_BASENAME = MODEL_INDEX_NAMES.get(
    EMBEDDING_MODEL,
    re.sub(r"[^A-Za-z0-9_.-]+", "-", EMBEDDING_MODEL).strip("-"),
)
INDEX_FILE = INDEX_DIR / f"{INDEX_BASENAME}_vector_index.json"
INDEX_META_FILE = INDEX_DIR / f"{INDEX_BASENAME}_vector_index.meta.json"

# Local JSON vector store is chosen for this individual task so no Weaviate
# server is required. It stores content, metadata and vectors in data/index/.
VECTOR_STORE = "local_json"


def load_documents() -> list[dict]:
    """Read all markdown files from data/standardized."""
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue

        lower_path = str(md_file).lower()
        name = md_file.name.lower()
        if "legal" in lower_path or "luat" in name or "nghi-dinh" in name:
            doc_type = "legal"
        elif "news" in lower_path or "bao" in name or "article" in name:
            doc_type = "news"
        else:
            doc_type = md_file.parent.name if md_file.parent != STANDARDIZED_DIR else "unknown"

        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "path": str(md_file.relative_to(STANDARDIZED_DIR)),
                "type": doc_type,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents into overlapping chunks."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        split_text = splitter.split_text
    except Exception:
        split_text = _split_text_fallback

    chunks = []
    for doc in documents:
        for i, chunk_text in enumerate(split_text(doc["content"])):
            text = chunk_text.strip()
            if not text:
                continue
            chunks.append({
                "content": text,
                "metadata": {**doc.get("metadata", {}), "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Add an embedding vector to every chunk."""
    embeddings = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """Persist chunks into the local JSON vector store."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    serializable = [
        {
            "content": c["content"],
            "metadata": c.get("metadata", {}),
            "embedding": c.get("embedding") or hashed_embedding(c["content"]),
        }
        for c in chunks
    ]
    INDEX_FILE.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    INDEX_META_FILE.write_text(
        json.dumps(
            {
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dim": len(serializable[0]["embedding"]) if serializable else EMBEDDING_DIM,
                "vector_store": VECTOR_STORE,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return INDEX_FILE


def ensure_index() -> list[dict]:
    """Load the index, building it from markdown files when needed."""
    if INDEX_FILE.exists() and _index_matches_config():
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    docs = load_documents()
    chunks = embed_chunks(chunk_documents(docs))
    index_to_vectorstore(chunks)
    return chunks


def _index_matches_config() -> bool:
    if not INDEX_META_FILE.exists():
        return False
    try:
        metadata = json.loads(INDEX_META_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return metadata.get("embedding_model") == EMBEDDING_MODEL


def embed_query(text: str) -> list[float]:
    """Embed a query with the same model preference used for indexing."""
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts using the selected EMBEDDING_MODEL, with local fallback."""
    if not texts:
        return []

    if EMBEDDING_MODEL.startswith("text-embedding"):
        openai_embeddings = embed_texts_openai(texts)
        if openai_embeddings:
            return openai_embeddings
        return [hashed_embedding(text) for text in texts]

    st_embeddings = embed_texts_sentence_transformers(texts)
    if st_embeddings:
        return st_embeddings
    return [hashed_embedding(text) for text in texts]


def embed_texts_sentence_transformers(texts: list[str]) -> list[list[float]]:
    """Create embeddings for MiniLM/BGE-M3 via sentence-transformers."""
    if not EMBEDDING_MODEL.startswith(("sentence-transformers/", "BAAI/")):
        return []

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return [[float(x) for x in emb] for emb in embeddings]
    except Exception:
        return []


def embed_texts_openai(texts: list[str]) -> list[list[float]]:
    """Create OpenAI embeddings when OPENAI_API_KEY is set."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = EMBEDDING_MODEL
    if not api_key or not model.startswith("text-embedding") or not texts:
        return []

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        vectors: list[list[float]] = []
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = client.embeddings.create(model=model, input=batch)
            vectors.extend([list(item.embedding) for item in response.data])
        return vectors
    except Exception:
        return []


def tokenize(text: str) -> list[str]:
    """Simple Unicode tokenizer for Vietnamese-friendly keyword matching."""
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def hashed_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Deterministic normalized hashed embedding fallback."""
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity for normalized vectors."""
    if not a or not b:
        return 0.0
    limit = min(len(a), len(b))
    return float(sum(a[i] * b[i] for i in range(limit)))


def _split_text_fallback(text: str) -> list[str]:
    """Small fallback splitter used when langchain-text-splitters is missing."""
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks


def run_pipeline():
    """Run load -> chunk -> embed -> index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")
    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")
    index_path = index_to_vectorstore(chunks)
    print(f"Indexed to {index_path}")


if __name__ == "__main__":
    run_pipeline()
