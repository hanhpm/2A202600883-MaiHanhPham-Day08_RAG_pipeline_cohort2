"""
Task 6 - Lexical Search Module.

Default method is BM25 as required by README. Bonus lexical methods are included
for demo/analysis: TF-IDF, Elasticsearch BM25, and Weaviate BM25 built-in.
"""

import math
import os
import sys
import uuid
from collections import Counter

import requests

from .task4_chunking_indexing import ensure_index, tokenize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CORPUS: list[dict] = []
DEFAULT_LEXICAL_METHOD = os.getenv("LEXICAL_METHOD", "bm25")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX", "drug_law_docs")
WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION", "DrugLawDocs")
ALLOW_EXTERNAL_FALLBACK = os.getenv("ALLOW_EXTERNAL_FALLBACK", "false").lower() in {"1", "true", "yes"}


def build_bm25_index(corpus: list[dict]):
    """Build a BM25 index from chunk corpus."""
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    try:
        from rank_bm25 import BM25Okapi

        return BM25Okapi(tokenized_corpus)
    except Exception:
        return SimpleBM25(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10, method: str = DEFAULT_LEXICAL_METHOD) -> list[dict]:
    """
    Keyword retrieval using the selected lexical method.

    Returns list[dict] with content, score and metadata sorted descending.
    """
    if top_k <= 0:
        return []

    method = method.lower().replace("-", "_")
    if method in {"bm25", "rank_bm25"}:
        return bm25_search(query, top_k)
    if method in {"tfidf", "tf_idf"}:
        return tfidf_search(query, top_k)
    if method in {"elasticsearch", "elastic"}:
        return elasticsearch_search(query, top_k)
    if method in {"weaviate", "weaviate_bm25"}:
        return weaviate_bm25_search(query, top_k)
    raise ValueError(f"Unknown lexical search method: {method}")


def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    """Keyword retrieval using BM25."""
    global CORPUS
    CORPUS = load_corpus()
    if not CORPUS:
        return []

    bm25 = build_bm25_index(CORPUS)
    scores = bm25.get_scores(tokenize(query))
    return _format_ranked_results(CORPUS, scores, top_k)


def tfidf_search(query: str, top_k: int = 10) -> list[dict]:
    """
    TF-IDF lexical retrieval.

    TF-IDF scores terms by local term frequency multiplied by inverse document
    frequency, then compares query/document vectors with cosine similarity.
    """
    corpus = load_corpus()
    if not corpus:
        return []

    tokenized_docs = [tokenize(doc["content"]) for doc in corpus]
    query_tokens = tokenize(query)
    doc_freq = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))

    total_docs = max(len(tokenized_docs), 1)
    idf = {
        term: math.log((1 + total_docs) / (1 + freq)) + 1
        for term, freq in doc_freq.items()
    }
    query_vector = _tfidf_vector(query_tokens, idf)
    scores = [
        _cosine_dict(query_vector, _tfidf_vector(tokens, idf))
        for tokens in tokenized_docs
    ]
    return _format_ranked_results(corpus, scores, top_k)


def elasticsearch_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Elasticsearch BM25 search.

    Requires ELASTICSEARCH_URL and an indexed ELASTICSEARCH_INDEX. If the service
    is unavailable, falls back to local BM25 so Task 6 can still be demoed.
    """
    try:
        ensure_elasticsearch_index()
        response = requests.post(
            f"{ELASTICSEARCH_URL.rstrip('/')}/{ELASTICSEARCH_INDEX}/_search",
            json={
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content", "source", "type"],
                    }
                },
            },
            timeout=5,
        )
        response.raise_for_status()
        hits = response.json().get("hits", {}).get("hits", [])
        results = []
        for hit in hits:
            source = hit.get("_source", {})
            results.append({
                "content": source.get("content", ""),
                "score": float(hit.get("_score", 0.0)),
                "metadata": source.get("metadata", source),
                "method": "elasticsearch_bm25",
            })
        return results
    except Exception as exc:
        return _external_fallback_or_raise("Elasticsearch BM25", exc, query, top_k)


def weaviate_bm25_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Weaviate BM25 built-in search.

    Requires a running local Weaviate instance and WEAVIATE_COLLECTION. If not
    available, falls back to local BM25.
    """
    client = None
    try:
        import weaviate
        from weaviate.classes.query import MetadataQuery

        ensure_weaviate_collection()
        client = weaviate.connect_to_local()
        collection = client.collections.get(WEAVIATE_COLLECTION)
        response = collection.query.bm25(
            query=query,
            limit=top_k,
            return_metadata=MetadataQuery(score=True),
        )
        results = []
        for obj in response.objects:
            props = obj.properties
            results.append({
                "content": props.get("content", ""),
                "score": float(getattr(obj.metadata, "score", 0.0) or 0.0),
                "metadata": {
                    "source": props.get("source"),
                    "type": props.get("doc_type") or props.get("type"),
                },
                "method": "weaviate_bm25",
            })
        return results
    except Exception as exc:
        return _external_fallback_or_raise("Weaviate BM25", exc, query, top_k)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def ensure_elasticsearch_index(corpus: list[dict] | None = None) -> None:
    """Create and populate the Elasticsearch index when the server is running."""
    corpus = corpus or load_corpus()
    base_url = ELASTICSEARCH_URL.rstrip("/")

    health = requests.get(base_url, timeout=5)
    health.raise_for_status()

    exists = requests.head(f"{base_url}/{ELASTICSEARCH_INDEX}", timeout=5)
    if exists.status_code == 404:
        response = requests.put(
            f"{base_url}/{ELASTICSEARCH_INDEX}",
            json={
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "source": {"type": "keyword"},
                        "type": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                    }
                }
            },
            timeout=10,
        )
        response.raise_for_status()
    elif exists.status_code >= 400:
        exists.raise_for_status()

    count_response = requests.get(f"{base_url}/{ELASTICSEARCH_INDEX}/_count", timeout=5)
    count_response.raise_for_status()
    if count_response.json().get("count", 0) > 0:
        return

    bulk_lines = []
    for doc in corpus:
        metadata = doc.get("metadata", {})
        doc_id = _stable_id(metadata, doc.get("content", ""))
        bulk_lines.append(json_dumps({"index": {"_index": ELASTICSEARCH_INDEX, "_id": doc_id}}))
        bulk_lines.append(json_dumps({
            "content": doc.get("content", ""),
            "source": metadata.get("source", ""),
            "type": metadata.get("type", ""),
            "chunk_index": metadata.get("chunk_index", 0),
            "metadata": metadata,
        }))

    if bulk_lines:
        response = requests.post(
            f"{base_url}/_bulk",
            data="\n".join(bulk_lines) + "\n",
            headers={"Content-Type": "application/x-ndjson"},
            timeout=60,
        )
        response.raise_for_status()


def ensure_weaviate_collection(corpus: list[dict] | None = None) -> None:
    """Create and populate the Weaviate collection when the server is running."""
    import weaviate
    from weaviate.classes.config import Configure, DataType, Property

    corpus = corpus or load_corpus()
    client = weaviate.connect_to_local()
    try:
        if not client.collections.exists(WEAVIATE_COLLECTION):
            client.collections.create(
                name=WEAVIATE_COLLECTION,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="type", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                ],
            )

        collection = client.collections.get(WEAVIATE_COLLECTION)
        aggregate = collection.aggregate.over_all(total_count=True)
        if aggregate.total_count and aggregate.total_count > 0:
            return

        with collection.batch.dynamic() as batch:
            for doc in corpus:
                metadata = doc.get("metadata", {})
                batch.add_object(
                    uuid=_stable_id(metadata, doc.get("content", "")),
                    properties={
                        "content": doc.get("content", ""),
                        "source": metadata.get("source", ""),
                        "type": metadata.get("type", ""),
                        "chunk_index": int(metadata.get("chunk_index", 0) or 0),
                    },
                )
    finally:
        client.close()


def load_corpus() -> list[dict]:
    """Load chunk corpus from Task 4 index."""
    return [
        {"content": item["content"], "metadata": item.get("metadata", {})}
        for item in ensure_index()
    ]


def _external_fallback_or_raise(name: str, exc: Exception, query: str, top_k: int) -> list[dict]:
    """Fallback is opt-in so external methods prove real server connectivity."""
    if not ALLOW_EXTERNAL_FALLBACK:
        raise RuntimeError(
            f"{name} is not available or not indexed. "
            f"Start the server and run the indexing helper first. Original error: {exc}"
        ) from exc
    fallback = bm25_search(query, top_k)
    for item in fallback:
        item["method"] = f"{name.lower().replace(' ', '_')}_fallback_local_bm25"
    return fallback


def _stable_id(metadata: dict, content: str) -> str:
    raw = f"{metadata.get('source', '')}:{metadata.get('chunk_index', '')}:{content[:120]}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def json_dumps(data: dict) -> str:
    import json

    return json.dumps(data, ensure_ascii=False)


def _format_ranked_results(corpus: list[dict], scores, top_k: int) -> list[dict]:
    ranked = sorted(enumerate(scores), key=lambda pair: float(pair[1]), reverse=True)

    results = []
    for idx, score in ranked[:top_k]:
        if float(score) <= 0:
            continue
        results.append({
            "content": corpus[idx]["content"],
            "score": float(score),
            "metadata": corpus[idx]["metadata"],
        })
    return results


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokens)
    total = max(sum(counts.values()), 1)
    return {
        term: (count / total) * idf.get(term, 0.0)
        for term, count in counts.items()
    }


def _cosine_dict(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(term, 0.0) for term, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values())) or 1.0
    norm_b = math.sqrt(sum(value * value for value in b.values())) or 1.0
    return float(dot / (norm_a * norm_b))


class SimpleBM25:
    """Small fallback BM25 scorer for environments without rank-bm25."""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc) for doc in self.docs]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.term_counts = [Counter(doc) for doc in self.docs]
        df = Counter()
        for doc in self.docs:
            df.update(set(doc))
        total_docs = max(len(self.docs), 1)
        self.idf = {
            term: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for counts, dl in zip(self.term_counts, self.doc_lengths):
            score = 0.0
            for term in query_tokens:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                score += self.idf.get(term, 0.0) * (tf * (self.k1 + 1)) / denom
            scores.append(score)
        return scores


if __name__ == "__main__":
    query = "Dieu 248 ma tuy"
    for method_name in ["bm25", "tfidf", "elasticsearch", "weaviate_bm25"]:
        print(f"\nMethod: {method_name}")
        try:
            for result in lexical_search(query, top_k=3, method=method_name):
                print(f"[{result['score']:.3f}] {result['content'][:100]}...")
        except RuntimeError as exc:
            print(f"External search unavailable: {exc}")
