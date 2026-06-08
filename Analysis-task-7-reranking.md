# Analysis - Task 7 Reranking

## 1. README Requirement

Task 7 asks for a reranking module that re-scores retrieval candidates and
returns the most relevant chunks first.

Required interface:

```python
def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Re-score and re-order candidates based on relevance to query.
    """
```

README lists four possible choices:

| Method | Library / Model | Feature |
|---|---|---|
| Cross-encoder reranker | `jinaai/jina-reranker-v2-base-multilingual` | Multilingual, good for Vietnamese |
| Cross-encoder reranker | `Qwen/Qwen3-Reranker-0.6B` | Lightweight, effective |
| MMR | Self implemented | Reduce duplication, improve diversity |
| RRF | Self implemented | Fuse results from multiple rankers |

The code in `src/task7_reranking.py` now supports all four options for
comparison.

## 2. Implemented Methods

### 2.1 Local Cross-Encoder Fallback

Function:

```python
rerank_cross_encoder(query, candidates, top_k)
```

This is the default safe method used by tests and Task 9. It approximates
cross-encoder reranking with:

- token overlap between query and chunk content
- incoming retrieval score

It is not as strong as a real cross-encoder, but it is deterministic, offline,
and does not require API keys.

Command:

```powershell
$env:RERANK_METHOD="cross_encoder"
python -m src.task7_reranking
```

### 2.2 Jina Reranker API

Function:

```python
rerank_jina_api(query, candidates, top_k)
```

Model:

```text
jinaai/jina-reranker-v2-base-multilingual
```

Mechanism:

- Sends query and candidate documents to Jina's rerank endpoint.
- Jina cross-encoder scores each query-document pair directly.
- Results are returned by relevance score.

Environment:

```powershell
$env:JINA_API_KEY="..."
$env:RERANK_METHOD="jina_api"
python -m src.task7_reranking
```

Status:

- Implemented as a real API call.
- If `JINA_API_KEY` is missing, fallback is used when
  `ALLOW_RERANK_FALLBACK=true`.
- To force strict real API behavior:

```powershell
$env:ALLOW_RERANK_FALLBACK="false"
```

### 2.3 Qwen Local Reranker

Function:

```python
rerank_qwen_local(query, candidates, top_k)
```

Model:

```text
Qwen/Qwen3-Reranker-0.6B
```

Mechanism:

- Loads model/tokenizer with `transformers`.
- Builds query-document pairs.
- Scores each pair using model logits.
- Sorts candidates by rerank score.

Environment:

```powershell
$env:RERANK_METHOD="qwen_local"
python -m src.task7_reranking
```

Status:

- Implemented as optional local model loading.
- Requires `transformers`, `torch`, enough memory, and model availability.
- Falls back to local cross-encoder approximation if unavailable and
  `ALLOW_RERANK_FALLBACK=true`.

### 2.4 MMR

Function:

```python
rerank_mmr(query_embedding, candidates, top_k, lambda_param=0.7)
```

Mechanism:

```text
MMR = lambda * relevance(query, doc)
      - (1 - lambda) * max_similarity(doc, already_selected_docs)
```

Meaning:

- High relevance keeps candidates related to the query.
- Diversity penalty reduces repeated chunks.
- Useful when many retrieved chunks are near-duplicates.

Command:

```powershell
$env:RERANK_METHOD="mmr"
python -m src.task7_reranking
```

### 2.5 RRF

Function:

```python
rerank_rrf(ranked_lists, top_k, k=60)
```

Mechanism:

```text
RRF(document) = sum(1 / (k + rank_in_each_list))
```

Meaning:

- Combines rankings from multiple retrievers.
- A document appearing high in many lists gets a high fused score.
- Useful for hybrid retrieval: semantic results + lexical BM25 results.

Command:

```powershell
$env:RERANK_METHOD="rrf"
python -m src.task7_reranking
```

## 3. Comparison Function

The module includes:

```python
compare_rerank_methods(query, candidates, top_k=3)
```

It runs:

- `cross_encoder`
- `jina_api`
- `qwen_local`
- `mmr`
- `rrf`

This is useful for demo comparison. If Jina/Qwen are unavailable, the result
still shows their fallback behavior or an unavailable message depending on
`ALLOW_RERANK_FALLBACK`.

## 4. Recommended Demo Commands

Run the Task 7 demo:

```powershell
python -m src.task7_reranking
```

Run tests:

```powershell
pytest tests/test_individual.py::TestTask7 -v
```

Run Task 9, where Task 7 is used in the complete retrieval pipeline:

```powershell
python -m src.task9_retrieval_pipeline
```

## 5. Practical Recommendation

For this project:

- Use `cross_encoder` as the stable default for offline tests.
- Use `jina_api` if a Jina API key is available and the demo needs strong
  multilingual reranking.
- Use `qwen_local` only if the machine can load the local reranker model.
- Use `MMR` when retrieved chunks are repetitive.
- Use `RRF` when combining semantic search and lexical search rankings.

In Task 9, RRF is especially useful for merging Task 5 semantic search with Task
6 lexical search before reranking.

## 6. Verified Run Output

Current configuration:

- `JINA_API_KEY` is loaded from `.env`.
- Jina API model sent to endpoint: `jina-reranker-v2-base-multilingual`.
- Qwen local model path:
  `models/rerankings/Qwen3-Reranker-0.6B`.
- Qwen is loaded with `sentence_transformers.CrossEncoder`.

Command:

```powershell
python -m src.task7_reranking
```

Output:

```text
Method: cross_encoder
[0.713] [local_cross_encoder_fallback] Dieu 248 toi tang tru trai phep chat ma tuy
[0.462] [local_cross_encoder_fallback] Nghe si bi bat vi su dung ma tuy

Method: jina_api
[0.682] [jina_api] Dieu 248 toi tang tru trai phep chat ma tuy
[0.391] [jina_api] Nghe si bi bat vi su dung ma tuy

Method: qwen_local
[7.312] [qwen_local] Dieu 248 toi tang tru trai phep chat ma tuy
[2.375] [qwen_local] Nghe si bi bat vi su dung ma tuy

Method: mmr
[0.053] [mmr] Dieu 248 toi tang tru trai phep chat ma tuy
[-0.014] [mmr] Python programming tutorial

Method: rrf
[0.016] [rrf] Dieu 248 toi tang tru trai phep chat ma tuy
[0.016] [rrf] Nghe si bi bat vi su dung ma tuy
```

Interpretation:

- `jina_api` is now a real Jina API result, not fallback.
- `qwen_local` is now a real local Qwen reranker result, not fallback.
- `cross_encoder` is still the offline local fallback scorer.
- `mmr` and `rrf` are self-implemented local algorithms.
