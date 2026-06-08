# Group Project - RAG Chatbot

## Project Summary

Đây là sản phẩm RAG Chatbot dạng solo cá nhân cho chủ đề pháp luật ma túy và tin tức liên quan. Mục tiêu là trả lời chính xác, có citation, hiển thị source documents đã dùng, hỗ trợ follow-up questions bằng conversation memory, và cho phép upload thêm tài liệu để embedding xuống database.

Sản phẩm chọn **Yêu cầu 1: Sản phẩm nhóm RAG Chatbot**. Phần evaluation vẫn được giữ trong `group_project/evaluation` để chứng minh chất lượng retrieval/generation.

## Mục Tiêu Độ Chính Xác

- Chỉ trả lời dựa trên context retrieve được.
- Mỗi ý quan trọng có citation dạng `[S1]`, `[S2]`.
- Retrieval dùng hybrid strategy:
  - Dense search: PostgreSQL + pgvector cho tài liệu upload, Task 5 semantic index cho corpus có sẵn.
  - Lexical search: Elasticsearch BM25 từ bản đã giải nén trong `tool/` hoặc Docker.
  - Reranking: Task 7 reranker để tăng precision top results.
- Nếu context không đủ, chatbot nói rõ chưa thể xác minh thay vì đoán.

## Kiến Trúc Hệ Thống

```mermaid
flowchart TD
    U["User"] --> UI["Streamlit Chat UI"]
    UI --> Upload["Upload source files"]
    Upload --> Chunk["Task 4 Chunking"]
    Chunk --> Embed["Jina/OpenAI/local Embedding"]
    Embed --> PG["PostgreSQL + pgvector"]
    Chunk --> ES["Elasticsearch BM25"]
    Chunk --> Local["Local JSON fallback"]

    UI --> Query["Question + conversation memory"]
    Query --> Retrieval["Hybrid Retrieval Service"]
    Retrieval --> T5["Task 5 Semantic Search"]
    Retrieval --> T6["Task 6 BM25 / Elasticsearch"]
    Retrieval --> PG
    Retrieval --> ES
    Retrieval --> Fusion["RRF fusion"]
    Fusion --> Rerank["Task 7 Reranking"]
    Rerank --> Gen["Task 10 Generation with citation"]
    Gen --> UI
```

## Stack

| Layer | Công nghệ |
|---|---|
| UI | Streamlit, layout giống ChatGPT + NotebookLM |
| Upload parsing | MarkItDown + text fallback |
| Chunking | Task 4 recursive chunking |
| Embedding | Jina embeddings nếu có `JINA_API_KEY`, fallback OpenAI/local |
| Dense DB | PostgreSQL + pgvector |
| Lexical retrieval | Elasticsearch BM25 |
| Hybrid retrieval | Semantic + BM25 + uploaded docs, RRF fusion |
| Generation | OpenAI Chat Completion, citation theo source |
| Logging | `group_project/logs/search_logs.jsonl`, `search_logs.csv` |

## Main Files

| File | Vai trò |
|---|---|
| `group_project/app.py` | Chat UI, upload, source panel, retrieval comparison |
| `group_project/document_store.py` | Upload ingestion, PostgreSQL/pgvector, Elasticsearch sync, local fallback |
| `group_project/search_service.py` | Hybrid retrieval, reranking, chat answer with citation, logging |
| `group_project/run_elasticsearch_local.ps1` | Chạy Elasticsearch đã giải nén trong `tool/` |
| `group_project/docker-compose.yml` | PostgreSQL + pgvector và Elasticsearch Docker |
| `group_project/evaluation/golden_dataset.json` | Golden dataset 15+ Q&A |
| `group_project/evaluation/eval_pipeline.py` | Script evaluation |
| `group_project/evaluation/results.md` | Bảng điểm và phân tích |

## Hướng Dẫn Chạy

Từ root repository:

```powershell
pip install -r requirements.txt
```

Chạy Elasticsearch đã giải nén trong `tool/`:

```powershell
.\group_project\run_elasticsearch_local.ps1
```

Hoặc chạy bằng Docker:

```powershell
docker compose -f group_project/docker-compose.yml up -d elasticsearch
```

Chạy PostgreSQL + pgvector bằng Docker:

```powershell
docker compose -f group_project/docker-compose.yml up -d postgres
```

Tạo file `.env` từ `.env.example`, tối thiểu nên có:

```text
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4o-mini
JINA_API_KEY=jina_xxx
DATABASE_URL=postgresql://rag:rag@localhost:5432/rag
PGVECTOR_DIM=1024
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_UPLOAD_INDEX=drug_law_uploads
ALLOW_EXTERNAL_FALLBACK=true
```

Nếu dùng embedding local mặc định `sentence-transformers/all-MiniLM-L6-v2`, đổi `PGVECTOR_DIM=384`.

Chạy chatbot:

```powershell
streamlit run group_project/app.py
```

## Cách Demo

1. Mở app Streamlit.
2. Upload tài liệu luật hoặc tin tức ở sidebar **Library**.
3. Bấm **Embed sources** để chunk, embed và lưu vào DB.
4. Hỏi câu bất kỳ ở khung chat.
5. Kiểm tra citations trong câu trả lời và source cards ở panel bên phải.
6. Mở **Retrieval comparison** để xem semantic, BM25, uploaded docs và reranked output.

Ví dụ câu hỏi:

```text
Điều 249 Bộ luật Hình sự quy định gì về tàng trữ trái phép chất ma túy?
Luật Phòng chống ma túy 2021 quy định hành vi nào bị nghiêm cấm?
Cai nghiện ma túy bắt buộc được quy định như thế nào?
Chi Dân liên quan đến vụ việc ma túy như thế nào?
Danh mục chất ma túy và tiền chất được sửa đổi trong nghị định nào?
```

## Evaluation Pipeline

- `group_project/evaluation/golden_dataset.json`: 15+ cặp Q&A.
- `group_project/evaluation/eval_pipeline.py`: chạy evaluation.
- `group_project/evaluation/results.md`: bảng điểm, worst performers, đề xuất cải tiến.
- A/B config: hybrid + reranking so với baseline không reranking hoặc dense-only.

Chạy:

```powershell
python -m group_project.evaluation.eval_pipeline
```

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|---|---|---|---|
| Mai Hạnh Phạm | 2A202600883 | Thu thập dữ liệu pháp luật và tin tức, chuẩn hóa markdown | Done |
| Mai Hạnh Phạm | 2A202600883 | Chunking, indexing, semantic search, lexical BM25 | Done |
| Mai Hạnh Phạm | 2A202600883 | Reranking, retrieval pipeline, generation with citation | Done |
| Mai Hạnh Phạm | 2A202600883 | Streamlit chatbot, upload file, pgvector, Elasticsearch | Done |
| Mai Hạnh Phạm | 2A202600883 | Evaluation dataset, pipeline, report | Done locally |

## Ghi Chú

Repo này có thể phát triển tiếp ở track 3 giai đoạn 2 bằng knowledge graph để xử lý các câu hỏi khó, nhiều bước suy luận hoặc cần liên kết thực thể pháp lý phức tạp.
