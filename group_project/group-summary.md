# Group Summary - Option A Search Engine

## Requirement Coverage

- Web UI: `group_project/app.py` using Streamlit.
- Hybrid search: semantic search + BM25 + RRF fusion.
- Reranking: Task 7 `cross_encoder` default, optional Jina/Qwen/MMR/RRF.
- Source and relevance score: shown for every result card.
- Comparison: semantic vs BM25 vs hybrid vs reranked.
- Explanation: UI explains why a stage is better/worse.
- Logs: every query is appended to `group_project/logs/search_logs.jsonl` and `.csv`.
- Generation with citation: integrated through Task 10 in the app and tested below.

## Accuracy Summary

- Average hybrid+rerank heuristic score: `0.75`
- Average BM25-only heuristic score: `0.75`

| ID | Better Config | Hybrid Score | BM25 Score | Top Hybrid Source | Top BM25 Source |
|---|---|---:|---:|---|---|
| legal_storage_penalty | hybrid_rerank | 0.750 | 0.750 | 2023-le-hang-dantri.md | 2022-huu-tin-thanh-nien.md |
| drug_prevention_law | hybrid_rerank | 0.500 | 0.500 | nghi-dinh-116-2021-cai-nghien-ma-tuy.md | nghi-dinh-116-2021-cai-nghien-ma-tuy.md |
| compulsory_rehab | hybrid_rerank | 0.750 | 0.750 | nghi-dinh-116-2021-cai-nghien-ma-tuy.md | nghi-dinh-116-2021-cai-nghien-ma-tuy.md |
| artist_news | hybrid_rerank | 0.500 | 0.500 | 2024-chi-dan-an-tay-thanh-nien.md | 2024-chi-dan-an-tay-dantri.md |
| chi_dan_news | hybrid_rerank | 1.000 | 1.000 | 2024-chi-dan-an-tay-thanh-nien.md | 2024-chi-dan-an-tay-thanh-nien.md |
| substance_list | hybrid_rerank | 1.000 | 1.000 | nghi-dinh-90-2024-sua-doi-danh-muc-chat-ma-tuy-tien-chat.md | nghi-dinh-90-2024-sua-doi-danh-muc-chat-ma-tuy-tien-chat.md |

## Scenario Analysis

### legal_storage_penalty

Question: Hình phạt cho tội tàng trữ trái phép chất ma túy là gì?

Why better/worse: Best stage: reranked. It has stronger query-term overlap and/or score. Semantic helps paraphrases, BM25 helps exact legal terms, hybrid+rerank balances both.

### drug_prevention_law

Question: Luật Phòng chống ma túy 2021 quy định hành vi nào bị nghiêm cấm?

Why better/worse: Best stage: lexical_bm25. It has stronger query-term overlap and/or score. Semantic helps paraphrases, BM25 helps exact legal terms, hybrid+rerank balances both.

### compulsory_rehab

Question: Cai nghiện ma túy bắt buộc được quy định như thế nào?

Why better/worse: Best stage: lexical_bm25. It has stronger query-term overlap and/or score. Semantic helps paraphrases, BM25 helps exact legal terms, hybrid+rerank balances both.

### artist_news

Question: Nghệ sĩ nào bị điều tra liên quan đến ma túy?

Why better/worse: Best stage: lexical_bm25. It has stronger query-term overlap and/or score. Semantic helps paraphrases, BM25 helps exact legal terms, hybrid+rerank balances both.

### chi_dan_news

Question: Chi Dân liên quan đến vụ việc ma túy như thế nào?

Why better/worse: Best stage: lexical_bm25. It has stronger query-term overlap and/or score. Semantic helps paraphrases, BM25 helps exact legal terms, hybrid+rerank balances both.

### substance_list

Question: Danh mục chất ma túy và tiền chất được sửa đổi trong nghị định nào?

Why better/worse: Best stage: lexical_bm25. It has stronger query-term overlap and/or score. Semantic helps paraphrases, BM25 helps exact legal terms, hybrid+rerank balances both.

## Generation With Citation Scenario

Question: Luật Phòng chống ma túy 2021 quy định hành vi nào bị nghiêm cấm?

Answer:

--- title: "Nghị định 116/2021/NĐ-CP về cai nghiện ma túy" source_url: "https://vanban.chinhphu.vn/?docid=204866&pageid=27160" crawled_at: "2026-06-08T04:54:26.093305+00:00" source_type: "legal_web_text" html_title: "Nghị định số 116/2021/NĐ-CP của Chính phủ: [nghi dinh 116 2021 cai nghien ma tuy, 2021]. Bùi Thị Lệ Hằng (Ảnh: Công an Hà Nội). Theo dõi vụ việc, luật sư Hà Thị Khuyên (Trưởng Văn phòng luật sư Nhân Chính) cho biết, theo quy định pháp luật, hành vi mua bán trái phép chất ma túy được hiểu là hành vi trao đổi trái phép chất ma túy dưới bất kỳ hình t [2023 le hang dantri, 2023]. . Do đó, cần thiết phải làm rõ hành vi tổ chức sử dụng trái phép chất ma túy để xử lý theo quy định tại Điều 255 Bộ luật hình sự 2015. [2018 chau viet cuong chinhphu, 2018].

Sources:

- nghi-dinh-116-2021-cai-nghien-ma-tuy.md
- nghi-dinh-105-2021.md
- 2023-le-hang-dantri.md
- 2023-le-hang-dantri.md
- 2018-chau-viet-cuong-chinhphu.md

## Team Architecture

Streamlit UI -> Search Service -> Task 5 Semantic + Task 6 BM25 -> RRF -> Task 7 Rerank -> Results

Citation flow: Streamlit UI -> Task 10 Generation -> source/citation display.

## Repository Push

Code is prepared locally. Push to the shared repository with:

```powershell
git add group_project src
git commit -m "Add group search engine option A"
git push
```