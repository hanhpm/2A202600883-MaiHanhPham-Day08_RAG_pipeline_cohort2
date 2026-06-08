# Group Project - Option A Search Engine

## Product

Search engine for Vietnamese drug law and drug-related news.

Requirements from the assignment:

- Web UI with Streamlit.
- Hybrid search + reranking.
- Display source and relevance score.
- Simple black/white UI.
- Show comparison output between retrieval configs.
- Explain why one config is better/worse.
- Log every user input.
- Include a Generation with Citation scenario.
- Provide architecture and team assignment.

## Architecture

```text
User
  |
  v
Streamlit UI: group_project/app.py
  |
  v
Search Service: group_project/search_service.py
  |
  +--> Task 5 semantic_search
  +--> Task 6 lexical_search BM25
  +--> Task 7 RRF fusion + reranking
  |
  v
Results: content + source + score + comparison explanation

Citation tab:
Streamlit UI -> Task 10 generate_with_citation -> answer + sources

Logging:
group_project/logs/search_logs.jsonl
group_project/logs/search_logs.csv
```

## Main Files

| File | Purpose |
|---|---|
| `group_project/app.py` | Streamlit black/white search UI |
| `group_project/search_service.py` | Hybrid search, comparison, logging, citation wrapper |
| `group_project/evaluation/group_test_scenarios.json` | Self-created test scenarios |
| `group_project/evaluation/run_group_tests.py` | Offline accuracy test and summary export |
| `group_project/group-summary.md` | Generated group report after running tests |

## Search Features

The app shows:

- Final reranked results.
- Semantic top results.
- BM25 top results.
- Hybrid RRF top results.
- Explanation of which stage is stronger and why.
- Source file and relevance score for every result.
- Citation generation output.
- Recent input logs.

## Run

From project root:

```powershell
pip install -r requirements.txt
streamlit run group_project/app.py
```

Then open the Streamlit URL shown in terminal.

## Run Evaluation Scenarios

```powershell
python -m group_project.evaluation.run_group_tests
```

This writes:

```text
group_project/group-summary.md
```

## Test Queries

Use these in the UI:

```text
Hình phạt cho tội tàng trữ trái phép chất ma túy là gì?
Luật Phòng chống ma túy 2021 quy định hành vi nào bị nghiêm cấm?
Cai nghiện ma túy bắt buộc được quy định như thế nào?
Nghệ sĩ nào bị điều tra liên quan đến ma túy?
Chi Dân liên quan đến vụ việc ma túy như thế nào?
Danh mục chất ma túy và tiền chất được sửa đổi trong nghị định nào?
```

## Team Assignment

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|---|---|---|---|
| Mai Hanh Pham | 2A202600883 | Data pipeline Task 1-3, standardized clean data | Done |
| Mai Hanh Pham | 2A202600883 | Task 4-6 indexing, semantic, lexical search | Done |
| Mai Hanh Pham | 2A202600883 | Task 7-10 rerank, PageIndex fallback, generation citation | Done |
| Group | TBD | Option A Streamlit search engine, evaluation summary, demo script | Done locally |

## Push To Shared Repository

After review:

```powershell
git add group_project src
git commit -m "Add group option A search engine"
git push
```

