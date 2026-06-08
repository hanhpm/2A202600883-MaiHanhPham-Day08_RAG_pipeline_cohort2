# Phân tích - Nhiệm vụ 6 Tìm kiếm từ vựng

## 1. Yêu cầu README

Nhiệm vụ 6 yêu cầu một mô-đun tìm kiếm từ vựng với giao diện sau:

```python

def lexical_search(query: str, top_k: int = 10) -> list[dict]:

""

Trả về:

Danh sách {'content': str, 'score': float, 'metadata': dict}

""

```

## 10. Verified external server status

Current Task 6 code does connect to real external services for the external
methods:

- Elasticsearch: `requests.post("http://localhost:9200/drug_law_docs/_search")`.
- Weaviate: `weaviate.connect_to_local()` and `collection.query.bm25(...)`.

Verification on the current machine:

```text
Elasticsearch localhost:9200: not running / connection refused
Weaviate localhost:8080 ready endpoint: 200
Weaviate collection DrugLawDocs object count: 1069
```

Because Elasticsearch is not running, `method="elasticsearch"` now prints an
external unavailable error instead of silently returning local BM25 results.
Because Weaviate is running and has indexed data, `method="weaviate_bm25"` uses
real Weaviate BM25 built-in search.

Local fallback for external methods is opt-in only:

```powershell
$env:ALLOW_EXTERNAL_FALLBACK="true"
```

Việc triển khai mặc định phải sử dụng BM25. README cũng đưa ra một tùy chọn thưởng:

nếu sử dụng phương pháp tìm kiếm từ vựng khác, chẳng hạn như TF-IDF, Elasticsearch hoặc Weaviate BM25 tích hợp sẵn,
hãy giải thích cơ chế trong phần trình diễn để được cộng thêm 5 điểm thưởng.

Việc triển khai trong `src/task6_lexical_search.py` giữ BM25 làm phương pháp mặc định.
Vì vậy, các bài kiểm tra tự động và quy trình truy xuất vẫn gọi:

```python
```lexical_search(query, top_k=10)```

Các phương pháp demo bổ sung có thể được chọn bằng:

```python
```python
```lexical_search(query, top_k=10, method="tfidf")`
```lexical_search(query, top_k=10, method="elasticsearch")`
```lexical_search(query, top_k=10, method="weaviate_bm25")`
```

## 2. Nguồn dữ liệu

Nhiệm vụ 6 tìm kiếm trên các khối được tạo bởi Nhiệm vụ 4. Tập dữ liệu được tải từ
chỉ mục vectơ hiện tại của Nhiệm vụ 4 thông qua:

```python
```ensure_index()`
```

Mỗi mục trong tập dữ liệu có:

```python
{ "nội dung": "...",

"siêu dữ liệu": {

"nguồn": "...",

"loại": "...",

"chỉ mục khối": ...

}

}
```

Điều này giúp việc truy xuất từ ​​vựng được đồng bộ với truy xuất ngữ nghĩa trong Nhiệm vụ 5 và
quy trình kết hợp trong Nhiệm vụ 9.

## 3. Phương pháp 1 - BM25

BM25 là phương pháp truy xuất từ ​​vựng mặc định.

BM25 chấm điểm một tài liệu bằng cách kết hợp:

- Tần suất thuật ngữ: thuật ngữ truy vấn xuất hiện thường xuyên hơn trong một khối sẽ làm tăng điểm số.

- Tần suất tài liệu nghịch đảo: các thuật ngữ hiếm gặp trong toàn bộ kho ngữ liệu được trọng số cao hơn.

- Chuẩn hóa độ dài tài liệu: các khối dài không được thưởng chỉ vì chúng dài.

Công thức trực quan:

```text
score(q, d) = tổng các thuật ngữ truy vấn:

IDF(term) * normalized_TF(term, document_length)
```

Trong dự án này:

- `rank_bm25.BM25Okapi` được sử dụng khi đã cài đặt.

- `SimpleBM25` được sử dụng làm phương án dự phòng với cùng logic cốt lõi.

Ưu điểm:

- Rất tốt cho các thuật ngữ chính xác như số điều luật, tên, tiêu đề tài liệu.

- Nhanh và dễ hiểu.

- Không yêu cầu embedding hoặc khóa API.

Hạn chế:

- Không hiểu sự tương đồng về ngữ nghĩa.

- Việc phân tách từ tiếng Việt được đơn giản hóa bằng bộ mã hóa Unicode, do đó việc khớp cụm từ

có thể kém chính xác hơn so với một công cụ phân tích tiếng Việt chuyên dụng.

Chạy lệnh:

```powershell
$env:LEXICAL_METHOD="bm25"
python -m src.task6_lexical_search
```

## 4. Phương pháp 2 - TF-IDF

TF-IDF là một phương pháp không gian vectơ từ vựng kinh điển.

Nó tính toán:

- TF: tần suất xuất hiện của một thuật ngữ trong tài liệu.

- IDF: độ hiếm của thuật ngữ trong toàn bộ tập dữ liệu.

- Độ tương đồng Cosine: so sánh vectơ truy vấn với từng vectơ tài liệu.

So với BM25:

- TF-IDF đơn giản hơn và thường dễ giải thích hơn.

- BM25 thường hoạt động tốt hơn trong tìm kiếm vì nó xử lý độ bão hòa thuật ngữ và

chuẩn hóa độ dài tài liệu cẩn thận hơn.

Tại sao nó hữu ích trong bản demo này:

- Nó là một tiêu chuẩn rõ ràng cho việc truy xuất từ ​​vựng.

- Nó giúp chứng minh tại sao BM25 thường được ưu tiên cho xếp hạng tìm kiếm.

Chạy lệnh:

```powershell
$env:LEXICAL_METHOD="tfidf"
python -m src.task6_lexical_search
```

## 5. Phương pháp 3 - Elasticsearch BM25

Elasticsearch sử dụng BM25 làm thuật toán xếp hạng mặc định cho tìm kiếm toàn văn.

Cơ chế:

- Tài liệu được lập chỉ mục vào chỉ mục Elasticsearch.

- Văn bản được phân tích thành các token.

- Truy vấn `multi_match` tìm kiếm các trường như `content`, `source` và `type`.

- Elasticsearch trả về `_score`, mặc định dựa trên BM25.

Trong dự án này, hàm `elasticsearch_search()` gửi truy vấn đến:

```text
ELASTICSEARCH_URL / ELASTICSEARCH_INDEX / _search
```

Các biến môi trường mặc định:

```powershell
$env:ELASTICSEARCH_URL="http://localhost:9200"
$env:ELASTICSEARCH_INDEX="drug_law_docs"
```

Triển khai hiện tại dùng Elasticsearch như một external service thật. Hàm sẽ
kiểm tra server, tạo index nếu cần, và bulk-index các chunk từ Task 4 khi index
đang rỗng.

Nếu Elasticsearch không chạy, hàm sẽ báo lỗi rõ thay vì fallback im lặng. Điều
này giúp chứng minh trong demo là phương pháp external có thật sự connect server
hay không. Fallback local BM25 chỉ bật khi chủ động set:

```powershell
$env:ALLOW_EXTERNAL_FALLBACK="true"
```

Tại sao nó hữu ích:

- Tìm kiếm từ khóa sẵn sàng cho môi trường sản xuất.

- Hỗ trợ các bộ phân tích, bộ lọc, tô sáng, tìm kiếm cụm từ và lập chỉ mục có thể mở rộng.

Chạy lệnh:

```powershell
$env:LEXICAL_METHOD="elasticsearch"
python -m src.task6_lexical_search
```

## 6. Phương pháp 4 - Tích hợp Weaviate BM25

Weaviate hỗ trợ tìm kiếm từ khóa BM25 bên trong cùng cơ sở dữ liệu vector được sử dụng cho
truy xuất ngữ nghĩa. Điều này hữu ích cho tìm kiếm kết hợp.

Cơ chế:

```powershell
Trien khai hien tai dung Weaviate nhu mot external service that:

- Ket noi `weaviate.connect_to_local()` toi `localhost:8080`.
- Tao collection `DrugLawDocs` neu collection chua ton tai.
- Index cac chunk tu Task 4 vao collection khi collection dang rong.
- Goi `collection.query.bm25(...)` de dung BM25 built-in cua Weaviate.

Ket qua kiem tra tren may hien tai:

```text
Weaviate ready endpoint: 200
DrugLawDocs object count: 1069
```

Neu Weaviate khong chay, ham se bao loi ro thay vi fallback im lang. Fallback
local BM25 chi bat khi chu dong set:

```powershell
$env:ALLOW_EXTERNAL_FALLBACK="true"
```

```powershell
$env:LEXICAL_METHOD="weaviate_bm25"
python -m src.task6_lexical_search
```

## 7. So sánh để minh họa

| Phương pháp | Sử dụng tốt nhất | Điểm mạnh | Điểm yếu |
|---|---|---|---|
| BM25 | Tìm kiếm từ khóa mặc định | Truy xuất khớp chính xác mạnh mẽ, có thể giải thích | Hiểu ngữ nghĩa hạn chế |
| TF-IDF | Truy xuất từ vựng cơ bản | Đơn giản và minh bạch | Xếp hạng kém tinh tế hơn BM25 |
| Elasticsearch BM25 | Tìm kiếm từ khóa trong môi trường sản xuất | Có khả năng mở rộng, hỗ trợ bộ phân tích, bộ lọc và làm nổi bật kết quả | Yêu cầu chạy Elasticsearch và lập chỉ mục |
| Weaviate BM25 | Hệ thống phụ trợ RAG lai | Hỗ trợ BM25 và tìm kiếm vector trong cùng một hệ thống | Yêu cầu chạy Weaviate và thiết lập collection |

## 8. Kịch bản demo được đề xuất

Sử dụng một truy vấn chứa các thuật ngữ pháp lý/tin tức chính xác:

```text

Điều 248 ma cà rồng
```

Sau đó hiển thị:

```powershell
python -m src.task6_lexical_search
```

Kịch bản sẽ in ra kết quả cho:

- BM25
- TF-IDF
- Elasticsearch BM25
- Weaviate BM25

Điều này trực tiếp đáp ứng phần giải thích bổ sung trong README vì nó so sánh BM25
với TF-IDF, Elasticsearch và Weaviate BM25 được tích hợp sẵn.

## 9. Output

```
(ai_action) PS E:\Downloads\Lab_Handson_AI_Action\2A202600883-MaiHanhPham-Day08_RAG_pipeline_cohort2> python -m src.task6_lexical_search

Method: bm25
[7.565] ###  [Người mẫu Andrea Aybar bị Công an TPHCM điều tra liên quan ma túy](https://dantri.com.vn/phap-...
[6.676] TIN LIÊN QUAN
### [Ca sĩ Chi Dân bị điều tra nghi liên quan đến ma túy](https://dantri.com.vn/phap-l...
[6.420] ###  [Cuộc sống của ca sĩ Chi Dân trước khi bị điều tra nghi liên quan ma túy](https://thanhnien.vn/...

Method: tfidf
[0.253] "Trần Hữu Tín") [ ma túy ](https://thanhnien.vn/ma-tuy-tags486312.html "ma túy") [ tổ chức sử dụng t...
[0.253] "Trần Hữu Tín") [ tổ chức sử dụng trái phép ma túy ](https://thanhnien.vn/to-chuc-su-dung-trai-phep-...
[0.249] ###  [Người mẫu Andrea Aybar bị Công an TPHCM điều tra liên quan ma túy](https://dantri.com.vn/phap-...

Method: elasticsearch
[7.565] ###  [Người mẫu Andrea Aybar bị Công an TPHCM điều tra liên quan ma túy](https://dantri.com.vn/phap-...
[6.676] TIN LIÊN QUAN
### [Ca sĩ Chi Dân bị điều tra nghi liên quan đến ma túy](https://dantri.com.vn/phap-l...
[6.420] ###  [Cuộc sống của ca sĩ Chi Dân trước khi bị điều tra nghi liên quan ma túy](https://thanhnien.vn/...

Method: weaviate_bm25
[7.565] ###  [Người mẫu Andrea Aybar bị Công an TPHCM điều tra liên quan ma túy](https://dantri.com.vn/phap-...
[6.676] TIN LIÊN QUAN
### [Ca sĩ Chi Dân bị điều tra nghi liên quan đến ma túy](https://dantri.com.vn/phap-l...
[6.420] ###  [Cuộc sống của ca sĩ Chi Dân trước khi bị điều tra nghi liên quan ma túy](https://thanhnien.vn/...
(ai_action) PS E:\Downloads\Lab_Handson_AI_Action\2A202600883-MaiHanhPham-Day08_RAG_pipeline_cohort2> 
```
