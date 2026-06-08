# from pathlib import Path
# from urllib.parse import urljoin

# import requests
# from bs4 import BeautifulSoup


# OUTPUT_DIR = Path("./landing")

# DOCUMENTS = [
#     {
#         "page_url": "https://vanban.chinhphu.vn/?docid=204940&pageid=27160",
#         "filename": "luat-phong-chong-ma-tuy-2021.pdf",
#     },
#     {
#         "page_url": "https://vanban.chinhphu.vn/?docid=204678&pageid=27160",
#         "filename": "nghi-dinh-105-2021.pdf",
#     },
#     {
#         "page_url": "https://vanban.chinhphu.vn/?docid=204866&pageid=27160",
#         "filename": "nghi-dinh-116-2021-cai-nghien-ma-tuy.pdf",
#     },
#     {
#         "page_url": "https://vanban.chinhphu.vn/default.aspx?docid=183216&pageid=27160",
#         "filename": "bo-luat-hinh-su-2015.pdf",
#     },
#     {
#         "page_url": "https://vanban.chinhphu.vn/default.aspx?docid=190507&pageid=27160",
#         "filename": "luat-sua-doi-bo-luat-hinh-su-2017.pdf",
#     },
#     {
#         "page_url": "https://vanban.chinhphu.vn/?docid=206454&pageid=27160",
#         "filename": "nghi-dinh-57-2022-danh-muc-chat-ma-tuy-tien-chat.pdf",
#     },
#     {
#         "page_url": "https://vanban.chinhphu.vn/?docid=210694&pageid=27160",
#         "filename": "nghi-dinh-90-2024-sua-doi-danh-muc-chat-ma-tuy-tien-chat.pdf",
#     },
#     {
#         "page_url": "https://vanban.chinhphu.vn/?docid=216717&pageid=27160",
#         "filename": "nghi-dinh-28-2026-danh-muc-chat-ma-tuy-tien-chat.pdf",
#     },
#     {
#         "page_url": "https://vanban.chinhphu.vn/?classid=1&docid=204486&pageid=27160&typegroupid=6",
#         "filename": "thong-tu-18-2021-byt-xac-dinh-tinh-trang-nghien-ma-tuy.pdf",
#     },
#     {
#         "page_url": "https://vanban.chinhphu.vn/?docid=209668&pageid=27160",
#         "filename": "quyet-dinh-140-2024-phong-chong-ma-tuy-thanh-thieu-nien.pdf",
#     },
# ]


# def get_session() -> requests.Session:
#     session = requests.Session()
#     session.headers.update({
#         "User-Agent": (
#             "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#             "AppleWebKit/537.36 (KHTML, like Gecko) "
#             "Chrome/124.0 Safari/537.36"
#         )
#     })
#     return session


# def find_attachment_url(session: requests.Session, page_url: str) -> str:
#     response = session.get(page_url, timeout=30)
#     response.raise_for_status()

#     soup = BeautifulSoup(response.text, "html.parser")

#     candidates = []
#     for tag in soup.find_all("a", href=True):
#         href = tag["href"].strip()
#         text = tag.get_text(" ", strip=True).lower()
#         href_lower = href.lower()

#         is_document = href_lower.endswith((".pdf", ".docx", ".doc"))
#         looks_like_attachment = "datafiles.chinhphu.vn" in href_lower or "tải" in text or "signed" in text

#         if is_document and looks_like_attachment:
#             candidates.append(urljoin(page_url, href))

#     if not candidates:
#         raise ValueError(f"Không tìm thấy file PDF/DOCX đính kèm trong trang: {page_url}")

#     return candidates[0]


# def download_file(session: requests.Session, file_url: str, output_path: Path) -> None:
#     with session.get(file_url, stream=True, timeout=60) as response:
#         response.raise_for_status()

#         content_type = response.headers.get("Content-Type", "").lower()
#         if not any(x in content_type for x in ["pdf", "word", "octet-stream"]):
#             print(f"Warning: Content-Type lạ cho {file_url}: {content_type}")

#         with output_path.open("wb") as file:
#             for chunk in response.iter_content(chunk_size=1024 * 256):
#                 if chunk:
#                     file.write(chunk)

#     if output_path.stat().st_size < 1024:
#         raise ValueError(f"File tải về quá nhỏ, có thể bị lỗi: {output_path}")


# def main() -> None:
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     session = get_session()

#     success_count = 0

#     for doc in DOCUMENTS:
#         page_url = doc["page_url"]
#         filename = doc["filename"]
#         output_path = OUTPUT_DIR / filename

#         try:
#             if output_path.exists() and output_path.stat().st_size > 1024:
#                 print(f"Skip existing: {output_path}")
#                 success_count += 1
#                 continue

#             attachment_url = find_attachment_url(session, page_url)
#             download_file(session, attachment_url, output_path)

#             print(f"Downloaded: {output_path}")
#             print(f"Source: {attachment_url}")
#             success_count += 1

#         except Exception as error:
#             print(f"Failed: {filename}")
#             print(f"Page: {page_url}")
#             print(f"Reason: {error}")

#     print(f"\nDone. Downloaded/available {success_count}/{len(DOCUMENTS)} files.")
#     print(f"Output folder: {OUTPUT_DIR.resolve()}")


# if __name__ == "__main__":
#     main()

import re
import time
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(
    r"E:\Downloads\Lab_Handson_AI_Action\2A202600883-MaiHanhPham-Day08_RAG_pipeline_cohort2"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "landing" / "legal_text"

DOCUMENTS = [
    {
        "page_url": "https://vanban.chinhphu.vn/?docid=204940&pageid=27160",
        "filename": "luat-phong-chong-ma-tuy-2021.md",
        "title": "Luật Phòng, chống ma túy 2021",
    },
    {
        "page_url": "https://vanban.chinhphu.vn/?docid=204678&pageid=27160",
        "filename": "nghi-dinh-105-2021.md",
        "title": "Nghị định 105/2021/NĐ-CP",
    },
    {
        "page_url": "https://vanban.chinhphu.vn/?docid=204866&pageid=27160",
        "filename": "nghi-dinh-116-2021-cai-nghien-ma-tuy.md",
        "title": "Nghị định 116/2021/NĐ-CP về cai nghiện ma túy",
    },
    {
        "page_url": "https://vanban.chinhphu.vn/default.aspx?docid=183216&pageid=27160",
        "filename": "bo-luat-hinh-su-2015.md",
        "title": "Bộ luật Hình sự 2015",
    },
    {
        "page_url": "https://vanban.chinhphu.vn/default.aspx?docid=190507&pageid=27160",
        "filename": "luat-sua-doi-bo-luat-hinh-su-2017.md",
        "title": "Luật sửa đổi Bộ luật Hình sự 2017",
    },
    {
        "page_url": "https://vanban.chinhphu.vn/?docid=206454&pageid=27160",
        "filename": "nghi-dinh-57-2022-danh-muc-chat-ma-tuy-tien-chat.md",
        "title": "Nghị định 57/2022/NĐ-CP về danh mục chất ma túy và tiền chất",
    },
    {
        "page_url": "https://vanban.chinhphu.vn/?docid=210694&pageid=27160",
        "filename": "nghi-dinh-90-2024-sua-doi-danh-muc-chat-ma-tuy-tien-chat.md",
        "title": "Nghị định 90/2024/NĐ-CP sửa đổi danh mục chất ma túy và tiền chất",
    },
    {
        "page_url": "https://vanban.chinhphu.vn/?docid=216717&pageid=27160",
        "filename": "nghi-dinh-28-2026-danh-muc-chat-ma-tuy-tien-chat.md",
        "title": "Nghị định 28/2026/NĐ-CP về danh mục chất ma túy và tiền chất",
    },
    {
        "page_url": "https://vanban.chinhphu.vn/?classid=1&docid=204486&pageid=27160&typegroupid=6",
        "filename": "thong-tu-18-2021-byt-xac-dinh-tinh-trang-nghien-ma-tuy.md",
        "title": "Thông tư 18/2021/TT-BYT xác định tình trạng nghiện ma túy",
    },
    {
        "page_url": "https://vanban.chinhphu.vn/?docid=209668&pageid=27160",
        "filename": "quyet-dinh-140-2024-phong-chong-ma-tuy-thanh-thieu-nien.md",
        "title": "Quyết định 140/QĐ-TTg về phòng, chống ma túy trong thanh, thiếu niên",
    },
]


NOISE_PATTERNS = [
    r"^Trang chủ$",
    r"^Văn bản$",
    r"^Tìm kiếm$",
    r"^Chính phủ$",
    r"^Cổng thông tin điện tử Chính phủ$",
    r"^Bản quyền.*$",
    r"^©.*$",
    r"^Liên hệ$",
    r"^Sơ đồ cổng$",
    r"^RSS$",
    r"^English$",
    r"^Tải về$",
    r"^Download$",
    r"^In văn bản$",
    r"^Gửi văn bản$",
]

NOISE_REGEX = [re.compile(pattern, re.IGNORECASE) for pattern in NOISE_PATTERNS]


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    })
    return session


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_noise_line(line: str) -> bool:
    line = line.strip()

    if not line:
        return True

    if len(line) <= 2:
        return True

    if re.fullmatch(r"[\W_]+", line):
        return True

    for pattern in NOISE_REGEX:
        if pattern.match(line):
            return True

    return False


def extract_metadata(soup: BeautifulSoup) -> dict:
    metadata = {}

    page_title = soup.find("title")
    if page_title:
        metadata["html_title"] = clean_text(page_title.get_text(" ", strip=True))

    for meta in soup.find_all("meta"):
        name = meta.get("name") or meta.get("property")
        content = meta.get("content")

        if name and content:
            name = name.strip().lower()
            if name in {"description", "keywords", "og:title", "og:description"}:
                metadata[name] = clean_text(content)

    return metadata


def find_text_page_url(soup: BeautifulSoup, base_url: str) -> str | None:
    keywords = [
        "toàn văn",
        "xem toàn văn",
        "nội dung",
        "chi tiết",
        "xem văn bản",
    ]

    for link in soup.find_all("a", href=True):
        href = link["href"].strip()
        text = link.get_text(" ", strip=True).lower()

        if href.lower().endswith((".pdf", ".doc", ".docx")):
            continue

        if any(keyword in text for keyword in keywords):
            return urljoin(base_url, href)

    return None


def remove_unwanted_tags(soup: BeautifulSoup) -> None:
    for tag_name in ["script", "style", "noscript", "iframe", "svg", "form"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()


def extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    remove_unwanted_tags(soup)

    candidate_selectors = [
        "article",
        ".content",
        ".detail",
        ".detail-content",
        ".news-content",
        ".item-page",
        ".main-content",
        "#content",
        "body",
    ]

    selected = None

    for selector in candidate_selectors:
        selected = soup.select_one(selector)
        if selected and len(selected.get_text(" ", strip=True)) > 500:
            break

    if selected is None:
        selected = soup.body or soup

    raw_lines = selected.get_text("\n", strip=True).splitlines()

    cleaned_lines = []
    previous_line = ""

    for line in raw_lines:
        line = clean_text(line)

        if is_noise_line(line):
            continue

        if line == previous_line:
            continue

        cleaned_lines.append(line)
        previous_line = line

    return "\n".join(cleaned_lines).strip()


def crawl_page_text(session: requests.Session, page_url: str) -> tuple[str, dict]:
    response = session.get(page_url, timeout=30)
    response.raise_for_status()
    response.encoding = response.apparent_encoding

    soup = BeautifulSoup(response.text, "html.parser")
    metadata = extract_metadata(soup)

    text_page_url = find_text_page_url(soup, page_url)

    if text_page_url and text_page_url != page_url:
        text_response = session.get(text_page_url, timeout=30)
        text_response.raise_for_status()
        text_response.encoding = text_response.apparent_encoding

        metadata["text_page_url"] = text_page_url
        text = extract_visible_text(text_response.text)
        return text, metadata

    text = extract_visible_text(response.text)
    return text, metadata


def build_markdown(doc: dict, text: str, metadata: dict) -> str:
    source_url = doc["page_url"]
    title = doc["title"]

    lines = [
        "---",
        f'title: "{title}"',
        f'source_url: "{source_url}"',
        f'crawled_at: "{datetime.now(timezone.utc).isoformat()}"',
        'source_type: "legal_web_text"',
        f'html_title: "{metadata.get("html_title", "")}"',
        f'text_page_url: "{metadata.get("text_page_url", "")}"',
        "---",
        "",
        f"# {title}",
        "",
        text.strip(),
        "",
    ]

    return "\n".join(lines)


def save_document(doc: dict, markdown: str) -> None:
    output_path = OUTPUT_DIR / doc["filename"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    print(f"Saved: {output_path}")
    print(f"Characters: {len(markdown)}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = get_session()
    success_count = 0

    for doc in DOCUMENTS:
        try:
            print(f"\nCrawling: {doc['title']}")
            print(f"URL: {doc['page_url']}")

            text, metadata = crawl_page_text(session, doc["page_url"])

            if len(text) < 300:
                print("Warning: extracted text is short. This page may only contain metadata, not full legal text.")

            markdown = build_markdown(doc, text, metadata)
            save_document(doc, markdown)

            success_count += 1
            time.sleep(1)

        except Exception as error:
            print(f"Failed: {doc['title']}")
            print(f"Reason: {error}")

    print(f"\nDone. Crawled {success_count}/{len(DOCUMENTS)} documents.")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()