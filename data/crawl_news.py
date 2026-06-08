import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crawl4ai import AsyncWebCrawler


OUTPUT_DIR = Path(
    r"E:\Downloads\Lab_Handson_AI_Action\2A202600883-MaiHanhPham-Day08_RAG_pipeline_cohort2\data\landing\news"
)

ARTICLES = [
    {
        "url": "https://tiengchuong.chinhphu.vn/vu-ca-si-chau-viet-cuong-hon-30-nhanh-cu-toi-gay-tac-duong-ho-hap-11322039.htm",
        "filename": "2018-chau-viet-cuong-chinhphu.json",
        "topic": "Ca sĩ Châu Việt Cường liên quan ma túy",
    },
    {
        "url": "https://laodong.vn/phap-luat/toan-canh-vu-ca-si-chau-viet-cuong-nhet-toi-vao-mieng-ban-tinh-661444.ldo",
        "filename": "2019-chau-viet-cuong-laodong.json",
        "topic": "Châu Việt Cường bị tuyên án",
    },
    {
        "url": "https://thanhnien.vn/tphcm-cong-an-q8-de-nghi-truy-to-dien-vien-huu-tin-to-chuc-su-dung-trai-phep-chat-ma-tuy-1851516857.htm",
        "filename": "2022-huu-tin-thanh-nien.json",
        "topic": "Diễn viên Hữu Tín bị đề nghị truy tố",
    },
    {
        "url": "https://thanhnien.vn/dien-vien-huu-tin-nghien-ma-tuy-gan-3-nam-moi-ban-ve-nha-su-dung-thuoc-lac-1851517030.htm",
        "filename": "2022-huu-tin-nghien-ma-tuy-thanh-nien.json",
        "topic": "Hữu Tín khai sử dụng ma túy",
    },
    {
        "url": "https://thanhnien.vn/nu-dien-vien-le-hang-bi-bat-vi-di-buon-ma-tuy-185230423181213443.htm",
        "filename": "2023-le-hang-thanh-nien.json",
        "topic": "Diễn viên Lệ Hằng bị bắt vì mua bán ma túy",
    },
    {
        "url": "https://dantri.com.vn/phap-luat/nu-dien-vien-dong-hoai-that-cho-co-the-doi-mat-hinh-phat-nao-20230424092227771.htm",
        "filename": "2023-le-hang-dantri.json",
        "topic": "Lệ Hằng và khía cạnh pháp lý",
    },
    {
        "url": "https://dantri.com.vn/phap-luat/khoi-to-bat-giam-nguoi-mau-andrea-aybar-ca-si-chi-dan-20241114115057035.htm",
        "filename": "2024-chi-dan-an-tay-dantri.json",
        "topic": "Chi Dân, An Tây bị khởi tố, bắt tạm giam",
    },
    {
        "url": "https://tienphong.vn/an-tay-chi-dan-khoa-trang-ca-nhan-post1690306.tpo",
        "filename": "2024-chi-dan-an-tay-tienphong.json",
        "topic": "Chi Dân, An Tây bị điều tra nghi liên quan ma túy",
    },
    {
        "url": "https://thanhnien.vn/chi-dan-huu-tin-va-loat-sao-viet-gay-on-ao-vi-dinh-toi-ma-tuy-185241110141122628.htm",
        "filename": "2024-chi-dan-an-tay-thanh-nien.json",
        "topic": "Tổng hợp nghệ sĩ Việt liên quan ma túy",
    },
    {
        "url": "https://dantri.com.vn/phap-luat/truy-to-ca-si-chi-dan-nguoi-mau-an-tay-20260402122649916.htm",
        "filename": "2026-truy-to-chi-dan-an-tay-dantri.json",
        "topic": "Truy tố Chi Dân, An Tây",
    },
]


def clean_title(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        line = line.strip()
        if not line:
            continue

        line = re.sub(r"^#+\s*", "", line).strip()
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line).strip()

        if len(line) >= 10:
            return line[:250]

    return fallback


def build_payload(article: dict[str, str], result: Any) -> dict[str, Any]:
    markdown = result.markdown or ""
    html = getattr(result, "html", None)

    return {
        "metadata": {
            "source_url": article["url"],
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "title": clean_title(markdown, article["topic"]),
            "topic": article["topic"],
            "crawler": "crawl4ai.AsyncWebCrawler",
        },
        "content": {
            "markdown": markdown,
            "html": html,
        },
    }


async def crawl_article(crawler: AsyncWebCrawler, article: dict[str, str]) -> None:
    output_path = OUTPUT_DIR / article["filename"]

    try:
        result = await crawler.arun(url=article["url"])

        if not getattr(result, "success", True):
            raise RuntimeError(getattr(result, "error_message", "Crawl failed"))

        payload = build_payload(article, result)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

        print(f"Saved: {output_path}")

    except Exception as error:
        print(f"Failed: {article['url']}")
        print(f"Reason: {error}")


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with AsyncWebCrawler() as crawler:
        for article in ARTICLES:
            await crawl_article(crawler, article)

    print(f"Done. Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())