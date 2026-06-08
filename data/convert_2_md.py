import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from markitdown import MarkItDown


PROJECT_ROOT = Path(
    r"E:\Downloads\Lab_Handson_AI_Action\2A202600883-MaiHanhPham-Day08_RAG_pipeline_cohort2"
)

LANDING_DIR = PROJECT_ROOT / "data" / "landing"
OUTPUT_DIR = PROJECT_ROOT / "data" / "standardized"

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".html",
    ".htm",
    ".json",
    ".txt",
}


def safe_yaml_value(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    text = text.replace("\n", " ")
    return text.strip()


def safe_read_json(file_path: Path) -> dict:
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def convert_json_article_to_markdown(file_path: Path) -> str:
    data = safe_read_json(file_path)

    metadata = data.get("metadata", {})
    content = data.get("content", {})

    title = metadata.get("title") or file_path.stem
    source_url = metadata.get("source_url", "")
    crawled_at = metadata.get("crawled_at", "")
    topic = metadata.get("topic", "")
    markdown = content.get("markdown", "")

    result = [
        "---",
        f'title: "{safe_yaml_value(title)}"',
        f'source_url: "{safe_yaml_value(source_url)}"',
        f'crawled_at: "{safe_yaml_value(crawled_at)}"',
        f'topic: "{safe_yaml_value(topic)}"',
        f'converted_at: "{datetime.now(timezone.utc).isoformat()}"',
        f'original_file: "{safe_yaml_value(file_path.name)}"',
        "---",
        "",
        f"# {title}",
        "",
        markdown.strip(),
        "",
    ]

    return "\n".join(result)


def convert_with_markitdown(md_converter: MarkItDown, file_path: Path) -> str:
    result = md_converter.convert(str(file_path))
    text_content = result.text_content or ""

    title = file_path.stem

    output = [
        "---",
        f'title: "{safe_yaml_value(title)}"',
        f'source_file: "{safe_yaml_value(str(file_path))}"',
        f'converted_at: "{datetime.now(timezone.utc).isoformat()}"',
        f'original_extension: "{safe_yaml_value(file_path.suffix.lower())}"',
        "---",
        "",
        f"# {title}",
        "",
        text_content.strip(),
        "",
    ]

    return "\n".join(output)


def build_output_path(input_path: Path) -> Path:
    relative_path = input_path.relative_to(LANDING_DIR)
    output_path = OUTPUT_DIR / relative_path
    return output_path.with_suffix(".md")


def convert_file(md_converter: MarkItDown, input_path: Path) -> bool:
    suffix = input_path.suffix.lower()

    if input_path.name == ".gitkeep":
        print(f"Skip .gitkeep: {input_path}")
        return False

    if suffix not in SUPPORTED_EXTENSIONS:
        print(f"Skip unsupported: {input_path}")
        return False

    output_path = build_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if suffix == ".json":
            markdown_text = convert_json_article_to_markdown(input_path)
        else:
            markdown_text = convert_with_markitdown(md_converter, input_path)

        output_path.write_text(markdown_text, encoding="utf-8")

        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"Converted: {input_path} -> {output_path}")
            return True

        print(f"Failed empty output: {input_path}")
        return False

    except Exception as error:
        print(f"Failed: {input_path}")
        print(f"Reason: {error}")
        return False


def collect_files() -> list[Path]:
    return [
        file_path
        for file_path in LANDING_DIR.rglob("*")
        if file_path.is_file()
    ]


def print_folder_status() -> None:
    legal_dir = LANDING_DIR / "legal"
    news_dir = LANDING_DIR / "news"

    print(f"Landing folder: {LANDING_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Legal folder exists: {legal_dir.exists()} -> {legal_dir}")
    print(f"News folder exists: {news_dir.exists()} -> {news_dir}")

    if legal_dir.exists():
        legal_files = [p for p in legal_dir.rglob("*") if p.is_file()]
        print(f"Legal files found: {len(legal_files)}")
        for file_path in legal_files:
            print(f"  - {file_path.name}")

    if news_dir.exists():
        news_files = [p for p in news_dir.rglob("*") if p.is_file()]
        print(f"News files found: {len(news_files)}")
        for file_path in news_files:
            print(f"  - {file_path.name}")

    print("-" * 80)


def main() -> None:
    if not LANDING_DIR.exists():
        raise FileNotFoundError(f"Landing folder not found: {LANDING_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_folder_status()

    md_converter = MarkItDown()
    files = collect_files()

    print(f"Total files found in landing: {len(files)}")
    print("-" * 80)

    success_count = 0
    failed_or_skipped_count = 0

    for file_path in files:
        success = convert_file(md_converter, file_path)

        if success:
            success_count += 1
        else:
            failed_or_skipped_count += 1

    print("-" * 80)
    print(f"Done. Markdown files saved to: {OUTPUT_DIR}")
    print(f"Success: {success_count}")
    print(f"Failed or skipped: {failed_or_skipped_count}")


if __name__ == "__main__":
    main()