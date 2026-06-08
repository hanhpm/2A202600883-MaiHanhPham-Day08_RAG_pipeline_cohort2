import re
from pathlib import Path


PROJECT_ROOT = Path(
    r"E:\Downloads\Lab_Handson_AI_Action\2A202600883-MaiHanhPham-Day08_RAG_pipeline_cohort2"
)

INPUT_DIR = PROJECT_ROOT / "data" / "standardized"
OUTPUT_DIR = PROJECT_ROOT / "data" / "cleaned"

NOISE_PATTERNS = [
    r"^Bạn cần biết$",
    r"^Tiện ích$",
    r"^Liên hệ$",
    r"^Quảng cáo$",
    r"^Đặt báo$",
    r"^Đăng nhập$",
    r"^Đăng xuất$",
    r"^Bình luận.*$",
    r"^Gửi bình luận$",
    r"^Xem thêm bình luận$",
    r"^Quan tâm nhất$",
    r"^Mới nhất$",
    r"^Theo dõi báo trên$",
    r"^Hotline$",
    r"^Liên hệ quảng cáo$",
    r"^Tổng biên tập.*$",
    r"^Phó tổng biên tập.*$",
    r"^Tổng thư ký.*$",
    r"^Giấy phép xuất bản.*$",
    r"^Bản quyền.*$",
    r"^©.*$",
    r"^RSS$",
    r"^Tòa soạn$",
    r"^Chính sách bảo mật$",
    r"^Thông tin tài khoản$",
    r"^Đổi mật khẩu$",
    r"^Tin đã lưu$",
    r"^Tin đã xem$",
    r"^Đóng menu$",
    r"^Top$",
    r"^Chia sẻ$",
    r"^Copy link$",
    r"^In$",
    r"^Xem tất cả$",
    r"^Tin liên quan$",
    r"^Khám phá thêm chủ đề$",
    r"^Trang chủ$",
    r"^Chính trị$",
    r"^Thời sự$",
    r"^Thế giới$",
    r"^Kinh tế$",
    r"^Đời sống$",
    r"^Sức khỏe$",
    r"^Sức khoẻ$",
    r"^Giới trẻ$",
    r"^Giáo dục$",
    r"^Du lịch$",
    r"^Văn hóa$",
    r"^Văn hoá$",
    r"^Giải trí$",
    r"^Thể thao$",
    r"^Công nghệ.*$",
    r"^Xe$",
    r"^Video$",
    r"^Magazine$",
    r"^Bạn đọc$",
    r"^Rao vặt$",
]

NOISE_REGEX = [re.compile(pattern, re.IGNORECASE) for pattern in NOISE_PATTERNS]

MIN_LINE_LENGTH = 4


def remove_markdown_links_keep_text(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def remove_raw_urls(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    return text


def remove_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_yaml_delimiter(line: str) -> bool:
    return line.strip() == "---"


def split_yaml_frontmatter(text: str) -> tuple[str, str]:
    lines = text.splitlines()

    if len(lines) < 3 or not is_yaml_delimiter(lines[0]):
        return "", text

    for index in range(1, len(lines)):
        if is_yaml_delimiter(lines[index]):
            yaml_part = "\n".join(lines[: index + 1])
            body_part = "\n".join(lines[index + 1 :])
            return yaml_part, body_part

    return "", text


def is_noise_line(line: str) -> bool:
    stripped = line.strip()

    if not stripped:
        return False

    if len(stripped) < MIN_LINE_LENGTH:
        return True

    if stripped in {"[]", "[ ]", "*", "-", "•"}:
        return True

    if re.fullmatch(r"[\W_]+", stripped):
        return True

    if re.fullmatch(r"\d+\s*giây nữa\.?", stripped, re.IGNORECASE):
        return True

    if re.fullmatch(r"0\d{8,11}", stripped):
        return True

    for pattern in NOISE_REGEX:
        if pattern.match(stripped):
            return True

    return False


def clean_markdown_text(text: str) -> str:
    yaml_part, body = split_yaml_frontmatter(text)

    body = remove_html_tags(body)
    body = remove_markdown_links_keep_text(body)
    body = remove_raw_urls(body)

    cleaned_lines = []

    for raw_line in body.splitlines():
        line = raw_line.strip()
        line = re.sub(r"\s+", " ", line)

        if is_noise_line(line):
            continue

        cleaned_lines.append(line)

    cleaned_body = "\n".join(cleaned_lines)
    cleaned_body = normalize_whitespace(cleaned_body)

    if yaml_part:
        return f"{yaml_part}\n\n{cleaned_body}\n"

    return f"{cleaned_body}\n"


def build_output_path(input_path: Path) -> Path:
    relative_path = input_path.relative_to(INPUT_DIR)
    return OUTPUT_DIR / relative_path


def clean_file(input_path: Path) -> None:
    output_path = build_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_text = input_path.read_text(encoding="utf-8", errors="ignore")
    cleaned_text = clean_markdown_text(raw_text)

    output_path.write_text(cleaned_text, encoding="utf-8")

    raw_len = len(raw_text)
    clean_len = len(cleaned_text)
    reduction = 0 if raw_len == 0 else (1 - clean_len / raw_len) * 100

    print(f"Cleaned: {input_path} -> {output_path}")
    print(f"Reduction: {reduction:.2f}%")


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    markdown_files = [
        file_path
        for file_path in INPUT_DIR.rglob("*.md")
        if file_path.is_file()
    ]

    if not markdown_files:
        print(f"No Markdown files found in: {INPUT_DIR}")
        return

    for file_path in markdown_files:
        clean_file(file_path)

    print(f"\nDone. Cleaned files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()