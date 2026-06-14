"""
extract_chunks.py – Trích xuất dữ liệu từ thư mục clean_data thành chunks tối ưu.
Phiên bản v2: Smart chunking cho Excel (gom 8 dòng/chunk), PDF table extraction,
phân loại metadata mở rộng, xuất JSONL compact.
"""

import os
import re
import sys
import json
import time
from datetime import datetime

import fitz       # PyMuPDF – đọc PDF
import pandas as pd

# ── Đảm bảo in Unicode trên Windows ─────────────────────────────────
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ── Cấu hình đường dẫn (portable — dựa trên vị trí script) ──────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

SOURCE_DIR  = os.path.join(_PROJECT_DIR, "clean_data")
OUTPUT_FILE = os.path.join(_SCRIPT_DIR, "extracted_chunks.jsonl")

# Số dòng Excel gộp thành 1 chunk. Mặc định 1 dòng để tối ưu tra cứu SKU/giá.
EXCEL_GROUP_SIZE = 1

# Kích thước tối thiểu (ký tự) cho 1 chunk PDF dạng paragraph
PDF_MIN_CHUNK_CHARS = 200

# Tên file cần bỏ qua (lowercase)
SKIP_FILES = {"search thông tin sản phẩm.xlsm"}

SKU_PATTERN = re.compile(r"(?:[A-Z]+-?\d+[A-Z]*-?\w*|\d+[A-Z]+-?\d*\w*)", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
PRICE_PATTERN = re.compile(r"(?<!\w)(?:VND|VNĐ|₫)?\s*([0-9]{1,3}(?:[.,\s][0-9]{3})+|[0-9]{5,})(?:\s*(?:VND|VNĐ|₫))?", re.IGNORECASE)

# ── Hàm phân loại metadata ──────────────────────────────────────────

def classify_brand(rel_path: str) -> str:
    """Phân loại thương hiệu dựa trên đường dẫn tương đối."""
    p = rel_path.lower()
    if "kallista" in p:
        return "Kallista"
    if "karat" in p:
        return "Karat"
    return "Kohler"


def classify_doc_type(rel_path: str) -> str:
    """Phân loại loại tài liệu dựa trên đường dẫn tương đối."""
    p = rel_path.lower()
    # Retail / Promotion – ưu tiên cao nhất vì thường nằm trong thư mục riêng
    if any(kw in p for kw in ("retail", "focus", "package", "promotion", "ctkm", "anniversary")):
        return "Retail/Promotion"
    if any(kw in p for kw in ("discontinued", "cancel")):
        return "Discontinued"
    if "bulletin" in p:
        return "Bulletin"
    if "price" in p:
        return "Price List"
    if "project" in p:
        return "Project"
    return "Other"


def normalize_sku(sku: str) -> str:
    """Chuẩn hoá SKU để exact lookup nhất quán giữa query và payload."""
    return re.sub(r"[^A-Z0-9]", "", sku.upper())


def extract_sku_variants(text: str) -> list[str]:
    """Trích SKU và biến thể không dấu gạch từ text."""
    variants = []
    seen = set()
    for sku in SKU_PATTERN.findall(text):
        for candidate in (sku.upper(), normalize_sku(sku)):
            if candidate and candidate not in seen:
                variants.append(candidate)
                seen.add(candidate)
    return variants


def extract_price(text: str) -> str:
    """Trích giá đầu tiên từ text nếu có."""
    match = PRICE_PATTERN.search(text)
    return match.group(0).strip() if match else ""


def _year_from_text(text: str) -> int | None:
    years = [int(y) for y in YEAR_PATTERN.findall(text)]
    return max(years) if years else None


def get_year(file_path: str, *text_hints: str) -> int:
    """Lấy năm từ filename/path/content trước, fallback mtime cuối cùng."""
    rel_or_name = os.path.basename(file_path)
    for source in (rel_or_name, file_path, *text_hints):
        detected = _year_from_text(source or "")
        if detected:
            return detected

    try:
        mtime = os.path.getmtime(file_path)
        return datetime.fromtimestamp(mtime).year
    except Exception:
        return datetime.now().year


# ── Xử lý file Excel ────────────────────────────────────────────────

def process_excel(file_path: str, rel_path: str, brand: str, doc_type: str, year: int) -> list[dict]:
    """
    Đọc từng sheet của file Excel, gom theo row-level/SKU-aware chunks.
    Luôn thêm dòng header ở đầu mỗi chunk để giữ ngữ cảnh cột.
    """
    chunks = []
    try:
        xls = pd.ExcelFile(file_path, engine="openpyxl")
    except Exception:
        try:
            xls = pd.ExcelFile(file_path, engine="xlrd")
        except Exception as e:
            print(f"  [SKIP] Không đọc được Excel: {rel_path} – {e}")
            return chunks

    for sheet_name in xls.sheet_names:
        try:
            df = xls.parse(sheet_name, header=None, dtype=str)
        except Exception:
            continue

        if df.empty:
            continue

        # Dòng đầu tiên là header
        headers = df.iloc[0].fillna("").astype(str).tolist()
        header_line = " | ".join(h.strip() for h in headers)

        # Các dòng dữ liệu (bỏ header)
        data_rows = df.iloc[1:]

        # Lọc bỏ dòng hoàn toàn trống
        non_empty_mask = data_rows.apply(
            lambda row: row.dropna().astype(str).str.strip().str.len().sum() > 0,
            axis=1
        )
        data_rows = data_rows[non_empty_mask]

        if data_rows.empty:
            continue

        # Gom nhóm EXCEL_GROUP_SIZE dòng
        row_indices = data_rows.index.tolist()
        for g_start in range(0, len(row_indices), EXCEL_GROUP_SIZE):
            g_end = min(g_start + EXCEL_GROUP_SIZE, len(row_indices))
            group_idx = row_indices[g_start:g_end]

            # Số thứ tự dòng gốc trong sheet (1-indexed, tính cả header)
            first_row = group_idx[0] + 1   # +1 vì 0-indexed trong pandas
            last_row  = group_idx[-1] + 1

            lines = []
            for idx in group_idx:
                row_vals = df.iloc[idx].fillna("").astype(str).tolist()
                lines.append(" | ".join(v.strip() for v in row_vals))

            row_text = "\n".join(lines)
            sku_variants = extract_sku_variants(row_text)
            price = extract_price(row_text)
            chunk_year = get_year(file_path, rel_path, sheet_name, row_text)

            content = (
                f"[File: {rel_path} | Sheet: {sheet_name} | Rows {first_row}-{last_row}]\n"
                f"Headers: {header_line}\n"
                + row_text
            )

            chunks.append({
                "content":     content,
                "source_file": rel_path,
                "location":    f"Sheet: {sheet_name} | Rows {first_row}-{last_row}",
                "year":        chunk_year or year,
                "brand":       brand,
                "doc_type":    doc_type,
                "chunk_type":  "excel_row" if len(group_idx) == 1 else "excel_rows",
                "sku_variants": sku_variants,
                "sku":         sku_variants[0] if sku_variants else "",
                "price":       price,
            })

    return chunks


# ── Xử lý file PDF ──────────────────────────────────────────────────

def _format_table(table) -> str:
    """Chuyển bảng PyMuPDF thành text có header."""
    rows = table.extract()
    if not rows:
        return ""
    # Dòng đầu là header
    header = " | ".join(str(c or "").strip() for c in rows[0])
    body_lines = []
    for row in rows[1:]:
        body_lines.append(" | ".join(str(c or "").strip() for c in row))
    return f"Headers: {header}\n" + "\n".join(body_lines)


def _chunk_paragraphs(text: str, min_chars: int) -> list[str]:
    """Chia text thành các đoạn, gộp đoạn nhỏ lại cho đủ min_chars."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    result = []
    buffer = ""
    for para in paragraphs:
        if buffer:
            buffer += "\n\n" + para
        else:
            buffer = para

        if len(buffer) >= min_chars:
            result.append(buffer)
            buffer = ""

    if buffer:
        # Gộp đoạn cuối vào chunk trước nếu quá nhỏ, hoặc tạo chunk mới
        if result and len(buffer) < min_chars:
            result[-1] += "\n\n" + buffer
        else:
            result.append(buffer)

    return result


def process_pdf(file_path: str, rel_path: str, brand: str, doc_type: str, year: int) -> list[dict]:
    """
    Đọc PDF bằng PyMuPDF. Ưu tiên trích xuất bảng (find_tables) nếu có,
    nếu không thì lấy text và chia paragraph.
    """
    chunks = []
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"  [SKIP] Không đọc được PDF: {rel_path} – {e}")
        return chunks

    has_find_tables = hasattr(fitz, "__version__") and tuple(
        int(x) for x in fitz.__version__.split(".")[:2]
    ) >= (1, 23)

    for page_num, page in enumerate(doc, start=1):
        table_texts = []

        # Thử trích xuất bảng nếu PyMuPDF đủ mới
        if has_find_tables:
            try:
                tables = page.find_tables()
                for tbl in tables:
                    formatted = _format_table(tbl)
                    if formatted.strip():
                        table_texts.append(formatted)
            except Exception:
                pass

        if table_texts:
            # Mỗi bảng tìm được → 1 chunk
            for t_idx, tbl_text in enumerate(table_texts, start=1):
                content = (
                    f"[File: {rel_path} | Page {page_num} | Table {t_idx}]\n"
                    + tbl_text
                )
                sku_variants = extract_sku_variants(tbl_text)
                chunks.append({
                    "content":     content,
                    "source_file": rel_path,
                    "location":    f"Page {page_num}, Table {t_idx}",
                    "year":        get_year(file_path, rel_path, tbl_text) or year,
                    "brand":       brand,
                    "doc_type":    doc_type,
                    "chunk_type":  "pdf_table",
                    "sku_variants": sku_variants,
                    "sku":         sku_variants[0] if sku_variants else "",
                    "price":       extract_price(tbl_text),
                })
        else:
            # Fallback: lấy text và chia paragraph
            text = page.get_text()
            if not text or not text.strip():
                continue

            para_chunks = _chunk_paragraphs(text, PDF_MIN_CHUNK_CHARS)
            for p_idx, para_text in enumerate(para_chunks, start=1):
                content = (
                    f"[File: {rel_path} | Page {page_num} | Part {p_idx}]\n"
                    + para_text
                )
                sku_variants = extract_sku_variants(para_text)
                chunks.append({
                    "content":     content,
                    "source_file": rel_path,
                    "location":    f"Page {page_num}, Part {p_idx}",
                    "year":        get_year(file_path, rel_path, para_text) or year,
                    "brand":       brand,
                    "doc_type":    doc_type,
                    "chunk_type":  "pdf_paragraph",
                    "sku_variants": sku_variants,
                    "sku":         sku_variants[0] if sku_variants else "",
                    "price":       extract_price(para_text),
                })

    doc.close()
    return chunks


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("EXTRACT CHUNKS v2 – Smart Chunking + JSONL Output")
    print("=" * 60)
    print(f"Thư mục nguồn : {SOURCE_DIR}")
    print(f"File xuất      : {OUTPUT_FILE}")
    print()

    # Thu thập danh sách file cần xử lý
    all_files = []
    for root, _dirs, files in os.walk(SOURCE_DIR):
        for fname in files:
            # Bỏ qua file tạm của Office và file blacklist
            if fname.startswith("~$"):
                continue
            if fname.lower() in SKIP_FILES:
                continue

            ext = os.path.splitext(fname)[1].lower()
            if ext in (".xlsx", ".xls", ".xlsm", ".pdf"):
                full_path = os.path.join(root, fname)
                all_files.append(full_path)

    print(f"Tìm thấy {len(all_files)} file cần xử lý.\n")

    total_chunks = 0
    stats_year     = {}
    stats_brand    = {}
    stats_doc_type = {}

    t_start = time.time()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for file_idx, file_path in enumerate(all_files, start=1):
            rel_path = os.path.relpath(file_path, SOURCE_DIR)
            ext      = os.path.splitext(file_path)[1].lower()

            brand    = classify_brand(rel_path)
            doc_type = classify_doc_type(rel_path)
            year     = get_year(file_path, rel_path)

            # Xử lý theo loại file
            if ext in (".xlsx", ".xls", ".xlsm"):
                file_chunks = process_excel(file_path, rel_path, brand, doc_type, year)
            elif ext == ".pdf":
                file_chunks = process_pdf(file_path, rel_path, brand, doc_type, year)
            else:
                file_chunks = []

            # Ghi JSONL – mỗi dòng 1 JSON object
            for chunk in file_chunks:
                out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            total_chunks += len(file_chunks)

            # Cập nhật thống kê
            for chunk in file_chunks:
                y = chunk["year"]
                b = chunk["brand"]
                d = chunk["doc_type"]
                stats_year[y]     = stats_year.get(y, 0) + 1
                stats_brand[b]    = stats_brand.get(b, 0) + 1
                stats_doc_type[d] = stats_doc_type.get(d, 0) + 1

            # In tiến độ mỗi 10 file
            if file_idx % 10 == 0 or file_idx == len(all_files):
                elapsed = time.time() - t_start
                print(f"[{file_idx}/{len(all_files)}] {elapsed:.1f}s – "
                      f"Tổng chunks: {total_chunks:,} – File: {rel_path}")

    elapsed_total = time.time() - t_start

    # ── In thống kê cuối cùng ────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"HOÀN THÀNH trong {elapsed_total:.1f}s")
    print(f"Tổng chunks : {total_chunks:,}")
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"File JSONL   : {OUTPUT_FILE}  ({file_size_mb:.1f} MB)")

    print("\n── Thống kê theo Năm (Year) ──")
    for y in sorted(stats_year.keys()):
        print(f"  {y}: {stats_year[y]:>8,} chunks")

    print("\n── Thống kê theo Thương hiệu (Brand) ──")
    for b in sorted(stats_brand.keys()):
        print(f"  {b:12s}: {stats_brand[b]:>8,} chunks")

    print("\n── Thống kê theo Loại tài liệu (DocType) ──")
    for d in sorted(stats_doc_type.keys()):
        print(f"  {d:20s}: {stats_doc_type[d]:>8,} chunks")

    print("=" * 60)


if __name__ == "__main__":
    main()
