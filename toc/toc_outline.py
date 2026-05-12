import fitz  # pip install pymupdf
import re
import json
import unicodedata
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional


# ============================================================
# Utilities
# ============================================================

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def slugify(text: str) -> str:
    """
    Create stable readable IDs from titles.

    Example:
    "1. Principe général du fonctionnement"
    -> "1-principe-general-du-fonctionnement"
    """
    text = strip_accents(normalize_space(text).lower())
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "section"


def is_noise_title(title: str) -> bool:
    t = strip_accents(normalize_space(title).lower()).strip(" :;.-")

    if not t:
        return True

    if t in {
        "sommaire",
        "table des matieres",
        "table des matiéres",
        "table de matieres",
        "table of contents",
        "contents",
        "toc",
    }:
        return True

    if re.fullmatch(r"page\s+\d+(\s+sur)?", t):
        return True

    if re.fullmatch(r"page\s+\d+\s+sur\s+\d+", t):
        return True

    if "version" in t:
        return True

    if "isbn" in t:
        return True

    if "tous droits reserves" in t:
        return True

    if re.fullmatch(r"[\d\W_]+", t):
        return True

    return False


def clean_number_spacing(text: str) -> str:
    """
    Fix spacing problems around section numbers.

    Examples:
    "1 . Principe" -> "1. Principe"
    "1 .1 Titre"  -> "1.1 Titre"
    "1. 1 Titre"  -> "1.1 Titre"
    """
    text = normalize_space(text)

    # 1 . Title -> 1. Title
    text = re.sub(r"^(\d+)\s+\.\s+", r"\1. ", text)

    # 1 .1 Title -> 1.1 Title
    text = re.sub(r"^(\d+)\s+\.(\d+)", r"\1.\2", text)

    # 1. 1 Title -> 1.1 Title
    text = re.sub(r"^(\d+)\.\s+(\d+)", r"\1.\2", text)

    # 1 . 1 . 2 Title -> 1.1.2 Title
    text = re.sub(r"(?<=\d)\s+\.\s+(?=\d)", ".", text)

    return normalize_space(text)


def extract_number(title: str) -> Optional[str]:
    """
    Extract visible section numbering from a title.

    Examples:
    - "1. Principe général" -> "1"
    - "1 Principe général" -> "1"
    - "1.2 Objectifs" -> "1.2"
    - "1.7.4. Synthèse" -> "1.7.4"
    - "I - Introduction" -> "I"
    """
    title = clean_number_spacing(title)

    # Decimal: 1, 1., 1.1, 1.2.3, 1.7.4.
    m = re.match(r"^(\d+(?:\.\d+)*\.?)\s+.+$", title)
    if m:
        return m.group(1).rstrip(".")

    # Roman: I - Title, II. Title, IV: Title
    m = re.match(r"^([IVXLCDM]+)\s*[\.\-\:]\s+.+$", title, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


def ensure_number_in_title(title: str, number: Optional[str]) -> str:
    """
    Ensure that the visible numbering is preserved inside the title.

    If the title already starts with the number, it is not duplicated.
    """
    title = clean_number_spacing(title)

    if not number:
        return title

    number = normalize_space(number).rstrip(".")

    # Already starts with:
    # "1. Title", "1 Title", "1 - Title", "1: Title"
    if re.match(rf"^{re.escape(number)}[\.\s\-:]+", title):
        return title

    return f"{number}. {title}"


# ============================================================
# 1. Embedded TOC extraction
# ============================================================

def extract_embedded_toc(pdf_path: str | Path) -> List[Dict[str, Any]]:
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    toc = doc.get_toc(simple=True)
    doc.close()

    entries = []

    for order, item in enumerate(toc):
        if len(item) < 3:
            continue

        level, raw_title, page = item
        raw_title = clean_number_spacing(raw_title)

        if not raw_title or is_noise_title(raw_title):
            continue

        if not isinstance(page, int) or page < 1 or page > total_pages:
            continue

        number = extract_number(raw_title)
        title = ensure_number_in_title(raw_title, number)

        entries.append({
            "level": int(level),
            "title": title,
            "number": number,
            "page": page,
            "order": order,
            "source": "embedded_toc"
        })

    return entries


# ============================================================
# 2. Printed TOC detection
# ============================================================

def page_contains_toc_keyword(text: str) -> bool:
    clean = strip_accents(text.lower())

    return (
        "sommaire" in clean
        or "table des matieres" in clean
        or "table de matieres" in clean
        or "table of contents" in clean
        or re.search(r"\bcontents\b", clean) is not None
    )


def clean_toc_line(line: str) -> str:
    line = normalize_space(line)

    # Remove dotted leaders: "Title ........ 5" -> "Title 5"
    line = re.sub(r"\.{2,}", " ", line)

    # Fix spacing around numbers
    line = clean_number_spacing(line)

    return normalize_space(line)


def count_toc_like_lines(text: str, total_pages: int) -> int:
    """
    Count lines that look like TOC entries:
    title ........ page_number
    title page_number
    """
    count = 0

    for raw_line in text.splitlines():
        line = clean_toc_line(raw_line)

        m = re.match(r"^(.+?)\s+(\d{1,4})$", line)
        if not m:
            continue

        title = normalize_space(m.group(1))
        page = int(m.group(2))

        if page < 1 or page > total_pages:
            continue

        if is_noise_title(title):
            continue

        count += 1

    return count


def find_printed_toc_pages(
    pdf_path: str | Path,
    max_scan_pages: int = 20,
    min_toc_like_lines: int = 3,
) -> List[int]:
    """
    Automatically find visible/printed TOC pages.

    Returns 1-based page numbers.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    candidates = []

    for i in range(min(max_scan_pages, total_pages)):
        page = doc[i]
        text = page.get_text("text")
        links = page.get_links()

        has_keyword = page_contains_toc_keyword(text)
        toc_line_count = count_toc_like_lines(text, total_pages)
        link_count = len(links)

        score = 0

        if has_keyword:
            score += 5

        score += min(toc_line_count, 10)

        if link_count >= 3:
            score += 3

        if has_keyword or toc_line_count >= min_toc_like_lines:
            candidates.append({
                "page": i + 1,
                "score": score,
                "toc_line_count": toc_line_count,
                "link_count": link_count,
                "has_keyword": has_keyword,
            })

    doc.close()

    candidates.sort(key=lambda x: (-x["score"], x["page"]))

    if not candidates:
        return []

    best_page = candidates[0]["page"]
    selected = [best_page]

    # Add directly following pages if they also look like TOC continuation pages
    for c in candidates[1:]:
        if c["page"] == best_page + 1 and c["toc_line_count"] >= min_toc_like_lines:
            selected.append(c["page"])

    selected = sorted(set(selected))

    print("Detected printed TOC candidate pages:")
    for c in candidates[:5]:
        print(
            f"  page={c['page']} score={c['score']} "
            f"toc_lines={c['toc_line_count']} links={c['link_count']} "
            f"keyword={c['has_keyword']}"
        )

    print("Selected printed TOC pages:", selected)

    return selected


# ============================================================
# 3. Printed TOC extraction
# ============================================================

def get_lines_with_positions(page: fitz.Page) -> List[Dict[str, Any]]:
    """
    Extract visual rows with coordinates.

    Some PDFs store the section number and title in separate text lines/blocks.
    This function groups fragments with close y-position into one visual row,
    then sorts them from left to right.

    Example visual row:
    1.    Principe général du fonctionnement du chef de groupe .... 5
    """
    data = page.get_text("dict")
    fragments = []

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            text = normalize_space(" ".join(span.get("text", "") for span in spans))
            if not text:
                continue

            bbox = line.get("bbox")
            if not bbox:
                continue

            x0, y0, x1, y1 = bbox

            fragments.append({
                "text": text,
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
            })

    fragments.sort(key=lambda x: (x["y0"], x["x0"]))

    rows = []
    y_tolerance = 8.0  # increase if numbers and titles are still separated

    for frag in fragments:
        added = False

        for row in rows:
            if abs(frag["y0"] - row["y0"]) <= y_tolerance:
                row["items"].append(frag)
                row["y0"] = min(row["y0"], frag["y0"])
                row["y1"] = max(row["y1"], frag["y1"])
                added = True
                break

        if not added:
            rows.append({
                "y0": frag["y0"],
                "y1": frag["y1"],
                "items": [frag],
            })

    visual_lines = []

    for row in rows:
        items = sorted(row["items"], key=lambda x: x["x0"])

        text = normalize_space(" ".join(item["text"] for item in items))
        text = clean_number_spacing(text)

        if not text:
            continue

        visual_lines.append({
            "text": text,
            "x0": min(item["x0"] for item in items),
            "y0": min(item["y0"] for item in items),
            "x1": max(item["x1"] for item in items),
            "y1": max(item["y1"] for item in items),
        })

    visual_lines.sort(key=lambda x: (x["y0"], x["x0"]))

    return visual_lines


def parse_toc_line(line: str, total_pages: int) -> Optional[Dict[str, Any]]:
    """
    Parse TOC lines like:
    Préambule ............ 3
    Bibliographie ........ 4
    1. Principe général .... 5
    1.1 Notion de compétence .... 6
    """
    line = clean_toc_line(line)

    m = re.match(r"^(.+?)\s+(\d{1,4})$", line)
    if not m:
        return None

    title = clean_number_spacing(m.group(1))
    page = int(m.group(2))

    if page < 1 or page > total_pages:
        return None

    if is_noise_title(title):
        return None

    number = extract_number(title)
    title = ensure_number_in_title(title, number)

    return {
        "title": title,
        "page": page,
        "number": number,
    }


def infer_levels_from_x(entries: List[Dict[str, Any]]) -> None:
    """
    Infer hierarchy level from x-position indentation.
    If explicit decimal numbering exists, use it first.

    Examples:
    - 1 -> level 1
    - 1.2 -> level 2
    - 1.2.3 -> level 3
    """
    if not entries:
        return

    x_values = sorted({round(e["x0"], 1) for e in entries if e.get("x0") is not None})

    if not x_values:
        for e in entries:
            e["level"] = 1
        return

    clusters = []
    tolerance = 12.0

    for x in x_values:
        if not clusters or abs(x - clusters[-1][-1]) > tolerance:
            clusters.append([x])
        else:
            clusters[-1].append(x)

    centers = [sum(c) / len(c) for c in clusters]

    def level_from_x(x: float) -> int:
        idx = min(range(len(centers)), key=lambda i: abs(x - centers[i]))
        return idx + 1

    for e in entries:
        number = e.get("number")

        # Explicit decimal numbering is the strongest hierarchy signal
        if number and re.fullmatch(r"\d+(?:\.\d+)*", number):
            e["level"] = len(number.split("."))

        # Roman numerals are usually top-level sections
        elif number and re.fullmatch(r"[IVXLCDM]+", number):
            e["level"] = 1

        # Otherwise, use indentation
        else:
            e["level"] = level_from_x(e["x0"])


def extract_printed_toc(
    pdf_path: str | Path,
    toc_pages: List[int],
) -> List[Dict[str, Any]]:
    """
    Extract visible/printed TOC from specified page numbers.

    toc_pages are 1-based.
    Example: toc_pages=[2]
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    entries = []
    order = 0

    for toc_page in toc_pages:
        if toc_page < 1 or toc_page > total_pages:
            continue

        page = doc[toc_page - 1]
        lines = get_lines_with_positions(page)

        print(f"\n--- Visual lines detected on TOC page {toc_page} ---")
        for line in lines:
            print(line["text"])

        for line in lines:
            parsed = parse_toc_line(line["text"], total_pages)

            if parsed is None:
                continue

            entries.append({
                "level": None,
                "title": parsed["title"],
                "number": parsed["number"],
                "page": parsed["page"],
                "order": order,
                "source": "printed_toc",
                "x0": line["x0"],
                "y0": line["y0"],
            })

            order += 1

    doc.close()

    infer_levels_from_x(entries)

    return entries


# ============================================================
# 4. Hybrid extractor
# ============================================================

def extract_toc(
    pdf_path: str | Path,
    toc_pages: Optional[List[int]] = None,
    prefer_embedded: bool = True,
    auto_detect_printed_toc: bool = True,
    max_scan_pages: int = 20,
) -> List[Dict[str, Any]]:
    """
    General TOC extractor.

    1. Try embedded PDF bookmarks.
    2. If not found:
       - use toc_pages if provided;
       - otherwise auto-detect printed TOC pages.
    """
    pdf_path = Path(pdf_path)

    entries = []

    if prefer_embedded:
        entries = extract_embedded_toc(pdf_path)

    if entries:
        return entries

    if toc_pages is None and auto_detect_printed_toc:
        toc_pages = find_printed_toc_pages(
            pdf_path=pdf_path,
            max_scan_pages=max_scan_pages,
        )

    if not toc_pages:
        print("No embedded TOC and no printed TOC page detected.")
        return []

    return extract_printed_toc(pdf_path, toc_pages=toc_pages)


# ============================================================
# 5. Build hierarchical outline
# ============================================================

def build_hierarchical_outline(
    entries: List[Dict[str, Any]],
    document_title: str,
    total_pages: int,
) -> Dict[str, Any]:
    """
    Convert flat TOC entries into a nested hierarchical outline.

    Uses:
    - level for hierarchy
    - page for start_page
    - next same-or-higher-level section to infer end_page
    """
    flat_sections = []

    for i, e in enumerate(entries):
        title = normalize_space(e.get("title", ""))
        number = e.get("number")
        level = int(e.get("level", 1))
        start_page = int(e.get("page", 1))

        flat_sections.append({
            "id": slugify(title),
            "number": number,
            "title": title,
            "role": "heading",
            "level": level,
            "start_page": start_page,
            "end_page": start_page,
            "sections": [],
            "_order": i,
        })

    # --------------------------------------------------------
    # Compute end_page
    # --------------------------------------------------------
    for i, sec in enumerate(flat_sections):
        current_level = sec["level"]
        current_start = sec["start_page"]
        next_boundary_page = total_pages + 1

        for j in range(i + 1, len(flat_sections)):
            next_sec = flat_sections[j]

            # Current section ends before the next section
            # with same or higher hierarchy.
            if next_sec["level"] <= current_level:
                next_boundary_page = next_sec["start_page"]
                break

        sec["end_page"] = max(current_start, next_boundary_page - 1)

    # --------------------------------------------------------
    # Build nested tree using stack
    # --------------------------------------------------------
    root = {
        "title": document_title,
        "sections": []
    }

    stack = []

    for sec in flat_sections:
        level = sec["level"]

        while stack and stack[-1]["level"] >= level:
            stack.pop()

        if stack:
            stack[-1]["sections"].append(sec)
        else:
            root["sections"].append(sec)

        stack.append(sec)

    # --------------------------------------------------------
    # Remove temporary fields
    # --------------------------------------------------------
    def clean_section(section: Dict[str, Any]) -> None:
        section.pop("level", None)
        section.pop("_order", None)

        for child in section.get("sections", []):
            clean_section(child)

    for section in root["sections"]:
        clean_section(section)

    return root


# ============================================================
# 6. Save ONLY hierarchical outline JSON
# ============================================================

def save_toc_json(
    pdf_path: str | Path,
    output_json: str | Path,
    toc_pages: Optional[List[int]] = None,
    prefer_embedded: bool = True,
    auto_detect_printed_toc: bool = True,
    max_scan_pages: int = 20,
) -> Dict[str, Any]:
    """
    Save only the hierarchical outline.

    Output format:
    {
      "title": "...",
      "sections": [...]
    }
    """
    pdf_path = Path(pdf_path)
    output_json = Path(output_json)

    entries = extract_toc(
        pdf_path=pdf_path,
        toc_pages=toc_pages,
        prefer_embedded=prefer_embedded,
        auto_detect_printed_toc=auto_detect_printed_toc,
        max_scan_pages=max_scan_pages,
    )

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    outline = build_hierarchical_outline(
        entries=entries,
        document_title=pdf_path.stem,
        total_pages=total_pages,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(outline, f, ensure_ascii=False, indent=2)

    return outline


# ============================================================
# CLI
# ============================================================

def parse_toc_pages(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    txt = normalize_space(value)
    if not txt:
        return None
    pages: List[int] = []
    for part in txt.split(","):
        p = normalize_space(part)
        if not p:
            continue
        pages.append(int(p))
    return pages or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract TOC outline from PDF and save as JSON.")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--out_json", required=True, help="Path to output outline JSON")
    parser.add_argument(
        "--toc_pages",
        default=None,
        help="Optional comma-separated printed TOC pages (1-based), e.g. 2,3,4",
    )
    parser.add_argument(
        "--prefer_embedded",
        action="store_true",
        default=True,
        help="Prefer embedded PDF bookmarks first (default: True)",
    )
    parser.add_argument(
        "--no_prefer_embedded",
        action="store_false",
        dest="prefer_embedded",
        help="Disable embedded bookmark preference",
    )
    parser.add_argument(
        "--auto_detect_printed_toc",
        action="store_true",
        default=True,
        help="Auto-detect printed TOC pages when no embedded TOC (default: True)",
    )
    parser.add_argument(
        "--no_auto_detect_printed_toc",
        action="store_false",
        dest="auto_detect_printed_toc",
        help="Disable auto-detection of printed TOC pages",
    )
    parser.add_argument(
        "--max_scan_pages",
        type=int,
        default=20,
        help="Max first pages to scan for printed TOC detection (default: 20)",
    )
    args = parser.parse_args()

    toc_pages = parse_toc_pages(args.toc_pages)

    outline = save_toc_json(
        pdf_path=Path(args.pdf),
        output_json=Path(args.out_json),
        toc_pages=toc_pages,
        prefer_embedded=args.prefer_embedded,
        auto_detect_printed_toc=args.auto_detect_printed_toc,
        max_scan_pages=args.max_scan_pages,
    )

    print("\nSaved:", args.out_json)
    print("Root title:", outline["title"])
    print("Top-level sections:", len(outline["sections"]))


if __name__ == "__main__":
    main()
