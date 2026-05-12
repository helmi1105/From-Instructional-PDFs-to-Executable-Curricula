import json
import re
import unicodedata
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set


# ============================================================
# Basic utilities
# ============================================================

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def slugify(text: str, fallback: str = "section") -> str:
    text = strip_accents(text.lower().strip())
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or fallback


def make_unique_id(base_id: str, used_ids: Set[str]) -> str:
    if base_id not in used_ids:
        used_ids.add(base_id)
        return base_id

    i = 2
    while f"{base_id}-{i}" in used_ids:
        i += 1

    new_id = f"{base_id}-{i}"
    used_ids.add(new_id)
    return new_id


def roman_to_int(roman: str) -> Optional[int]:
    roman = roman.upper()
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0

    if not re.fullmatch(r"[IVXLCDM]+", roman):
        return None

    for ch in reversed(roman):
        val = values[ch]
        if val < prev:
            total -= val
        else:
            total += val
        prev = val

    return total


# ============================================================
# Page handling: PDF provenance + TXT page markers
# ============================================================

def detect_page_marker(text: str) -> Optional[int]:
    """
    Detect page markers in TXT/Markdown-derived Docling JSON:
    PAGE 3, Page 3, page 3.
    """
    m = re.fullmatch(r"PAGE\s+(\d+)", normalize_space(text), flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def get_page_from_prov(item: Dict[str, Any]) -> Optional[int]:
    """
    Extract page number from direct PDF Docling provenance.
    """
    prov = item.get("prov", [])
    if isinstance(prov, list) and prov:
        page = prov[0].get("page_no")
        if isinstance(page, int):
            return page
    return None


def get_page(item: Dict[str, Any], current_page: Optional[int]) -> Optional[int]:
    """
    General page extraction.

    Priority:
    1. Direct PDF Docling provenance: prov.page_no
    2. TXT/Markdown/OCR page marker: current PAGE n
    3. None
    """
    page = get_page_from_prov(item)
    if page is not None:
        return page
    return current_page


# ============================================================
# Docling structure inspection
# ============================================================

def build_docling_index(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build reference index:
    "#/texts/12" -> text object
    "#/groups/3" -> group object
    "#/tables/0" -> table object
    "#/pictures/0" -> picture object
    """
    index: Dict[str, Dict[str, Any]] = {}

    for name in ["texts", "groups", "tables", "pictures"]:
        collection = doc.get(name, [])
        if not isinstance(collection, list):
            continue

        for obj in collection:
            if isinstance(obj, dict) and "self_ref" in obj:
                index[obj["self_ref"]] = obj

    if isinstance(doc.get("body"), dict):
        index["#/body"] = doc["body"]

    if isinstance(doc.get("furniture"), dict):
        index["#/furniture"] = doc["furniture"]

    return index


def get_parent_ref(obj: Dict[str, Any]) -> Optional[str]:
    parent = obj.get("parent")
    if isinstance(parent, dict):
        ref = parent.get("$ref")
        if isinstance(ref, str):
            return ref
    return None


def get_ancestor_refs(obj: Dict[str, Any], index: Dict[str, Dict[str, Any]]) -> List[str]:
    refs = []
    current = obj
    seen = set()

    while True:
        parent_ref = get_parent_ref(current)
        if not parent_ref or parent_ref in seen:
            break

        refs.append(parent_ref)
        seen.add(parent_ref)

        parent_obj = index.get(parent_ref)
        if not parent_obj:
            break

        current = parent_obj

    return refs


def is_inside_table_or_local_area(item: Dict[str, Any], index: Dict[str, Dict[str, Any]]) -> bool:
    """
    General structural filter:
    remove headings inside tables, form/key-value areas, rich-cell groups,
    or pictures because they are usually local labels, captions, or visual content.
    """
    for ref in get_ancestor_refs(item, index):
        if ref.startswith("#/tables/"):
            return True

        if ref.startswith("#/pictures/"):
            return True

        ancestor = index.get(ref, {})
        label = ancestor.get("label")
        name = ancestor.get("name", "")

        if label in {"key_value_area", "form_area"}:
            return True

        if isinstance(name, str) and name.startswith("rich_cell_group"):
            return True

    return False


# ============================================================
# Heading detection
# ============================================================

def is_image_placeholder(text: str) -> bool:
    return bool(re.fullmatch(r"img-\d+\.(jpeg|jpg|png|webp)", text.lower()))


def is_page_number_noise(text: str) -> bool:
    t = normalize_space(text).lower()
    return bool(
        re.fullmatch(r"page\s+\d+(\s+sur\s+\d+)?", t)
        or re.fullmatch(r"\d+", t)
    )


def is_repeated_footer_like(text: str) -> bool:
    """
    Generic footer detector.
    """
    t = normalize_space(text).lower()
    t_no_accents = strip_accents(t)

    if "isbn" in t:
        return True

    if "tous droits réservés" in t or "tous droits reserves" in t_no_accents:
        return True

    if "version" in t and re.search(r"\d{4}|\d+\.\d+", t):
        return True

    return False


def extract_heading_info(text: str) -> Optional[Dict[str, Any]]:
    """
    Detect explicit heading patterns.

    Important:
    - title keeps the heading exactly as visible in the document.
    - number stores the extracted number separately.
    - clean_title stores the heading without the number for internal filtering only.
    """
    text = normalize_space(text)

    if not text:
        return None

    # PAGE marker
    page = detect_page_marker(text)
    if page is not None:
        return {
            "kind": "page_marker",
            "page": page,
        }

    if is_image_placeholder(text):
        return None

    # Decimal heading:
    # 1 Title
    # 1.1 Title
    # 1.2.3 Title
    # 1.2.3. Title
    m = re.match(r"^(\d+(?:\.\d+)*\.?)\s+(.+)$", text)
    if m:
        number = m.group(1).rstrip(".")
        clean_title = normalize_space(m.group(2))

        return {
            "kind": "heading",
            "number": number,
            "numbering_type": "decimal",
            "title": text,              # keep visible title with number
            "clean_title": clean_title, # internal only
            "depth": len(number.split(".")),
        }

    # Roman heading:
    # I - Title
    # II. Title
    # IV: Title
    m = re.match(r"^([IVXLCDM]+)\s*[\.\-\:]\s+(.+)$", text, flags=re.IGNORECASE)
    if m:
        roman = m.group(1).upper()
        clean_title = normalize_space(m.group(2))

        return {
            "kind": "heading",
            "number": roman,
            "numbering_type": "roman",
            "title": text,              # keep visible title with Roman prefix
            "clean_title": clean_title, # internal only
            "depth": 1,
            "roman_index": roman_to_int(roman),
        }

    # Bullet heading:
    # - Cheminements
    m = re.match(r"^\-\s+(.+)$", text)
    if m:
        clean_title = normalize_space(m.group(1))

        return {
            "kind": "heading",
            "number": None,
            "numbering_type": "bullet",
            "title": text,              # keep visible bullet
            "clean_title": clean_title, # internal only
            "depth": 99,
        }

    return None


def is_generic_toc_heading(title: str, number: Optional[str]) -> bool:
    """
    Detect generic table-of-contents headings.
    Use clean_title when available.
    """
    if number is not None:
        return False

    t = strip_accents(normalize_space(title).lower()).strip(" :;.")

    return t in {
        "sommaire",
        "table des matieres",
        "table de matieres",
        "table of contents",
        "contents",
        "summary",
        "toc",
    }


def is_candidate_heading(
    item: Dict[str, Any],
    index: Dict[str, Dict[str, Any]],
    exclude_table_like_areas: bool = True,
) -> bool:
    """
    General candidate filter.

    Accepts:
    - Docling section_header
    - Docling title
    - explicit numbered / Roman / bullet headings even if label is text
    """
    text = normalize_space(item.get("text", ""))
    label = item.get("label")

    if not text:
        return False

    if is_image_placeholder(text):
        return False

    if is_page_number_noise(text):
        return False

    if is_repeated_footer_like(text):
        return False

    if item.get("content_layer") == "furniture":
        return False

    if exclude_table_like_areas and is_inside_table_or_local_area(item, index):
        return False

    info = extract_heading_info(text)
    if info and info.get("kind") == "heading":
        return True

    if label in {"section_header", "title"}:
        return True

    return False


def make_section_id(number: Optional[str], title: str, idx: int, used_ids: Set[str]) -> str:
    base = title
    base_id = slugify(base, fallback=f"section-{idx}")
    return make_unique_id(base_id, used_ids)


# ============================================================
# Flat extraction
# ============================================================

def extract_flat_sections(
    doc: Dict[str, Any],
    exclude_table_like_areas: bool = True,
) -> List[Dict[str, Any]]:
    texts = doc.get("texts", [])
    if not isinstance(texts, list):
        return []

    index = build_docling_index(doc)
    used_ids: Set[str] = set()
    flat: List[Dict[str, Any]] = []

    current_page: Optional[int] = None

    for idx, item in enumerate(texts):
        if not isinstance(item, dict):
            continue

        text = normalize_space(item.get("text", ""))

        # TXT/Markdown mode: update PAGE marker
        page_marker = detect_page_marker(text)
        if page_marker is not None:
            current_page = page_marker
            continue

        if not is_candidate_heading(
            item,
            index=index,
            exclude_table_like_areas=exclude_table_like_areas,
        ):
            continue

        info = extract_heading_info(text)

        # If no explicit numbering but Docling says title/section_header
        if not info or info.get("kind") != "heading":
            info = {
                "kind": "heading",
                "number": None,
                "numbering_type": None,
                "title": text,        # keep as visible
                "clean_title": text,  # same for filtering
                "depth": item.get("level") if isinstance(item.get("level"), int) else 1,
            }

        number = info.get("number")
        title = normalize_space(info.get("title", ""))
        clean_title = normalize_space(info.get("clean_title", title))
        numbering_type = info.get("numbering_type")

        if not title:
            continue

        if is_generic_toc_heading(clean_title, number):
            continue

        page = get_page(item, current_page=current_page)

        sec = {
            "id": make_section_id(number, title, idx, used_ids),
            "number": number,
            "title": title,   # full visible title, not normalized by removing number
            "role": "heading" if numbering_type != "bullet" else "bullet_heading",
            "start_page": page,
            "end_page": page,
            "sections": [],

            # private fields
            "_order": idx,
            "_depth": info.get("depth", 1),
            "_numbering_type": numbering_type,
            "_roman_index": info.get("roman_index"),
        }

        flat.append(sec)

    return flat


# ============================================================
# Hierarchy construction
# ============================================================

def parent_decimal_number(number: str) -> Optional[str]:
    parts = number.split(".")
    if len(parts) <= 1:
        return None
    return ".".join(parts[:-1])


def build_hierarchy(flat: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    General hierarchy builder.

    Rules:
    - Roman headings are top-level.
    - Decimal 1.1 attaches under Roman I when Roman I exists.
    - Decimal 2.1 attaches under Roman II when Roman II exists.
    - Deeper decimal headings attach by numeric prefix.
    - Bullet headings attach to the most recent structural heading.
    - Unnumbered headings remain top-level unless they are children by Docling level.
    """
    roots: List[Dict[str, Any]] = []

    by_decimal: Dict[str, Dict[str, Any]] = {}
    by_roman_index: Dict[int, Dict[str, Any]] = {}

    unnumbered_stack: List[Tuple[int, Dict[str, Any]]] = []
    last_structural_heading: Optional[Dict[str, Any]] = None

    for sec in flat:
        ntype = sec.get("_numbering_type")
        number = sec.get("number")
        depth = sec.get("_depth", 1)

        # Roman top-level
        if ntype == "roman":
            roots.append(sec)
            roman_index = sec.get("_roman_index")
            if isinstance(roman_index, int):
                by_roman_index[roman_index] = sec
            last_structural_heading = sec
            unnumbered_stack = [(depth, sec)]
            continue

        # Decimal hierarchy
        if ntype == "decimal" and number:
            parent_num = parent_decimal_number(number)

            if parent_num and parent_num in by_decimal:
                by_decimal[parent_num]["sections"].append(sec)
            else:
                first_part = number.split(".")[0]
                try:
                    roman_parent_index = int(first_part)
                except ValueError:
                    roman_parent_index = None

                if roman_parent_index in by_roman_index:
                    by_roman_index[roman_parent_index]["sections"].append(sec)
                else:
                    roots.append(sec)

            by_decimal[number] = sec
            last_structural_heading = sec

            # Important: unnumbered headings cannot capture numbered headings
            unnumbered_stack = [(depth, sec)]
            continue

        # Bullet headings attach to nearest structural parent
        if ntype == "bullet":
            if last_structural_heading is not None:
                last_structural_heading["sections"].append(sec)
            else:
                roots.append(sec)
            continue

        # Plain unnumbered heading
        while unnumbered_stack and unnumbered_stack[-1][0] >= depth:
            unnumbered_stack.pop()

        if unnumbered_stack:
            unnumbered_stack[-1][1]["sections"].append(sec)
        else:
            roots.append(sec)

        unnumbered_stack.append((depth, sec))
        last_structural_heading = sec

    return roots


# ============================================================
# Optional cleanup: remove isolated unnumbered headings inside numbered body
# ============================================================

def remove_isolated_unnumbered_inside_numbered_body(flat: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    General filter:
    if a document has numbered headings, isolated unnumbered headings that appear
    between numbered sections are often local labels rather than global outline nodes.

    This does NOT remove Roman/decimal/bullet headings.
    """
    numbered_positions = [
        i for i, sec in enumerate(flat)
        if sec.get("_numbering_type") in {"decimal", "roman"}
    ]

    if not numbered_positions:
        return flat

    first_num = min(numbered_positions)
    last_num = max(numbered_positions)

    cleaned = []

    for i, sec in enumerate(flat):
        ntype = sec.get("_numbering_type")

        if ntype in {"decimal", "roman", "bullet"}:
            cleaned.append(sec)
            continue

        if i < first_num or i > last_num:
            cleaned.append(sec)
            continue

        # Inside numbered body: drop isolated unnumbered headings
        continue

    return cleaned


# ============================================================
# Page span and cleanup
# ============================================================

def iter_sections(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def walk(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            out.append(node)
            walk(node.get("sections", []))

    walk(sections)
    return out


def assign_end_pages(sections: List[Dict[str, Any]], total_pages: Optional[int] = None) -> None:
    flat = iter_sections(sections)

    for i, sec in enumerate(flat):
        start = sec.get("start_page")

        if not isinstance(start, int):
            sec["end_page"] = None
            continue

        next_start = None
        for nxt in flat[i + 1:]:
            if isinstance(nxt.get("start_page"), int):
                next_start = nxt["start_page"]
                break

        if next_start is not None:
            sec["end_page"] = max(start, next_start - 1)
        elif isinstance(total_pages, int):
            sec["end_page"] = max(start, total_pages)
        else:
            sec["end_page"] = start

    def post_order(nodes: List[Dict[str, Any]]) -> None:
        for n in nodes:
            post_order(n.get("sections", []))
            child_ends = [
                c.get("end_page")
                for c in n.get("sections", [])
                if isinstance(c.get("end_page"), int)
            ]
            if child_ends:
                n["end_page"] = max(child_ends)

    post_order(sections)


def cleanup(sections: List[Dict[str, Any]]) -> None:
    private_keys = {"_order", "_depth", "_numbering_type", "_roman_index"}

    for sec in sections:
        for key in private_keys:
            sec.pop(key, None)
        cleanup(sec.get("sections", []))


# ============================================================
# Main converter
# ============================================================

def convert_docling_json_to_outline(
    input_json: str | Path,
    output_json: str | Path,
    document_title: Optional[str] = None,
    total_pages: Optional[int] = None,
    exclude_table_like_areas: bool = True,
    remove_isolated_unnumbered: bool = True,
) -> Dict[str, Any]:
    input_json = Path(input_json)
    output_json = Path(output_json)

    with input_json.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    title = (
        document_title
        or doc.get("name")
        or doc.get("origin", {}).get("filename")
        or input_json.stem
    )

    flat = extract_flat_sections(
        doc,
        exclude_table_like_areas=exclude_table_like_areas,
    )

    if remove_isolated_unnumbered:
        flat = remove_isolated_unnumbered_inside_numbered_body(flat)

    hierarchy = build_hierarchy(flat)
    assign_end_pages(hierarchy, total_pages=total_pages)
    cleanup(hierarchy)

    outline = {
        "title": title,
        "sections": hierarchy,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(outline, f, ensure_ascii=False, indent=2)

    return outline


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Docling JSON file to an outline JSON."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to input Docling JSON file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to output outline JSON file.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional document title override.",
    )
    parser.add_argument(
        "--total-pages",
        type=int,
        default=None,
        help="Optional total number of pages for end_page propagation.",
    )
    parser.add_argument(
        "--include-table-like-areas",
        action="store_true",
        help="Include headings inside table/form/key-value/picture-related areas.",
    )
    parser.add_argument(
        "--keep-isolated-unnumbered",
        action="store_true",
        help="Keep isolated unnumbered headings found between numbered sections.",
    )
    args = parser.parse_args()

    outline = convert_docling_json_to_outline(
        input_json=args.input,
        output_json=args.output,
        document_title=args.title,
        total_pages=args.total_pages,
        exclude_table_like_areas=not args.include_table_like_areas,
        remove_isolated_unnumbered=not args.keep_isolated_unnumbered,
    )

    print("Saved:", args.output)
    print("Top-level sections:", len(outline["sections"]))
