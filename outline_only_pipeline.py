import os
import re
import json
import argparse
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------
# Mistral import compatibility + OpenAI
# ------------------------------------------------------------
try:
    from mistralai import Mistral
except Exception:
    from mistralai.client import Mistral
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def to_plain(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [to_plain(x) for x in obj]
    if isinstance(obj, tuple):
        return [to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return to_plain(obj.model_dump())
    if hasattr(obj, "__dict__"):
        return {str(k): to_plain(v) for k, v in vars(obj).items()}
    return str(obj)


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\u00a0", " ").split()).strip()


def normalize_key(value: str) -> str:
    value = clean_text(value).lower().replace("’", "'")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_plain(data), f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_text_content(resp: Any) -> str:
    try:
        msg = resp.choices[0].message
    except Exception as e:
        raise RuntimeError(f"Unexpected chat response structure: {e}")

    content = getattr(msg, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()

    return str(content).strip()


def extract_text_content_openai(resp: Any) -> str:
    try:
        text = getattr(resp, "output_text", "")
        if isinstance(text, str):
            return text.strip()
    except Exception:
        pass
    return ""


# ------------------------------------------------------------
# OCR
# ------------------------------------------------------------
def upload_pdf_for_ocr(client: Mistral, pdf_path: str):
    with open(pdf_path, "rb") as f:
        uploaded = client.files.upload(
            file={
                "file_name": os.path.basename(pdf_path),
                "content": f,
            },
            purpose="ocr",
        )
    return uploaded


def get_signed_url(client: Mistral, file_id: str) -> str:
    resp = client.files.get_signed_url(file_id=file_id)
    data = to_plain(resp)
    url = data.get("url")
    if not url:
        raise RuntimeError("Could not retrieve signed URL.")
    return url


def run_ocr(client: Mistral, signed_url: str):
    return client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url,
        },
        include_image_base64=False,
    )


def ocr_pages_to_dicts(ocr_response: Any) -> List[Dict[str, Any]]:
    data = to_plain(ocr_response)
    pages = data.get("pages", []) or []

    indices = [p.get("index") for p in pages if isinstance(p.get("index"), int)]
    offset = 1 if indices and min(indices) == 0 else 0

    out = []
    for i, p in enumerate(pages, start=1):
        page_number = p.get("index", i)
        if isinstance(page_number, int):
            page_number = page_number + offset
        out.append(
            {
                "page_number": page_number,
                "markdown": p.get("markdown", "") or "",
                "images": p.get("images", []) or [],
                "dimensions": p.get("dimensions", {}) or {},
            }
        )
    return out


def pages_to_markdown_blob(pages: List[Dict[str, Any]]) -> str:
    chunks = []
    for p in pages:
        chunks.append(f"\n\n# PAGE {p['page_number']}\n\n{p.get('markdown', '')}")
    return "\n".join(chunks).strip()


# ------------------------------------------------------------
# Prompting
# ------------------------------------------------------------
OUTLINE_RULES = r"""
You are an outline extraction assistant.

Extract a clean global document outline from OCR pages.

Rules:
- Use only visible section titles and instructional headings.
- Do not invent titles, levels, or missing content.
- Ignore page numbers, headers, footers, decorative text, and body text.
- Preserve hierarchy only when clearly visible from numbering, layout, or typography.
- If hierarchy is unclear, keep headings at the same level.
- Keep numbering when visible (Roman or decimal), otherwise null.
- Return only JSON.

JSON format:
{
  "title": "<document title>",
  "sections": [
    {
      "title": "<section title>",
      "number": "<section number or null>",
      "role": null,
      "start_page": <int or null>,
      "end_page": <int or null>,
      "sections": [...]
    }
  ]
}
"""

REPAIR_RULES = r"""
You are a repair assistant.

Repair the outline using the OCR context and the current extracted outline.

Rules:
- Check each section in context: consider its title, its parent section, and its subsections.
- Keep a section only if it is structurally necessary and supported by OCR.
- Remove unnecessary, duplicated, noisy, misplaced, or non-structural sections or subsections.
- Add a missing section or subsection only when clearly supported by OCR and necessary for the structure.
- If a section is not a real container, remove it or keep its useful subsections under the correct parent.
- Fix the hierarchy when it is clearly wrong.
- Keep correct titles and structure unchanged.
- Do not invent new content, titles, or unsupported hierarchy.
- If evidence is weak, keep the current structure.
- Prefer minimal changes.
- Return only JSON with the same format.
"""


def _usage_from_mistral(resp: Any) -> Dict[str, Any]:
    usage = getattr(resp, "usage", None)
    usage = to_plain(usage) if usage is not None else {}
    if not isinstance(usage, dict):
        usage = {}
    return {
        "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
        "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _usage_from_openai(resp: Any) -> Dict[str, Any]:
    usage = getattr(resp, "usage", None)
    usage = to_plain(usage) if usage is not None else {}
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def ask_json_mistral(client: Mistral, model: str, prompt: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    resp = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = extract_text_content(resp)
    return _parse_json_or_raise(text), _usage_from_mistral(resp)


def ask_json_openai(client: Any, model: str, prompt: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = extract_text_content_openai(resp)
    return _parse_json_or_raise(text), _usage_from_openai(resp)


def _parse_json_or_raise(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise RuntimeError("Model did not return valid JSON.")
        return json.loads(match.group(0))

def extract_outline(client: Any, model: str, pages: List[Dict[str, Any]], provider: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    text_blob = pages_to_markdown_blob(pages)
    prompt = f"{OUTLINE_RULES}\n\nOCR PAGES:\n{text_blob}"
    if provider == "openai":
        return ask_json_openai(client, model, prompt)
    return ask_json_mistral(client, model, prompt)


def repair_outline_missing_children(
    client: Any,
    model: str,
    raw_outline: Dict[str, Any],
    pages: List[Dict[str, Any]],
    provider: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    text_blob = pages_to_markdown_blob(pages)
    prompt = (
        f"{REPAIR_RULES}\n\n"
        f"CURRENT OUTLINE:\n{json.dumps(raw_outline, ensure_ascii=False, indent=2)}\n\n"
        f"OCR PAGES:\n{text_blob}"
    )
    if provider == "openai":
        return ask_json_openai(client, model, prompt)
    return ask_json_mistral(client, model, prompt)


# ------------------------------------------------------------
# Outline normalization only
# ------------------------------------------------------------
def normalize_section(section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    title = clean_text(section.get("title"))
    if not title:
        return None

    number = clean_text(section.get("number")) or None
    role = clean_text(section.get("role")) or None

    start_page = section.get("start_page")
    end_page = section.get("end_page")
    if not isinstance(start_page, int):
        start_page = None
    if not isinstance(end_page, int):
        end_page = None

    children_out: List[Dict[str, Any]] = []
    seen = set()

    for child in section.get("sections", []) or []:
        child_norm = normalize_section(child)
        if not child_norm:
            continue

        key = (
            normalize_key(child_norm.get("number") or ""),
            normalize_key(child_norm.get("title") or ""),
            child_norm.get("start_page"),
            child_norm.get("end_page"),
        )
        if key in seen:
            continue
        seen.add(key)
        children_out.append(child_norm)

    return {
        "title": title,
        "number": number,
        "role": role,
        "start_page": start_page,
        "end_page": end_page,
        "sections": children_out,
    }


def recompute_spans_from_children(section: Dict[str, Any]) -> None:
    children = section.get("sections", []) or []

    for child in children:
        recompute_spans_from_children(child)

    own_start = section.get("start_page") if isinstance(section.get("start_page"), int) else None
    own_end = section.get("end_page") if isinstance(section.get("end_page"), int) else None

    child_starts = [c.get("start_page") for c in children if isinstance(c.get("start_page"), int)]
    child_ends = [c.get("end_page") for c in children if isinstance(c.get("end_page"), int)]

    starts = ([own_start] if own_start is not None else []) + child_starts
    ends = ([own_end] if own_end is not None else []) + child_ends

    section["start_page"] = min(starts) if starts else None
    section["end_page"] = max(ends) if ends else None

    if isinstance(section["start_page"], int) and isinstance(section["end_page"], int):
        if section["start_page"] > section["end_page"]:
            section["end_page"] = section["start_page"]


def normalize_outline(raw_outline: Dict[str, Any]) -> Dict[str, Any]:
    title = clean_text(raw_outline.get("title")) or "Untitled Course"

    sections_out: List[Dict[str, Any]] = []
    seen = set()

    for sec in raw_outline.get("sections", []) or []:
        sec_norm = normalize_section(sec)
        if not sec_norm:
            continue

        key = (
            normalize_key(sec_norm.get("number") or ""),
            normalize_key(sec_norm.get("title") or ""),
            sec_norm.get("start_page"),
            sec_norm.get("end_page"),
        )
        if key in seen:
            continue
        seen.add(key)
        sections_out.append(sec_norm)

    root_wrapper = {
        "title": title,
        "number": None,
        "role": None,
        "start_page": raw_outline.get("start_page") if isinstance(raw_outline.get("start_page"), int) else None,
        "end_page": raw_outline.get("end_page") if isinstance(raw_outline.get("end_page"), int) else None,
        "sections": sections_out,
    }
    recompute_spans_from_children(root_wrapper)

    return {
        "title": title,
        "start_page": root_wrapper.get("start_page"),
        "end_page": root_wrapper.get("end_page"),
        "sections": root_wrapper.get("sections", []),
    }


def build_summary(outline_clean: Dict[str, Any]) -> Dict[str, Any]:
    counts = {"sections_total": 0, "leaf_sections": 0, "max_depth": 0}

    def walk(section: Dict[str, Any], depth: int) -> None:
        counts["sections_total"] += 1
        counts["max_depth"] = max(counts["max_depth"], depth)
        children = section.get("sections", []) or []
        if not children:
            counts["leaf_sections"] += 1
        for child in children:
            walk(child, depth + 1)

    for sec in outline_clean.get("sections", []) or []:
        walk(sec, 1)

    return {
        "title": outline_clean.get("title"),
        "start_page": outline_clean.get("start_page"),
        "end_page": outline_clean.get("end_page"),
        "top_level_sections": len(outline_clean.get("sections", []) or []),
        "sections_total": counts["sections_total"],
        "leaf_sections": counts["leaf_sections"],
        "max_depth": counts["max_depth"],
    }


# ------------------------------------------------------------
# Outline diff
# ------------------------------------------------------------
def _collect_outline_paths(outline: Dict[str, Any]) -> List[str]:
    paths: List[str] = []

    def walk(node: Dict[str, Any], prefix: List[str]) -> None:
        title = clean_text(node.get("title"))
        if not title:
            return
        number = clean_text(node.get("number") or "")
        label = f"{number} {title}".strip() if number else title
        cur = prefix + [label]
        paths.append(" > ".join(cur))
        for child in node.get("sections", []) or []:
            walk(child, cur)

    for sec in outline.get("sections", []) or []:
        walk(sec, [])
    return paths


def build_outline_diff(raw_outline: Dict[str, Any], repaired_outline: Dict[str, Any]) -> Dict[str, Any]:
    raw_paths = set(_collect_outline_paths(raw_outline))
    repaired_paths = set(_collect_outline_paths(repaired_outline))
    added = sorted(repaired_paths - raw_paths)
    removed = sorted(raw_paths - repaired_paths)
    def count_nodes(outline: Dict[str, Any]) -> int:
        total = 0
        def walk(node: Dict[str, Any]) -> None:
            nonlocal total
            title = clean_text(node.get("title"))
            if title:
                total += 1
            for child in node.get("sections", []) or []:
                walk(child)
        for sec in outline.get("sections", []) or []:
            walk(sec)
        return total

    return {
        "added_paths": added,
        "removed_paths": removed,
        "added_count": len(added),
        "removed_count": len(removed),
        "raw_node_count": count_nodes(raw_outline),
        "repaired_node_count": count_nodes(repaired_outline),
        "node_count_delta": count_nodes(repaired_outline) - count_nodes(raw_outline),
    }


# ------------------------------------------------------------
# Run modes
# ------------------------------------------------------------
def build_from_existing_outline(outline_path: str, outdir: str) -> None:
    ensure_dir(outdir)

    print("[1/3] Loading outline JSON")
    raw_outline = load_json(outline_path)
    write_json(os.path.join(outdir, "outline_input.json"), raw_outline)

    print("[2/3] Normalizing outline")
    outline_clean = normalize_outline(raw_outline)
    write_json(os.path.join(outdir, "outline_clean.json"), outline_clean)

    print("[3/3] Writing summary")
    write_json(os.path.join(outdir, "summary.json"), build_summary(outline_clean))

    print("\nDone.")
    print(f"Output folder: {outdir}")
    print(f"Top-level sections: {len(outline_clean.get('sections', []))}")


def run_pipeline(
    pdf_path: str,
    outdir: str,
    chat_model: str,
    provider: str,
) -> None:
    ensure_dir(outdir)
    ocr_raw_path = os.path.join(outdir, "ocr_raw.json")
    ocr_pages_path = os.path.join(outdir, "ocr_pages.json")
    ocr_markdown_path = os.path.join(outdir, "ocr_markdown.txt")

    if provider == "openai":
        if OpenAI is None:
            raise RuntimeError("OpenAI SDK not installed. pip install openai")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing environment variable: OPENAI_API_KEY")
        llm_client = OpenAI(api_key=api_key)
    else:
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("Missing environment variable: MISTRAL_API_KEY")
        llm_client = Mistral(api_key=api_key)

    if os.path.exists(ocr_pages_path) and os.path.exists(ocr_markdown_path):
        print("[1/6] Using existing OCR outputs")
        pages = load_json(ocr_pages_path)
    elif os.path.exists(ocr_raw_path):
        print("[1/6] Using existing OCR raw output")
        ocr_response = load_json(ocr_raw_path)
        pages = ocr_pages_to_dicts(ocr_response)
        write_json(ocr_pages_path, pages)
        write_text(ocr_markdown_path, pages_to_markdown_blob(pages))
    else:
        # OCR still uses Mistral OCR
        ocr_key = os.environ.get("MISTRAL_API_KEY")
        if not ocr_key:
            raise RuntimeError("Missing environment variable: MISTRAL_API_KEY (required for OCR)")
        ocr_client = Mistral(api_key=ocr_key)

        print("[1/6] Uploading PDF")
        uploaded = upload_pdf_for_ocr(ocr_client, pdf_path)
        uploaded_plain = to_plain(uploaded)
        write_json(os.path.join(outdir, "uploaded_file.json"), uploaded_plain)

        file_id = uploaded_plain.get("id")
        if not file_id:
            raise RuntimeError("Could not retrieve uploaded file id.")

        print("[2/6] Getting signed URL")
        signed_url = get_signed_url(ocr_client, file_id)

        print("[3/6] Running OCR")
        ocr_response = run_ocr(ocr_client, signed_url)
        write_json(ocr_raw_path, to_plain(ocr_response))

        pages = ocr_pages_to_dicts(ocr_response)
        write_json(ocr_pages_path, pages)
        write_text(ocr_markdown_path, pages_to_markdown_blob(pages))

    print("[4/6] Extracting outline")
    raw_outline, outline_usage = extract_outline(llm_client, chat_model, pages, provider)
    write_json(os.path.join(outdir, "outline_raw.json"), raw_outline)

    print("[5/6] Repairing outline")
    repaired_outline, repair_usage = repair_outline_missing_children(llm_client, chat_model, raw_outline, pages, provider)
    write_json(os.path.join(outdir, "outline_repaired.json"), repaired_outline)
    write_json(os.path.join(outdir, "outline_diff.json"), build_outline_diff(raw_outline, repaired_outline))

    print("[6/6] Normalizing outline")
    outline_clean = normalize_outline(raw_outline)
    write_json(os.path.join(outdir, "outline_clean.json"), outline_clean)
    write_json(os.path.join(outdir, "summary.json"), build_summary(outline_clean))
    write_json(
        os.path.join(outdir, "token_usage.json"),
        {
            "provider": provider,
            "chat_model": chat_model,
            "steps": [
                {"step": "extract_outline", **outline_usage},
                {"step": "repair_outline_missing_children", **repair_usage},
            ],
        },
    )

    print("\nDone.")
    print(f"Output folder: {outdir}")
    print(f"Top-level sections: {len(outline_clean.get('sections', []))}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="PDF -> OCR -> outline_clean.json only, or normalize an existing outline JSON"
    )
    parser.add_argument("--pdf", help="Path to input PDF")
    parser.add_argument("--outline_json", help="Path to existing outline JSON")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "mistral"), help="mistral or openai")
    parser.add_argument("--chat_model", default=None, help="Chat model (provider-specific)")
    args = parser.parse_args()

    if not args.pdf and not args.outline_json:
        raise SystemExit("Provide either --pdf or --outline_json")

    provider = (args.provider or "mistral").lower()
    if args.chat_model:
        chat_model = args.chat_model
    else:
        if provider == "openai":
            chat_model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4.1")
        else:
            chat_model = os.environ.get("MISTRAL_CHAT_MODEL", "mistral-large-latest")

    if args.outline_json:
        build_from_existing_outline(args.outline_json, args.outdir)
    else:
        run_pipeline(
            pdf_path=args.pdf,
            outdir=args.outdir,
            chat_model=chat_model,
            provider=provider,
        )


if __name__ == "__main__":
    main()
