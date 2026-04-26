import os
import json
import argparse
from typing import Any, Dict, List, Tuple

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


def parse_json_or_raise(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Model did not return valid JSON: {e}\n\n{text[:2000]}")


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
# Prompt
# ------------------------------------------------------------
DIRECT_ECG_RULES = r"""
You are a curriculum-structure extraction assistant.

Your task is to read OCR pages from one instructional PDF and directly produce
a final Executable Curriculum Graph (ECG) as JSON.

The ECG must represent the document as a page-grounded curriculum structure,
not as flat text.

General principles:
- Return ONLY valid JSON.
- Do not include markdown fences.
- Do not explain anything outside the JSON.
- Use only structure supported by the OCR pages.
- Do not invent titles, pages, nodes, numbering, or hierarchy.
- Ignore running headers, footers, decorative text, page numbers, and ordinary body text.
- Extract only real document structure: title, instructional headings, subheadings, and terminal instructional units.
- If hierarchy is unclear, prefer a conservative shallower structure rather than inventing unsupported nesting.

ECG representation:
- There must be exactly one root node of type COURSE.
- Top-level document sections with children should be represented as MODULE nodes.
- Internal non-leaf sections below a MODULE or UNIT should be represented as UNIT nodes.
- Terminal instructional units (leaf nodes) must be represented as KC nodes.
- All leaf nodes should be KC.
- Parent-child hierarchy must be represented with contains edges.
- Local order between adjacent siblings under the same parent must be represented with sequence edges.

Grounding:
- Every node must remain grounded in the source document.
- Each node should include:
  - title
  - number (if visibly present, otherwise null)
  - role (if clearly inferable, otherwise null)
  - page_start
  - page_end
  - evidence_pages
  - path
- page_start and page_end should reflect the node span as supported by OCR.
- evidence_pages should list the page numbers supporting the node.
- path should be the hierarchical title path from the course title to the node title.

Conservativeness rules:
- Keep numbering exactly when visible; otherwise use null.
- Do not merge distinct visible headings unless they are clearly duplicates.
- Do not split one heading into multiple nodes unless the OCR clearly supports it.
- If a heading is visible but its depth is uncertain, attach it at the nearest safe level.
- Prefer false split over false merge only when the OCR strongly suggests distinct headings.
- If a title is repeated as a header/footer, do not treat it as a structural node unless supported by content.

Typing rules:
- COURSE: the document root.
- MODULE: top-level section with children.
- UNIT: internal non-leaf section below the top level.
- KC: terminal instructional unit with no children.

Edge rules:
- contains edges encode hierarchy.
- sequence edges connect adjacent sibling nodes under the same parent only.
- Do not create sequence edges across different parents.
- Do not create duplicate edges.

ID rules:
- Create stable readable ids when possible.
- Each node id must be unique.

Return this JSON schema exactly:
{
  "metadata": {
    "source": "direct_llm_baseline",
    "course_title": "<string>"
  },
  "nodes": [
    {
      "id": "<string>",
      "label": "<string>",
      "kind": "COURSE|MODULE|UNIT|KC",
      "title": "<string>",
      "number": "<string or null>",
      "role": "<string or null>",
      "page_start": <int or null>,
      "page_end": <int or null>,
      "evidence_pages": [<int>, ...],
      "path": ["<title1>", "<title2>", ...]
    }
  ],
  "edges": [
    {
      "source": "<node_id>",
      "target": "<node_id>",
      "type": "contains|sequence"
    }
  ]
}
"""


# ------------------------------------------------------------
# LLM call
# ------------------------------------------------------------
def ask_json_mistral(client: Mistral, model: str, prompt: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    resp = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = extract_text_content(resp)
    usage = to_plain(getattr(resp, "usage", None)) or {}
    return parse_json_or_raise(text), usage


def ask_json_openai(client: Any, model: str, prompt: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = extract_text_content_openai(resp)
    usage = to_plain(getattr(resp, "usage", None)) or {}
    return parse_json_or_raise(text), usage


def generate_direct_ecg(
    client: Any,
    model: str,
    pages: List[Dict[str, Any]],
    provider: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    text_blob = pages_to_markdown_blob(pages)
    prompt = f"{DIRECT_ECG_RULES}\n\nOCR PAGES:\n{text_blob}"
    if provider == "openai":
        return ask_json_openai(client, model, prompt)
    return ask_json_mistral(client, model, prompt)


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
def run_pipeline(pdf_path: str, outdir: str, chat_model: str, provider: str) -> None:
    ensure_dir(outdir)

    ocr_raw_path = os.path.join(outdir, "ocr_raw.json")
    ocr_pages_path = os.path.join(outdir, "ocr_pages.json")
    ocr_markdown_path = os.path.join(outdir, "ocr_markdown.txt")

    if provider == "openai":
        if OpenAI is None:
            raise RuntimeError("OpenAI SDK not installed. pip install openai")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        llm_client = OpenAI(api_key=api_key)
    else:
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("Missing MISTRAL_API_KEY")
        llm_client = Mistral(api_key=api_key)

    if os.path.exists(ocr_pages_path) and os.path.exists(ocr_markdown_path):
        print("[1/4] Using existing OCR outputs")
        pages = load_json(ocr_pages_path)
    elif os.path.exists(ocr_raw_path):
        print("[1/4] Using existing OCR raw output")
        ocr_response = load_json(ocr_raw_path)
        pages = ocr_pages_to_dicts(ocr_response)
        write_json(ocr_pages_path, pages)
        write_text(ocr_markdown_path, pages_to_markdown_blob(pages))
    else:
        ocr_key = os.environ.get("MISTRAL_API_KEY")
        if not ocr_key:
            raise RuntimeError("Missing MISTRAL_API_KEY (required for OCR)")
        ocr_client = Mistral(api_key=ocr_key)

        print("[1/4] Uploading PDF")
        uploaded = upload_pdf_for_ocr(ocr_client, pdf_path)
        uploaded_plain = to_plain(uploaded)
        write_json(os.path.join(outdir, "uploaded_file.json"), uploaded_plain)

        file_id = uploaded_plain.get("id")
        if not file_id:
            raise RuntimeError("Could not retrieve uploaded file id.")

        print("[2/4] Getting signed URL")
        signed_url = get_signed_url(ocr_client, file_id)

        print("[3/4] Running OCR")
        ocr_response = run_ocr(ocr_client, signed_url)
        write_json(ocr_raw_path, to_plain(ocr_response))

        pages = ocr_pages_to_dicts(ocr_response)
        write_json(ocr_pages_path, pages)
        write_text(ocr_markdown_path, pages_to_markdown_blob(pages))

    print("[4/4] LLM directly outputs ECG")
    ecg_raw, usage = generate_direct_ecg(llm_client, chat_model, pages, provider)

    write_json(os.path.join(outdir, "ecg_raw_direct.json"), ecg_raw)
    write_json(os.path.join(outdir, "token_usage.json"), {
        "provider": provider,
        "chat_model": chat_model,
        "usage": usage,
    })

    print("\nDone.")
    print(f"Output folder: {outdir}")
    print("Saved: ecg_raw_direct.json")


def main():
    parser = argparse.ArgumentParser(description="PDF -> OCR pages -> LLM directly outputs ECG")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "mistral"), help="mistral or openai")
    parser.add_argument("--chat_model", default=None, help="Chat model (provider-specific)")
    args = parser.parse_args()

    provider = (args.provider or "mistral").lower()
    if args.chat_model:
        chat_model = args.chat_model
    else:
        if provider == "openai":
            chat_model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4.1")
        else:
            chat_model = os.environ.get("MISTRAL_CHAT_MODEL", "mistral-large-latest")

    run_pipeline(
        pdf_path=args.pdf,
        outdir=args.outdir,
        chat_model=chat_model,
        provider=provider,
    )


if __name__ == "__main__":
    main()
