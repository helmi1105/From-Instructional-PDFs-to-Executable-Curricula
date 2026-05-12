import argparse
import json
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter
from docling.document_converter import PdfFormatOption


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one PDF to Docling JSON and Markdown.")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument(
        "--outdir",
        default="outputs/docling",
        help="Directory to write docling artifacts (default: outputs/docling)",
    )
    parser.add_argument(
        "--do_ocr",
        action="store_true",
        help="Enable OCR in Docling pipeline (disabled by default).",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = bool(args.do_ocr)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(str(pdf_path))
    doc = result.document

    md_path = out_dir / f"{pdf_path.stem}_docling.md"
    md_path.write_text(doc.export_to_markdown(), encoding="utf-8")

    json_path = out_dir / f"{pdf_path.stem}_docling.json"
    json_path.write_text(
        json.dumps(doc.export_to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Saved:")
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
