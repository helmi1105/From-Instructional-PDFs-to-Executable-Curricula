import argparse
import json
from pathlib import Path
import fitz
import time
from conversion_checks import conversion_complete
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

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
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError('Use a fresh conversion output directory')
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = bool(args.do_ocr)
    # Bound in-flight work; previous defaults buffered up to 100 pages per queue.
    pipeline_options.ocr_batch_size = 1
    pipeline_options.layout_batch_size = 1
    pipeline_options.table_batch_size = 1
    pipeline_options.queue_max_size = 2
    pipeline_options.accelerator_options = AcceleratorOptions(num_threads=1, device=AcceleratorDevice.CPU)
    with fitz.open(pdf_path) as source:
        expected_pages = set(range(1, len(source) + 1))
    report_path = out_dir / 'conversion_report.json'
    report = dict(status='running', expected_pages=sorted(expected_pages),
                  pipeline_options=pipeline_options.model_dump(mode='json'))
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    started = time.perf_counter()
    try:
        result = converter.convert(str(pdf_path), raises_on_error=False)
    except Exception as exc:
        report.update(status='exception', error=str(exc), seconds=time.perf_counter()-started)
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        raise
    processed = [p.page_no for p in result.pages]
    errors = [e.model_dump(mode='json') for e in result.errors]
    complete = conversion_complete(result.status.value, errors, processed, expected_pages)
    report.update(status=result.status.value, complete=complete, processed_pages=processed,
                  missing_pages=sorted(expected_pages-set(processed)), errors=errors,
                  seconds=time.perf_counter()-started)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not complete:
        # Preserve diagnostics, but never export a partial document as a usable baseline.
        raise RuntimeError(f'Incomplete Docling conversion: {len(processed)}/{len(expected_pages)} pages; see {report_path}')
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
