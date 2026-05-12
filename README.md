# From Instructional PDFs to Executable Curricula

Core codebase for extracting structured outlines from instructional PDFs and compiling them into ECG graphs.

## Included in this repository

- `outline_only_pipeline.py`: PDF/OCR -> cleaned outline JSON
- `build_clean_ecg_from_outline.py`: cleaned outline -> ECG graph JSON/DOT/CSV
- `baseline_direct_ecg.py`: direct ECG baseline from OCR pages
- `ecg_stability.py`: run-to-run stability metrics
- `evaluate_ecg_against_gold.py`: optional evaluation script (requires your own gold files)
- `structural_error_report.py`: optional structural error reporting
- `graph.py`: render DOT graph to PNG
- `error.py`: qualitative figure generation from evaluation outputs
- `docling/`
  - `run_docling_one.py`: convert PDF to Docling JSON/Markdown
  - `adapter.py`: convert Docling JSON to outline JSON
- `toc/`
  - `toc_outline.py`: extract TOC outline from embedded or printed TOC

 

## Environment

- Python 3.10+

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set API keys (PowerShell):

```powershell
$env:OPENAI_API_KEY="YOUR_OPENAI_KEY"
$env:MISTRAL_API_KEY="YOUR_MISTRAL_KEY"
```

## Quick start

### 1) Extract outline from PDF

```powershell
python outline_only_pipeline.py `
  --pdf "C:\path\input.pdf" `
  --outdir "outputs\run1" `
  --provider openai
```

### 2) Build clean ECG

```powershell
python build_clean_ecg_from_outline.py `
  --outline_json "outputs\run1\outline_clean.json" `
  --outdir "outputs\run1"
```

### 3) Render graph

```powershell
python graph.py --dot "outputs\run1\ecg.dot" --outdir "outputs\run1\graphs"
```

## TOC workflow (optional)

```powershell
python toc\toc_outline.py `
  --pdf "C:\path\input.pdf" `
  --out_json "outputs\toc\outline.json"
```

## Docling workflow (optional)

```powershell
python docling\run_docling_one.py `
  --pdf "C:\path\input.pdf" `
  --outdir "outputs\docling"

python docling\adapter.py `
  --input "outputs\docling\input_docling.json" `
  --output "outputs\docling\outline.json"
```

## Reproducibility notes

- Keep model and provider fixed across runs.
- Keep prompts/versioned code fixed across runs.
- Save each run in a separate output folder (`run1`, `run2`, ...).
- Do not commit generated artifacts.
