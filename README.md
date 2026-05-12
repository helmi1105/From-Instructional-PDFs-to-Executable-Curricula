# ECG Project

End-to-end pipeline to extract a document outline from PDF/OCR, build an ECG graph, and evaluate structural quality and reproducibility.

## Project layout

- `outline_only_pipeline.py`: PDF/OCR -> `outline_clean.json`
- `build_clean_ecg_from_outline.py`: outline -> `ecg.json`
- `evaluate_ecg_against_gold.py`: prediction vs gold metrics
- `ecg_stability.py`: multi-run stability summary
- `inter_annotator_agreement.py`: agreement metrics for annotation studies
- `structural_error_report.py`: structural error reports + heatmap
- `graph.py`: render `ecg.dot` to PNG
- `error.py`: qualitative error-analysis figure
- `docling/`: Docling conversion utilities
- `toc/`: TOC extraction utilities (embedded or printed TOC)

## Reproducible setup

1. Create a virtual environment.
2. Install dependencies.
3. Set environment variables.

```powershell
cd project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set keys:

```powershell
$env:OPENAI_API_KEY="YOUR_OPENAI_KEY"
$env:MISTRAL_API_KEY="YOUR_MISTRAL_KEY"
```

## Core pipeline

From `project/`:

```powershell
python outline_only_pipeline.py `
  --pdf "C:\path\input.pdf" `
  --outdir "outputs\run1" `
  --provider openai

python build_clean_ecg_from_outline.py `
  --outline_json "outputs\run1\outline_clean.json" `
  --outdir "outputs\run1"

python evaluate_ecg_against_gold.py `
  --gold "gold_ecg_annotation_v1.json" `
  --pred "outputs\run1\ecg.json" `
  --outdir "eval_out\run1"
```

## TOC and Docling utilities

Extract TOC outline directly from PDF bookmarks/printed TOC:

```powershell
python toc\toc_outline.py `
  --pdf "C:\path\input.pdf" `
  --out_json "outputs\toc\outline.json"
```

Convert one PDF using Docling:

```powershell
python docling\run_docling_one.py `
  --pdf "C:\path\input.pdf" `
  --outdir "outputs\docling"
```

Convert Docling JSON to outline JSON:

```powershell
python docling\adapter.py `
  --input "outputs\docling\input_docling.json" `
  --output "outputs\docling\outline.json"
```

## Visualization

```powershell
python graph.py --dot "outputs\run1\ecg.dot" --outdir "outputs\run1\graphs"

python error.py `
  --diff_json "eval_out\run1\evaluation_results.json" `
  --outdir "eval_out\run1\paper_figures" `
  --out_name "figure_qualitative_error_analysis"
```

## Structural report

Default report (uses local sample pairs):

```powershell
python structural_error_report.py
```

Custom pairs:

```json
[
  {"name": "D1", "gold": "gold_ecg_annotation_v1.json", "pred": "outputs/run1/ecg.json"}
]
```

```powershell
python structural_error_report.py --pairs_json "pairs.json" --outdir "eval_out\structural_errors"
```

## Publishing checklist

- Keep generated artifacts outside version control (`outputs/`, `eval_out/`).
- Pin dependencies before release (`pip freeze > requirements-lock.txt`).
- Keep API keys in environment variables only.
- Record model names used for each experiment in output metadata.
