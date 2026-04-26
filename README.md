# ECG Project (Outline → ECG → Evaluation)

This folder contains the end-to-end pipeline to:
- extract a document outline from OCR,
- compile a clean ECG graph,
- evaluate against gold annotations,
- measure stability across repeated runs.

## 1) What is inside

- `outline_only_pipeline.py`  
  PDF/OCR to outline artifacts:
  - `ocr_raw.json`, `ocr_pages.json`, `ocr_markdown.txt`
  - `outline_raw.json`, `outline_repaired.json`, `outline_clean.json`
  - `outline_diff.json`, `token_usage.json`, `summary.json`

- `build_clean_ecg_from_outline.py`  
  Builds validated ECG from outline:
  - `ecg.json`, `validation.json`, `nodes.csv`, `edges.csv`, `ecg.dot`, `summary.json`

- `evaluate_ecg_against_gold.py`  
  Compares predicted ECG vs gold ECG:
  - `evaluation_results.json`, `evaluation_summary.md`

- `ecg_stability.py`  
  Computes multi-run stability (validation pass rate, node/edge mean+std, Dice overlaps).

- `baseline_direct_ecg.py`  
  Baseline that asks LLM to output ECG directly from OCR pages.

- `gold_ecg_*.json`  
  Gold ECG annotations for evaluation.

## 2) Architecture

Pipeline flow:
1. **OCR stage** (Mistral OCR): PDF pages to markdown/page objects.
2. **Outline stage** (OpenAI or Mistral LLM): extract structural hierarchy.
3. **Repair stage** (OpenAI or Mistral LLM): minimal structural fixes.
4. **Compile stage**: normalize outline, generate typed ECG nodes/edges, validate constraints.
5. **Evaluation stage**: compare prediction against gold metrics.
6. **Stability stage**: compare repeated runs for reproducibility.

## 3) Requirements

Use Python 3.10+.

Install dependencies:

```powershell
pip install mistralai openai matplotlib pillow graphviz
```

Graphviz binary is required for rendering DOT/PNG (if you use graph rendering scripts).

## 4) Environment variables

Set keys (PowerShell):

```powershell
$env:OPENAI_API_KEY="YOUR_OPENAI_KEY"
$env:MISTRAL_API_KEY="YOUR_MISTRAL_KEY"
```

Notes:
- OpenAI key is used when `--provider openai`.
- Mistral key is still required if OCR cache does not already exist (OCR is done by Mistral in this project pipeline).

## 5) Quick start

From repo root (`C:\Users\hbaaz\OneDrive\Bureau\PHD\ecg`):

### A) Build outline artifacts

```powershell
python project\outline_only_pipeline.py `
  --pdf "C:\path\your_document.pdf" `
  --outdir "C:\path\outputs\run1" `
  --provider openai
```

If `ocr_pages.json` and `ocr_markdown.txt` already exist in `--outdir`, OCR is skipped and cached outputs are reused.

### B) Build clean ECG

```powershell
python project\build_clean_ecg_from_outline.py `
  --outline_json "C:\path\outputs\run1\outline_clean.json" `
  --outdir "C:\path\outputs\run1"
```

### C) Evaluate against gold ECG

```powershell
python project\evaluate_ecg_against_gold.py `
  --gold "project\gold_ecg_sauvegarde_operationnelle_2024.json" `
  --pred "C:\path\outputs\run1\ecg.json" `
  --outdir "project\eval_out_run1"
```

### D) Stability across 5 runs

```powershell
python project\ecg_stability.py `
  --inputs `
    "C:\path\outputs\run1\ecg.json" `
    "C:\path\outputs\run2\ecg.json" `
    "C:\path\outputs\run3\ecg.json" `
    "C:\path\outputs\run4\ecg.json" `
    "C:\path\outputs\run5\ecg.json" `
  --out "project\outputs\stability_summary.json"
```

## 6) Optional: direct ECG baseline

```powershell
python project\baseline_direct_ecg.py `
  --pdf "C:\path\your_document.pdf" `
  --outdir "C:\path\outputs\baseline_direct" `
  --provider openai
```

## 7) Main outputs to inspect

- `outline_clean.json`: final cleaned hierarchy used for ECG compilation.
- `ecg.json`: compiled graph (nodes, edges, metadata, validation).
- `summary.json`: compact numeric summary.
- `token_usage.json`: token usage for extraction/repair (when available).
- `evaluation_results.json`: detailed metric breakdown vs gold.
- `stability_summary.json`: reproducibility metrics over repeated runs.

## 8) Troubleshooting

- **`Missing OPENAI_API_KEY`**  
  Set `$env:OPENAI_API_KEY` in the same terminal session.

- **`Missing MISTRAL_API_KEY (required for OCR)`**  
  Set `$env:MISTRAL_API_KEY`, or reuse an outdir with existing OCR cache files.

- **`RateLimitError` / `insufficient_quota`**  
  API account quota/billing issue for the selected provider.

- **File path with spaces fails**  
  Wrap paths in quotes: `"C:\path with spaces\file.pdf"`.

## 9) Reproducibility tips

- Keep `temperature=0` (already used in calls).
- Use the same model name across runs.
- Keep same PDF input and same prompt version.
- Save each run under a separate directory (`run1`, `run2`, ...).
