# From Instructional PDFs to Executable Curricula

Extract document outlines from PDF OCR and compile them into hierarchical ECG graphs.
Compare direct LLM generation with deterministic compilation, evaluate graphs against
external gold references, and run a controlled downstream navigation pilot.

This repository publishes code and tests only. PDFs, gold annotations, OCR caches,
API responses, experimental results and article tables are not distributed here.
Supply your own inputs. Local credentials and generated files are excluded by `.gitignore`.

The publication allowlist includes the active pipeline, shared evaluator, baseline adapters,
downstream navigation runner and regression tests. It excludes document-specific article
export scripts, local benchmark cases, frozen study plans and unfinished controller tests.
These excluded files can remain in your working directory without being uploaded.

## Main components

| Component | Files |
|---|---|
| OCR, extraction and optional repair | `outline_only_pipeline.py`, `revision_pipeline.py` |
| Deterministic compilation and validation | `build_clean_ecg_from_outline.py` |
| Direct graph generation and revision | `baseline_direct_ecg.py` |
| Tokens, raw responses, failures and provenance | `experiment_logging.py` |
| Gold evaluation and page decoding | `evaluate_ecg_against_gold.py`, `ecg_evaluation_core.py`, `page_grounding.py` |
| Repair analysis, statistics and stability | `experiment_analysis.py`, `ecg_stability.py` |
| TOC and Docling adapters | `toc/`, `docling/`, `run_secondary_baselines.py` |
| Structural diagnostics and visualization | `structural_error_report.py`, `graph.py`, `error.py` |
| Downstream navigation pilot | `ecg_downstream_eval/` |

The paired pipeline uses one extracted outline for both branches:

```text
PDF -> OCR -> extracted outline -> normalize -> compiler -> without_repair ECG
                            |
                            +-> repair -> normalize -> compiler -> with_repair ECG
```

## Installation

Use Python 3.10 or newer. In PowerShell, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The examples below assume `python` selects that environment. Docling can require model
downloads and substantial resources. Rendering DOT graphs also requires the Graphviz
system executable, separately from the Python package.

Configure keys only when using paid services:

```powershell
$env:OPENAI_API_KEY = "YOUR_OPENAI_KEY"
$env:MISTRAL_API_KEY = "YOUR_MISTRAL_KEY"
```

`.env.example` documents available variables; the scripts read process environment
variables and do not automatically load `.env` files. Fresh OCR uses Mistral even when
OpenAI handles extraction and repair. Offline evaluation needs no API keys.

## Generate paired ECGs

Choose a new output directory for every run:

```powershell
python outline_only_pipeline.py `
  --pdf "data/document.pdf" `
  --provider openai `
  --chat_model "gpt-4.1-2025-04-14" `
  --ocr_model "mistral-ocr-2505" `
  --repair both `
  --max_output_tokens 32768 `
  --max_retries 0 `
  --outdir "outputs/paired_run"
```

This makes paid OCR and LLM calls. Model availability depends on your provider account.
Use `--repair off` or `--repair on` for one branch, or `--ocr_pages` to reuse saved OCR.
The compiler already runs inside this command. Inspect:

- `outputs/paired_run/without_repair/ecg.json`
- `outputs/paired_run/with_repair/ecg.json`
- `outputs/paired_run/manifest.json` and saved call artifacts

Compile an existing outline without API calls:

```powershell
python outline_only_pipeline.py --outline_json "data/outline.json" --repair off --outdir "outputs/offline_compile"
```

## Direct baselines

Use the same OCR as the compiler comparison:

```powershell
python baseline_direct_ecg.py --ocr_pages "outputs/paired_run/ocr_pages.json" --provider openai --chat_model "gpt-4.1-2025-04-14" --variant original --max_output_tokens 32768 --max_retries 0 --outdir "outputs/direct"
python baseline_direct_ecg.py --ocr_pages "outputs/paired_run/ocr_pages.json" --provider openai --chat_model "gpt-4.1-2025-04-14" --variant revision --max_output_tokens 32768 --max_retries 0 --outdir "outputs/direct_revision"
```

These are paid calls. The revision variant generates a graph and then revises it;
its final graph is validated. Equal output caps do not imply equal input tokens,
actual usage, cost or compute. Preserve validation failures and incomplete attempts.

## Evaluate and analyze saved graphs

```powershell
python evaluate_ecg_against_gold.py `
  --gold "data/gold.json" `
  --pred "outputs/paired_run/with_repair/ecg.json" `
  --outdir "outputs/evaluation_exact"

python experiment_analysis.py --outdir "outputs/repair_analysis" repair --before "outputs/paired_run/without_repair/ecg.json" --after "outputs/paired_run/with_repair/ecg.json" --gold "data/gold.json"
```

Evaluation writes JSON, Markdown and alignment artifacts. Legacy metrics remain under
`metrics`; instance-based metrics are under `corrected.metrics`. Keep these families
separate. Corrected metrics include node/kind recovery, path, aligned contains/sequence,
and conditional and end-to-end grounding. COURSE is excluded by default.

For optional fuzzy sensitivity, repeat evaluation into a fresh directory with
`--fuzzy_threshold 0.9`. Do not relabel those scores as exact matching.

Gold references use `nodes` with unique IDs, titles, kinds and pages; `contains` and
`sequence` may be ID pairs. Optional `grounding` stores per-node evidence. Page arrays
are enumerated unless the reference explicitly declares:

```json
{"page_semantics": {"format": "[start_page,end_page]", "inclusive": true}}
```

Under that declaration, `[10, 15]` means pages 10 through 15 inclusive. Without it,
those values mean only pages 10 and 15. Validate annotations before running experiments.
Never change gold to fit predictions; version source-supported corrections and rescore
all compared systems consistently.

## Downstream document navigation pilot

This is a development benchmark, not a student-learning study. Three systems share a
model, question and OCR source: Full Context, local BM25 Flat RAG, and the same retrieved
chunks plus predicted ECG context. The retrieval baseline is local lexical retrieval,
not hosted file search. Input lengths differ and are measured, not budget matched.

First generate draft questions from YOUR gold, then manually check them against the PDF:

```powershell
python -m ecg_downstream_eval.build_benchmark --gold "data/gold.json" --out "data/navigation.jsonl" --n 24
```

The builder rejects ambiguous repeated titles: those require independently authored
location cues. Next-section means next sibling under the same parent. Section localization
is not scored because the questions already supply the title.

Prepare the pilot offline with explicit input paths (built-in D01 defaults refer to
private local experiments and are not included in this repository):

```powershell
python -m ecg_downstream_eval.run_pilot `
  --cases "data/navigation.jsonl" `
  --gold "data/gold.json" `
  --ocr "outputs/paired_run/ocr_pages.json" `
  --predicted "outputs/paired_run/with_repair/ecg.json" `
  --n 4 `
  --outdir "outputs/navigation_pilot"
```

Repeat the same command with `--resume --execute` to make 12 paid requests (four questions
across three systems). Output includes parent/next accuracy, strict and root-excluded
path accuracy, page precision/recall/F1, both traceability variants, input/output tokens,
observed API latency and failure counts. Traceability requires a correct path AND the
complete correct page set on the same question. Root-excluded scores are diagnostics;
page coverage does not establish semantic citation support.

To expand into a NEW directory, retain the input/model settings, increase `--n` and add
`--reuse-from "outputs/navigation_pilot"`. Compatible completed calls are imported rather
than repeated. Dry run first. Reuse verifies actual requests, settings and source hashes;
resume rejects changes to frozen settings/code. Existing failed/interrupted attempts are
preserved and skipped, not silently retried. Run only one process per output directory.

The actual tutoring-controller integration (Experiment B) is not part of this release.

## Tests and reproducibility

```powershell
python -m unittest discover -s tests -v
python -m unittest ecg_downstream_eval.test_pilot -v
```

Tests use local fixtures/mocks and do not launch paid requests. Keep gold, prompts,
evaluator versions, model settings and selected prediction artifacts frozen when
combining results. Missing API usage must remain unknown rather than being counted as
free. Reports from incomplete runs are not final results. Source datasets and historical
experiment snapshots are excluded from this code-only release.
