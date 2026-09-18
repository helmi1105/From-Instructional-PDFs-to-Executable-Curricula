# Source corpus

The manifest indexes 20 source PDFs and their current gold ECG annotations (D01-D20).
Paths are relative to this directory. It records domain, language, document length,
page encoding where supplied, and SHA-256 checksums. Generated predictions, OCR caches,
API responses, evaluation results and review screenshots are excluded.

## Reference versions

D01 uses `D01/gold_ecg_annotation_v2.json`. The original v1 file is retained for historical
reproducibility. V2 adds the visibly printed `3.1 LES ACTIONS` UNIT on pages 13-16 beneath
`III - LES ACTIONS` and reparents Reconnaissances, Actions offensives and Actions defensives.
Grounding paths are updated; existing leaf page sets and sequence edges are preserved.
The annotation records the revision rationale and prior checksum.

On 2026-09-18, manifest checksums were refreshed to match the current D09-D20 annotations.
The former manifest values are retained as `gold_previous_manifest_sha256`; those values
are not hashes of additional files included here. Historical result snapshots may use
other reference versions. Do not combine their scores without checking provenance.

These files are supplied references, not a claim of independent double annotation or
complete semantic validation. Publication checks verified file presence, JSON parsing,
declared page decoding and checksums; they do not certify every heading or relation.

## Page interpretation

Unspecified page arrays are enumerated pages. When `page_semantics` declares the inclusive
`[start_page,end_page]` format, a pair such as `[10,15]` represents every page from 10 to 15.
Use `page_grounding.py` and the shared evaluator rather than guessing from array length.

## Using the corpus

Select PDF and gold paths from `manifest.json`. Supply the chosen gold explicitly when
running evaluation; older scripts or commands may still name a previous reference file.
Store newly generated artifacts outside this source directory, in an ignored output folder.

Source PDFs retain their original publishers' notices. Their inclusion does not grant
additional reuse rights or apply a new license to the source documents.
