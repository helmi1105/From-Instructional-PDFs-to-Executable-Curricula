# Annotation Protocol (Semi-Automatic, Human-Validated Gold)

## Core principle

The pipeline output is a **draft ECG** used as **pre-annotation** only.  
The draft is **not** the gold.

Gold ECG is created only after human verification and correction by annotators.

## Workflow

1. Automatic pipeline generates draft ECG (`ecg.json`).
2. Each annotator receives the same source document + draft ECG.
3. Annotators independently verify and correct:
   - node inventory (titles),
   - node type (`kind`),
   - `contains` edges,
   - `sequence` edges,
   - page grounding (`grounding.pages`).
4. Independent annotations are exported as ECG JSON files.
5. Agreement is measured on independent annotations.
6. Final consensus meeting resolves disagreements.
7. Consensus file is stored as final human-validated gold ECG.

## Required reporting

For 2 annotators + consensus:
- Dice for node inventory.
- Dice for node + kind.
- Dice for contains.
- Dice for sequence.
- Dice for page grounding.
- Cohen's kappa for node type.

For 3 independent annotators:
- Dice pairwise for the five structural aspects above.
- Fleiss' kappa for node type.

## Annotator metadata

Annotator profiles must be documented in a separate file:
- role/background,
- years of experience,
- ECG training details,
- annotation load/splits.

Use `annotator_profiles_template.json`.

## Guideline publication

The annotation guideline must be versioned and archived with experiments.

Use `annotation_guideline_template.md` as a starting point.
