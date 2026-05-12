# ECG Annotation Guideline (Template)

Version: `v1.0`  
Date: `YYYY-MM-DD`

## Scope

- Input: source document + pipeline draft ECG.
- Output: corrected ECG JSON.
- Rule: draft ECG is pre-annotation only and is never considered gold.

## Node policy

- Keep headings that represent curriculum structure.
- Remove OCR artifacts and duplicates.
- Normalize minor punctuation/noise but preserve meaning.
- Assign one node type (`kind`) from allowed labels:
  - `COURSE`, `MODULE`, `UNIT`, `KC`

## Edge policy

- `contains`: parent/child conceptual hierarchy.
- `sequence`: immediate learning/order relation among siblings when explicit.
- Do not create transitive closure edges.

## Grounding policy

- Each kept node must have page grounding when evidence exists.
- `grounding.pages` contains integer page numbers.
- If uncertain, mark for adjudication and do not invent pages.

## Conflict policy

- Mark ambiguous cases for consensus.
- Consensus decision log must be saved with rationale.

## Quality checklist

- Node IDs unique.
- No dangling references in `contains` / `sequence`.
- Node type set for each node.
- Grounding pages valid integers.

## Changelog

- `v1.0`: initial template.
