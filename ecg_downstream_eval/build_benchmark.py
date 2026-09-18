from __future__ import annotations
import argparse, random
from pathlib import Path
from .common import *

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--predicted", required=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    if Path(args.out).exists():
        raise SystemExit('Output already exists. Choose a new filename to preserve the old benchmark.')

    gold = load_json(args.gold)
    ids, parent, next_by = build_indexes(gold)
    decoded = graph_pages(gold)
    for src, dst in next_by.items():
        if src not in ids or dst not in ids or parent.get(src) != parent.get(dst):
            raise ValueError(f'Sequence is not a sibling relation: {src} -> {dst}')

    candidates = []
    for nid, n in ids.items():
        title = node_title(n)
        kind = node_kind(n).lower()
        pages = decoded[nid]
        if not title:
            continue
        if kind == "course":
            continue
        par_id = parent.get(nid)
        nxt_id = next_by.get(nid)
        path = full_path_from_parent(ids, parent, nid)

        # Require at least one independently checkable target.
        if not (par_id or nxt_id or pages or path):
            continue

        candidates.append({
            "gold_node_id": nid,
            "duplicate_title": sum(norm(node_title(x)) == norm(title) for x in ids.values()) > 1,
            "gold_title": title,
            "gold_kind": node_kind(n),
            "gold_parent_title": title_for_id(ids, par_id),
            "gold_next_title": title_for_id(ids, nxt_id),
            "gold_path": path,
            "gold_pages": pages,
        })

    rnd = random.Random(args.seed)
    rnd.shuffle(candidates)
    chosen = candidates[: min(args.n, len(candidates))]

    rows = []
    for i, c in enumerate(chosen, 1):
        if c['duplicate_title']:
            # Do not reveal a gold parent while also scoring parent identification.
            # Ambiguous repeated headings need independently authored location cues.
            raise ValueError('Repeated title requires a manually authored disambiguating question: ' + c['gold_title'])
        # We intentionally avoid exposing gold graph IDs in the user-facing prompt.
        rows.append({
            "case_id": f"NAV_{i:03d}",
            "task_type": "navigation_traceability",
            "benchmark_version": "navigation-v2-sibling-exact",
            "gold_node_id": c['gold_node_id'],
            "query": (
                f'For the document section "{c["gold_title"]}", identify its immediate parent section, '
                f'the immediately following sibling section under that same parent, '
                f'its full hierarchical path including the document root and this section, '
                f'and all PDF pages covered by this section (including its subsections). '
                f'Use an empty next_title if this is the last sibling; do not cross parent boundaries. '
                f'Use one-based physical PDF pages, not printed page labels.'
            ),
            "anchor_title": c["gold_title"],
            "gold": {
                "section_title": c["gold_title"],
                "parent_title": c["gold_parent_title"],
                "next_title": c["gold_next_title"],
                "path": c["gold_path"],
                "pages": c["gold_pages"],
            },
            "manual_check_required": True,
        })

    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} cases to {args.out}")
    if not rows:
        print("No compatible nodes found. Inspect the gold JSON schema and adjust common.py adapters.")

if __name__ == "__main__":
    main()
