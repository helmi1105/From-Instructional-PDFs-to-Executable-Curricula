#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def normalize_title(text: str) -> str:
    text = strip_accents(str(text or ""))
    text = text.replace("â€™", "'").replace("â€˜", "'").replace("`", "'").replace("Â´", "'")
    text = text.replace("â€“", "-").replace("â€”", "-")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*[-â€“â€”]+\s*", " - ", text)
    text = re.sub(r"\s*[:;,.]+\s*$", "", text)
    return text.strip()


def dice(a: Set[Tuple[str, ...]], b: Set[Tuple[str, ...]]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return round(2 * len(a & b) / (len(a) + len(b)), 6)


def build_canonical_sets(data: Dict[str, Any], include_course: bool = False) -> Dict[str, Any]:
    nodes = data.get("nodes", []) or []
    id_to_node = {n.get("id"): n for n in nodes if n.get("id")}

    node_inventory: Set[Tuple[str]] = set()
    node_kind: Set[Tuple[str, str]] = set()
    kind_by_title: Dict[str, str] = {}

    for n in nodes:
        kind = str(n.get("kind", "")).strip()
        if not include_course and kind == "COURSE":
            continue
        title = normalize_title(n.get("title", ""))
        if not title:
            continue
        node_inventory.add((title,))
        node_kind.add((title, kind))
        kind_by_title[title] = kind

    valid_ids = {
        nid for nid, n in id_to_node.items()
        if include_course or str(n.get("kind", "")).strip() != "COURSE"
    }

    contains: Set[Tuple[str, str]] = set()
    sequence: Set[Tuple[str, str]] = set()

    for e in data.get("contains", []) or []:
        if len(e) < 2:
            continue
        s, t = (e[0], e[2]) if len(e) >= 3 else (e[0], e[1])
        if s not in valid_ids or t not in valid_ids:
            continue
        st = normalize_title(id_to_node.get(s, {}).get("title", ""))
        tt = normalize_title(id_to_node.get(t, {}).get("title", ""))
        if st and tt:
            contains.add((st, tt))

    for e in data.get("sequence", []) or []:
        if len(e) < 2:
            continue
        s, t = (e[0], e[2]) if len(e) >= 3 else (e[0], e[1])
        if s not in valid_ids or t not in valid_ids:
            continue
        st = normalize_title(id_to_node.get(s, {}).get("title", ""))
        tt = normalize_title(id_to_node.get(t, {}).get("title", ""))
        if st and tt:
            sequence.add((st, tt))

    page_grounding: Set[Tuple[str, str]] = set()
    grounding = data.get("grounding", {}) or {}
    for nid, g in grounding.items():
        if nid not in valid_ids:
            continue
        title = normalize_title(id_to_node.get(nid, {}).get("title", ""))
        if not title:
            continue
        for p in g.get("pages", []) or []:
            if isinstance(p, int):
                page_grounding.add((title, str(p)))

    return {
        "node_inventory": node_inventory,
        "node_kind": node_kind,
        "contains": contains,
        "sequence": sequence,
        "page_grounding": page_grounding,
        "kind_by_title": kind_by_title,
    }


def cohen_kappa(labels_a: List[str], labels_b: List[str]) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0
    n = len(labels_a)
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    ca = Counter(labels_a)
    cb = Counter(labels_b)
    cats = set(ca) | set(cb)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if pe >= 1.0:
        return 1.0
    return round((po - pe) / (1 - pe), 6)


def fleiss_kappa(rows: List[Counter], categories: List[str]) -> float:
    if not rows or not categories:
        return 0.0
    n = sum(rows[0].values())
    if n <= 1:
        return 0.0
    N = len(rows)
    pj = {c: 0.0 for c in categories}
    for c in categories:
        pj[c] = sum(r.get(c, 0) for r in rows) / (N * n)
    P_i: List[float] = []
    for r in rows:
        num = sum(v * (v - 1) for v in (r.get(c, 0) for c in categories))
        P_i.append(num / (n * (n - 1)))
    P_bar = sum(P_i) / N
    P_e = sum(v * v for v in pj.values())
    if P_e >= 1.0:
        return 1.0
    return round((P_bar - P_e) / (1 - P_e), 6)


def pairwise_dice(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    return {
        "node_inventory": dice(a["node_inventory"], b["node_inventory"]),
        "node_kind": dice(a["node_kind"], b["node_kind"]),
        "contains": dice(a["contains"], b["contains"]),
        "sequence": dice(a["sequence"], b["sequence"]),
        "page_grounding": dice(a["page_grounding"], b["page_grounding"]),
    }


def report_two_annotators(
    ann_a: Dict[str, Any],
    ann_b: Dict[str, Any],
    consensus: Dict[str, Any] | None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["pairwise_dice"] = pairwise_dice(ann_a, ann_b)

    all_titles = sorted(set(ann_a["kind_by_title"]) | set(ann_b["kind_by_title"]))
    labels_a = [ann_a["kind_by_title"].get(t, "ABSENT") for t in all_titles]
    labels_b = [ann_b["kind_by_title"].get(t, "ABSENT") for t in all_titles]
    out["cohen_kappa_node_type"] = {
        "kappa": cohen_kappa(labels_a, labels_b),
        "items": len(all_titles),
        "label_space": sorted(set(labels_a) | set(labels_b)),
        "absent_label_used": True,
    }

    if consensus is not None:
        out["vs_consensus"] = {
            "annotator_1_dice": pairwise_dice(ann_a, consensus),
            "annotator_2_dice": pairwise_dice(ann_b, consensus),
        }
    return out


def report_three_annotators(anns: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["pairwise_dice"] = {
        "1_vs_2": pairwise_dice(anns[0], anns[1]),
        "1_vs_3": pairwise_dice(anns[0], anns[2]),
        "2_vs_3": pairwise_dice(anns[1], anns[2]),
    }

    all_titles = sorted(set().union(*(set(a["kind_by_title"]) for a in anns)))
    categories = sorted(
        set().union(*(set(a["kind_by_title"].values()) for a in anns)) | {"ABSENT"}
    )
    rows: List[Counter] = []
    for t in all_titles:
        rows.append(Counter([a["kind_by_title"].get(t, "ABSENT") for a in anns]))

    out["fleiss_kappa_node_type"] = {
        "kappa": fleiss_kappa(rows, categories),
        "items": len(all_titles),
        "categories": categories,
        "absent_label_used": True,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inter-annotator agreement for ECG annotations (2 or 3 annotators)."
    )
    parser.add_argument(
        "--annotations",
        nargs="+",
        required=True,
        help="List of annotation JSON files (2 or 3).",
    )
    parser.add_argument(
        "--consensus",
        default=None,
        help="Optional consensus/final gold JSON path (recommended for 2 annotators).",
    )
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument(
        "--include_course",
        action="store_true",
        help="Include COURSE nodes in agreement metrics.",
    )
    args = parser.parse_args()

    if len(args.annotations) not in (2, 3):
        raise SystemExit("--annotations must contain exactly 2 or 3 files.")
    if len(args.annotations) == 3 and args.consensus:
        raise SystemExit("--consensus is only for the 2-annotator setup.")

    canonical = [
        build_canonical_sets(load_json(p), include_course=args.include_course)
        for p in args.annotations
    ]

    out: Dict[str, Any] = {
        "inputs": {
            "annotations": args.annotations,
            "consensus": args.consensus,
            "include_course": args.include_course,
        },
        "protocol_note": "Pipeline draft ECG is a pre-annotation only and is never treated as gold.",
    }

    if len(canonical) == 2:
        consensus = (
            build_canonical_sets(load_json(args.consensus), include_course=args.include_course)
            if args.consensus
            else None
        )
        out["agreement"] = report_two_annotators(canonical[0], canonical[1], consensus)
        out["kappa_type"] = "cohen"
    else:
        out["agreement"] = report_three_annotators(canonical)
        out["kappa_type"] = "fleiss"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved inter-annotator agreement report to {out_path}")


if __name__ == "__main__":
    main()
