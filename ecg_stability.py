from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_title(text: str) -> str:
    text = (text or "").lower().replace("’", "'").strip()
    text = " ".join(text.split())
    return text


def dice(a: Set[Tuple[str, ...]], b: Set[Tuple[str, ...]]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return round(2 * inter / (len(a) + len(b)), 4)


def mean(xs: List[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def std(xs: List[float]) -> float | None:
    if not xs:
        return None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return round(math.sqrt(var), 4)


def build_sets(ecg: Dict[str, Any]) -> Dict[str, Set[Tuple[str, ...]]]:
    nodes = ecg.get("nodes", []) or []
    edges = ecg.get("edges", []) or []
    id_to_node = {n.get("id"): n for n in nodes if n.get("id")}

    titles = set()
    title_kind = set()
    for n in nodes:
        t = normalize_title(n.get("title", ""))
        k = normalize_title(n.get("kind", ""))
        if t:
            titles.add((t,))
            title_kind.add((t, k))

    contains = set()
    sequence = set()
    for e in edges:
        src = id_to_node.get(e.get("source"))
        tgt = id_to_node.get(e.get("target"))
        if not src or not tgt:
            continue
        st = normalize_title(src.get("title", ""))
        tt = normalize_title(tgt.get("title", ""))
        if not st or not tt:
            continue
        if e.get("type") == "contains":
            contains.add((st, tt))
        elif e.get("type") == "sequence":
            sequence.add((st, tt))

    return {
        "titles": titles,
        "title_kind": title_kind,
        "contains": contains,
        "sequence": sequence,
    }


def summarize_runs(paths: List[str]) -> Dict[str, Any]:
    runs = []
    for path in paths:
        ecg = load_json(path)
        validation_ok = bool(ecg.get("validation", {}).get("ok"))
        nodes = ecg.get("nodes", []) or []
        edges = ecg.get("edges", []) or []
        runs.append(
            {
                "path": str(path),
                "validation_ok": validation_ok,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "sets": build_sets(ecg),
            }
        )

    val_rate = mean([1.0 if r["validation_ok"] else 0.0 for r in runs])
    node_counts = [r["node_count"] for r in runs]
    edge_counts = [r["edge_count"] for r in runs]

    pair_scores = {"titles": [], "title_kind": [], "contains": [], "sequence": []}
    for a, b in combinations(runs, 2):
        for key in pair_scores.keys():
            pair_scores[key].append(dice(a["sets"][key], b["sets"][key]))

    return {
        "runs": [
            {
                "path": r["path"],
                "validation_ok": r["validation_ok"],
                "node_count": r["node_count"],
                "edge_count": r["edge_count"],
            }
            for r in runs
        ],
        "validation_pass_rate": val_rate,
        "nodes_mean": mean([float(n) for n in node_counts]),
        "nodes_std": std([float(n) for n in node_counts]),
        "edges_mean": mean([float(e) for e in edge_counts]),
        "edges_std": std([float(e) for e in edge_counts]),
        "dice_overlap_mean": {
            "titles": mean(pair_scores["titles"]),
            "title_kind": mean(pair_scores["title_kind"]),
            "contains": mean(pair_scores["contains"]),
            "sequence": mean(pair_scores["sequence"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ECG stability across multiple runs")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of ecg.json paths (5 runs, etc.)")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    summary = summarize_runs(args.inputs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved stability summary to {args.out}")


if __name__ == "__main__":
    main()
