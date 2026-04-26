#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def normalize_title(text: str) -> str:
    if text is None:
        return ""
    text = strip_accents(str(text))
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*[-–—]+\s*", " - ", text)
    text = re.sub(r"\s*[:;,.]+\s*$", "", text)
    return text.strip()


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def prf1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f1 = safe_div(2 * p * r, p + r) if (p + r) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(p, 6),
        "recall": round(r, 6),
        "f1": round(f1, 6),
    }


def jaccard(a: Set[Any], b: Set[Any]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a or b) else 0.0


PURE_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)*\.?$")
ROMAN_RE = re.compile(r"^[ivxlcdm]+\.?$", re.IGNORECASE)
LEADING_NUMBER_RE = re.compile(
    r"^\s*((?:\d+(?:\.\d+)*\.?)|(?:[ivxlcdm]+\.?))\s+(.*)$",
    re.IGNORECASE,
)


def extract_leading_number_from_title(title: str) -> Tuple[str, str]:
    raw = str(title or "").strip()
    m = LEADING_NUMBER_RE.match(raw)
    if not m:
        return "", raw
    return m.group(1).strip(), m.group(2).strip()


def build_visible_heading(title: str, number: Optional[str] = None) -> str:
    title = str(title or "").strip()
    number = str(number or "").strip()

    if not title and not number:
        return ""

    title_num, _ = extract_leading_number_from_title(title)
    if title_num:
        return title.strip()

    if number:
        number_out = number
        if PURE_NUMBER_RE.match(number_out) and not number_out.endswith("."):
            number_out = number_out + "."
        elif ROMAN_RE.match(number_out) and not number_out.endswith("."):
            number_out = number_out + "."
        return f"{number_out} {title}".strip()

    return title.strip()


def canonical_heading_full(title: str, number: Optional[str] = None) -> str:
    return normalize_title(build_visible_heading(title, number))


def canonical_heading_text_only(title: str, number: Optional[str] = None) -> str:
    visible = build_visible_heading(title, number)
    _, body = extract_leading_number_from_title(visible)
    return normalize_title(body)


def heading_pages_from_pred_node(n: Dict[str, Any]) -> List[int]:
    evidence_pages = n.get("evidence_pages")
    if isinstance(evidence_pages, list) and evidence_pages:
        return sorted(set(int(p) for p in evidence_pages if isinstance(p, int)))

    start = n.get("page_start")
    end = n.get("page_end", start)
    if isinstance(start, int) and isinstance(end, int):
        if end < start:
            end = start
        return list(range(start, end + 1))
    if isinstance(start, int):
        return [start]
    return []


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def canonicalize_gold(data: Dict[str, Any], include_course: bool = False) -> Dict[str, Any]:
    nodes = data.get("nodes", [])
    node_by_id = {n["id"]: n for n in nodes}

    canonical_nodes = []
    full_to_kinds = defaultdict(set)

    for n in nodes:
        kind = n.get("kind", "")
        if not include_course and kind == "COURSE":
            continue

        raw_title = n.get("title", "")
        full_norm = canonical_heading_full(raw_title)
        text_norm = canonical_heading_text_only(raw_title)

        entry = {
            "id": n["id"],
            "title": raw_title,
            "full_norm": full_norm,
            "text_norm": text_norm,
            "kind": kind,
        }
        canonical_nodes.append(entry)
        if full_norm:
            full_to_kinds[full_norm].add(kind)

    valid_ids = {n["id"] for n in canonical_nodes}

    contains_edges = set()
    for e in data.get("contains", []):
        s, t = (e[0], e[2]) if len(e) == 3 else (e[0], e[1])
        if s in valid_ids and t in valid_ids:
            contains_edges.add((
                canonical_heading_full(node_by_id[s].get("title", "")),
                canonical_heading_full(node_by_id[t].get("title", "")),
            ))

    sequence_edges = set()
    for e in data.get("sequence", []):
        s, t = (e[0], e[2]) if len(e) == 3 else (e[0], e[1])
        if s in valid_ids and t in valid_ids:
            sequence_edges.add((
                canonical_heading_full(node_by_id[s].get("title", "")),
                canonical_heading_full(node_by_id[t].get("title", "")),
            ))

    grounding = {}
    for nid, g in data.get("grounding", {}).items():
        if nid in valid_ids:
            full_norm = canonical_heading_full(node_by_id[nid].get("title", ""))
            grounding[full_norm] = {
                "pages": sorted(set(g.get("pages", []))),
                "heading_path": g.get("heading_path", []),
                "support_spans": g.get("support_spans", []),
            }

    return {
        "nodes": canonical_nodes,
        "node_titles": {n["full_norm"] for n in canonical_nodes if n["full_norm"]},
        "node_title_kind": {(n["full_norm"], n["kind"]) for n in canonical_nodes if n["full_norm"]},
        "kind_by_title": {k: sorted(v) for k, v in full_to_kinds.items()},
        "contains": contains_edges,
        "sequence": sequence_edges,
        "grounding": grounding,
        "source_title": data.get("source_title", ""),
    }


def canonicalize_pred(data: Dict[str, Any], include_course: bool = False) -> Dict[str, Any]:
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    node_by_id = {n["id"]: n for n in nodes}

    canonical_nodes = []
    full_to_kinds = defaultdict(set)

    for n in nodes:
        kind = n.get("kind", "")
        if not include_course and kind == "COURSE":
            continue

        visible_title = build_visible_heading(n.get("title", ""), n.get("number"))
        full_norm = canonical_heading_full(n.get("title", ""), n.get("number"))
        text_norm = canonical_heading_text_only(n.get("title", ""), n.get("number"))

        entry = {
            "id": n["id"],
            "title": visible_title,
            "full_norm": full_norm,
            "text_norm": text_norm,
            "kind": kind,
            "pages": heading_pages_from_pred_node(n),
        }
        canonical_nodes.append(entry)
        if full_norm:
            full_to_kinds[full_norm].add(kind)

    valid_ids = {n["id"] for n in canonical_nodes}

    def node_full_title(nid: str) -> str:
        n = node_by_id[nid]
        return canonical_heading_full(n.get("title", ""), n.get("number"))

    contains_edges = set()
    sequence_edges = set()
    for e in edges:
        s, t, typ = e.get("source"), e.get("target"), e.get("type")
        if s not in valid_ids or t not in valid_ids:
            continue
        st = node_full_title(s)
        tt = node_full_title(t)
        if typ == "contains":
            contains_edges.add((st, tt))
        elif typ == "sequence":
            sequence_edges.add((st, tt))

    grounding = {}
    for n in canonical_nodes:
        if n["full_norm"]:
            grounding[n["full_norm"]] = {
                "pages": n["pages"],
                "heading_path": node_by_id[n["id"]].get("path", []),
            }

    return {
        "nodes": canonical_nodes,
        "node_titles": {n["full_norm"] for n in canonical_nodes if n["full_norm"]},
        "node_title_kind": {(n["full_norm"], n["kind"]) for n in canonical_nodes if n["full_norm"]},
        "kind_by_title": {k: sorted(v) for k, v in full_to_kinds.items()},
        "contains": contains_edges,
        "sequence": sequence_edges,
        "grounding": grounding,
        "metadata": data.get("metadata", {}),
        "validation": data.get("validation", {}),
    }


def compare_sets(gold: Set[Any], pred: Set[Any]) -> Dict[str, float]:
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    out = prf1(tp, fp, fn)
    out["jaccard"] = round(jaccard(gold, pred), 6)
    return out


def evaluate_page_grounding(gold_ground: Dict[str, Dict[str, Any]], pred_ground: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    per_title = {}
    total_tp = total_fp = total_fn = 0
    grounded_nodes = 0
    titles_in_both = sorted(set(gold_ground) & set(pred_ground))

    for title in titles_in_both:
        gpages = set(gold_ground[title].get("pages", []))
        ppages = set(pred_ground[title].get("pages", []))
        if ppages:
            grounded_nodes += 1
        tp = len(gpages & ppages)
        fp = len(ppages - gpages)
        fn = len(gpages - ppages)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_title[title] = {
            "gold_pages": sorted(gpages),
            "pred_pages": sorted(ppages),
            **prf1(tp, fp, fn),
        }

    micro = prf1(total_tp, total_fp, total_fn)
    micro["grounded_kc_rate"] = round(safe_div(grounded_nodes, len(gold_ground)), 6) if gold_ground else 0.0
    return {"micro": micro, "per_title": per_title}


def evaluate_kind_agreement(gold_kind_by_title: Dict[str, List[str]], pred_kind_by_title: Dict[str, List[str]]) -> Dict[str, Any]:
    titles = sorted(set(gold_kind_by_title) & set(pred_kind_by_title))
    correct = 0
    per_title = {}
    confusion = Counter()
    for t in titles:
        gold_kinds = gold_kind_by_title[t]
        pred_kinds = pred_kind_by_title[t]
        ok = bool(set(gold_kinds) & set(pred_kinds))
        correct += int(ok)
        per_title[t] = {
            "gold_kinds": gold_kinds,
            "pred_kinds": pred_kinds,
            "match": ok,
        }
        confusion[(gold_kinds[0] if gold_kinds else "", pred_kinds[0] if pred_kinds else "")] += 1
    return {
        "title_kind_accuracy": round(safe_div(correct, len(titles)), 6) if titles else 0.0,
        "matched_titles": len(titles),
        "per_title": per_title,
        "confusion": [{"gold": g, "pred": p, "count": c} for (g, p), c in sorted(confusion.items())],
    }


def collect_differences(gold: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "missing_titles": sorted(gold["node_titles"] - pred["node_titles"]),
        "extra_titles": sorted(pred["node_titles"] - gold["node_titles"]),
        "missing_contains": sorted([list(x) for x in gold["contains"] - pred["contains"]]),
        "extra_contains": sorted([list(x) for x in pred["contains"] - gold["contains"]]),
        "missing_sequence": sorted([list(x) for x in gold["sequence"] - pred["sequence"]]),
        "extra_sequence": sorted([list(x) for x in pred["sequence"] - gold["sequence"]]),
    }


def collect_text_only_near_matches(gold: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    gold_only = gold["node_titles"] - pred["node_titles"]
    pred_only = pred["node_titles"] - gold["node_titles"]

    gold_text_map = defaultdict(list)
    pred_text_map = defaultdict(list)

    for full in gold_only:
        _, body = extract_leading_number_from_title(full)
        gold_text_map[normalize_title(body)].append(full)

    for full in pred_only:
        _, body = extract_leading_number_from_title(full)
        pred_text_map[normalize_title(body)].append(full)

    near_matches = []
    for text_norm in sorted(set(gold_text_map) & set(pred_text_map)):
        near_matches.append({
            "text_norm": text_norm,
            "gold_candidates": sorted(gold_text_map[text_norm]),
            "pred_candidates": sorted(pred_text_map[text_norm]),
        })

    return {"count": len(near_matches), "examples": near_matches[:50]}


def evaluate(gold_path: Path, pred_path: Path, include_course: bool = False) -> Dict[str, Any]:
    gold_raw = load_json(gold_path)
    pred_raw = load_json(pred_path)

    gold = canonicalize_gold(gold_raw, include_course=include_course)
    pred = canonicalize_pred(pred_raw, include_course=include_course)

    node_metrics = compare_sets(gold["node_titles"], pred["node_titles"])
    node_kind_metrics = compare_sets(gold["node_title_kind"], pred["node_title_kind"])
    contains_metrics = compare_sets(gold["contains"], pred["contains"])
    sequence_metrics = compare_sets(gold["sequence"], pred["sequence"])
    page_grounding = evaluate_page_grounding(gold["grounding"], pred["grounding"])
    kind_agreement = evaluate_kind_agreement(gold["kind_by_title"], pred["kind_by_title"])
    diffs = collect_differences(gold, pred)
    near_matches = collect_text_only_near_matches(gold, pred)

    validation = pred.get("validation", {})
    metadata = pred.get("metadata", {})

    return {
        "inputs": {
            "gold": str(gold_path),
            "pred": str(pred_path),
            "include_course": include_course,
        },
        "summary": {
            "gold_node_count": len(gold["node_titles"]),
            "pred_node_count": len(pred["node_titles"]),
            "gold_contains_count": len(gold["contains"]),
            "pred_contains_count": len(pred["contains"]),
            "gold_sequence_count": len(gold["sequence"]),
            "pred_sequence_count": len(pred["sequence"]),
        },
        "metrics": {
            "nodes_by_title": node_metrics,
            "nodes_by_title_and_kind": node_kind_metrics,
            "contains": contains_metrics,
            "sequence": sequence_metrics,
            "page_grounding": page_grounding["micro"],
            "kind_agreement": {
                "title_kind_accuracy": kind_agreement["title_kind_accuracy"],
                "matched_titles": kind_agreement["matched_titles"],
            },
            "pred_validation": {
                "ok": validation.get("ok", metadata.get("validation_ok")),
                "violation_count": len(validation.get("violations", []))
                if isinstance(validation.get("violations", []), list)
                else metadata.get("violation_count"),
            },
        },
        "details": {
            "page_grounding_per_title": page_grounding["per_title"],
            "kind_agreement_per_title": kind_agreement["per_title"],
            "kind_confusion": kind_agreement["confusion"],
            "differences": diffs,
            "text_only_near_matches": near_matches,
        },
    }


def print_summary(results: Dict[str, Any]) -> None:
    m = results["metrics"]
    s = results["summary"]
    print("\nECG evaluation summary")
    print("=" * 72)
    print(f"Gold nodes      : {s['gold_node_count']}")
    print(f"Pred nodes      : {s['pred_node_count']}")
    print(f"Gold contains   : {s['gold_contains_count']}")
    print(f"Pred contains   : {s['pred_contains_count']}")
    print(f"Gold sequence   : {s['gold_sequence_count']}")
    print(f"Pred sequence   : {s['pred_sequence_count']}")
    print("-" * 72)
    for key, label in [
        ("nodes_by_title", "Node recovery (title)"),
        ("nodes_by_title_and_kind", "Node recovery (title+kind)"),
        ("contains", "Hierarchy recovery (contains)"),
        ("sequence", "Order recovery (sequence)"),
    ]:
        mm = m[key]
        print(f"{label:30s} P={mm['precision']:.3f} R={mm['recall']:.3f} F1={mm['f1']:.3f}")
    pg = m["page_grounding"]
    print(f"{'Page grounding':30s} P={pg['precision']:.3f} R={pg['recall']:.3f} F1={pg['f1']:.3f}  grounded_rate={pg['grounded_kc_rate']:.3f}")
    ka = m["kind_agreement"]
    print(f"{'Kind agreement':30s} acc={ka['title_kind_accuracy']:.3f}  matched_titles={ka['matched_titles']}")
    pv = m["pred_validation"]
    print(f"{'Pred validation':30s} ok={pv['ok']}  violations={pv['violation_count']}")
    near = results["details"]["text_only_near_matches"]
    print(f"{'Text-only near matches':30s} count={near['count']}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted ECG against gold annotation")
    parser.add_argument("--gold", required=True, help="Path to gold annotation JSON")
    parser.add_argument("--pred", required=True, help="Path to predicted ECG JSON")
    parser.add_argument("--outdir", default="eval_out", help="Directory for outputs")
    parser.add_argument("--include_course", action="store_true", help="Include COURSE nodes in comparison")
    args = parser.parse_args()

    gold_path = Path(args.gold)
    pred_path = Path(args.pred)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = evaluate(gold_path, pred_path, include_course=args.include_course)
    out_json = outdir / "evaluation_results.json"
    out_md = outdir / "evaluation_summary.md"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    m = results["metrics"]
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# ECG Evaluation Summary\n\n")
        f.write(f"- Gold: `{gold_path}`\n")
        f.write(f"- Prediction: `{pred_path}`\n")
        f.write(f"- Include COURSE: `{args.include_course}`\n\n")
        for key, label in [
            ("nodes_by_title", "Node recovery (title)"),
            ("nodes_by_title_and_kind", "Node recovery (title+kind)"),
            ("contains", "Hierarchy recovery (contains)"),
            ("sequence", "Order recovery (sequence)"),
        ]:
            mm = m[key]
            f.write(f"- **{label}**: P={mm['precision']:.3f}, R={mm['recall']:.3f}, F1={mm['f1']:.3f}\n")
        pg = m["page_grounding"]
        f.write(f"- **Page grounding**: P={pg['precision']:.3f}, R={pg['recall']:.3f}, F1={pg['f1']:.3f}, grounded_rate={pg['grounded_kc_rate']:.3f}\n")
        ka = m["kind_agreement"]
        f.write(f"- **Kind agreement**: acc={ka['title_kind_accuracy']:.3f}, matched_titles={ka['matched_titles']}\n")
        pv = m["pred_validation"]
        f.write(f"- **Pred validation**: ok={pv['ok']}, violations={pv['violation_count']}\n")
        near = results["details"]["text_only_near_matches"]
        f.write(f"- **Text-only near matches**: count={near['count']}\n")

    print_summary(results)
    print(f"\nSaved JSON results to: {out_json}")
    print(f"Saved Markdown summary to: {out_md}")


if __name__ == "__main__":
    main()
