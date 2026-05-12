#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def normalize_text(text: str) -> str:
    text = strip_accents(str(text or ""))
    text = text.replace("â€™", "'").replace("â€˜", "'").replace("`", "'")
    text = text.replace("â€“", "-").replace("â€”", "-")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*[:;,.]+\s*$", "", text)
    return text.strip()


LEADING_NUMBER_RE = re.compile(
    r"^\s*((?:\d+(?:\.\d+)*\.?)|(?:[ivxlcdm]+\.?))\s+(.*)$",
    re.IGNORECASE,
)


def build_visible_heading(title: str, number: str | None = None) -> str:
    title = str(title or "").strip()
    number = str(number or "").strip()
    m = LEADING_NUMBER_RE.match(title)
    if m:
        return title
    if number:
        return f"{number} {title}".strip()
    return title


def canonical_title(title: str, number: str | None = None) -> str:
    return normalize_text(build_visible_heading(title, number))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def canonicalize_gold(data: Dict[str, Any]) -> Dict[str, Any]:
    nodes = data.get("nodes", []) or []
    node_by_id = {n.get("id"): n for n in nodes}

    titles: Set[str] = set()
    title_kind: Set[Tuple[str, str]] = set()
    titles_to_kind: Dict[str, str] = {}

    for n in nodes:
        nid = n.get("id")
        if not nid:
            continue
        title = canonical_title(n.get("title", ""))
        kind = str(n.get("kind", "")).strip()
        if not title:
            continue
        titles.add(title)
        if kind:
            title_kind.add((title, kind))
            if title not in titles_to_kind:
                titles_to_kind[title] = kind

    contains: Set[Tuple[str, str]] = set()
    for e in data.get("contains", []) or []:
        if len(e) < 2:
            continue
        s = e[0]
        t = e[2] if len(e) >= 3 else e[1]
        if s in node_by_id and t in node_by_id:
            st = canonical_title(node_by_id[s].get("title", ""))
            tt = canonical_title(node_by_id[t].get("title", ""))
            if st and tt:
                contains.add((st, tt))

    sequence: Set[Tuple[str, str]] = set()
    for e in data.get("sequence", []) or []:
        if len(e) < 2:
            continue
        s = e[0]
        t = e[2] if len(e) >= 3 else e[1]
        if s in node_by_id and t in node_by_id:
            st = canonical_title(node_by_id[s].get("title", ""))
            tt = canonical_title(node_by_id[t].get("title", ""))
            if st and tt:
                sequence.add((st, tt))

    grounding_pages: Dict[str, Set[int]] = {}
    for nid, g in (data.get("grounding", {}) or {}).items():
        if nid not in node_by_id:
            continue
        title = canonical_title(node_by_id[nid].get("title", ""))
        pages = set(p for p in (g.get("pages", []) or []) if isinstance(p, int))
        if title:
            grounding_pages[title] = pages

    return {
        "titles": titles,
        "title_kind": title_kind,
        "titles_to_kind": titles_to_kind,
        "contains": contains,
        "sequence": sequence,
        "grounding_pages": grounding_pages,
    }


def pages_from_pred_node(node: Dict[str, Any]) -> Set[int]:
    out: Set[int] = set()
    ev = node.get("evidence_pages")
    if isinstance(ev, list):
        out = {p for p in ev if isinstance(p, int)}
        if out:
            return out
    s = node.get("page_start")
    e = node.get("page_end", s)
    if isinstance(s, int) and isinstance(e, int):
        if e < s:
            e = s
        return set(range(s, e + 1))
    if isinstance(s, int):
        return {s}
    return set()


def canonicalize_pred(data: Dict[str, Any]) -> Dict[str, Any]:
    nodes = data.get("nodes", []) or []
    edges = data.get("edges", []) or []
    node_by_id = {n.get("id"): n for n in nodes}

    titles: Set[str] = set()
    title_kind: Set[Tuple[str, str]] = set()
    titles_to_kind: Dict[str, str] = {}
    title_pages: Dict[str, Set[int]] = {}

    for n in nodes:
        nid = n.get("id")
        if not nid:
            continue
        title = canonical_title(n.get("title", ""), n.get("number"))
        kind = str(n.get("kind", "")).strip()
        if not title:
            continue
        titles.add(title)
        if kind:
            title_kind.add((title, kind))
            if title not in titles_to_kind:
                titles_to_kind[title] = kind
        title_pages[title] = pages_from_pred_node(n)

    contains: Set[Tuple[str, str]] = set()
    sequence: Set[Tuple[str, str]] = set()

    for e in edges:
        s = e.get("source")
        t = e.get("target")
        typ = e.get("type")
        if s not in node_by_id or t not in node_by_id:
            continue
        st = canonical_title(node_by_id[s].get("title", ""), node_by_id[s].get("number"))
        tt = canonical_title(node_by_id[t].get("title", ""), node_by_id[t].get("number"))
        if not st or not tt:
            continue
        if typ == "contains":
            contains.add((st, tt))
        elif typ == "sequence":
            sequence.add((st, tt))

    return {
        "titles": titles,
        "title_kind": title_kind,
        "titles_to_kind": titles_to_kind,
        "contains": contains,
        "sequence": sequence,
        "grounding_pages": title_pages,
    }


def compare_one(gold_path: Path, pred_path: Path, name: str) -> Dict[str, Any]:
    gold = canonicalize_gold(load_json(gold_path))
    pred = canonicalize_pred(load_json(pred_path))

    gold_titles = gold["titles"]
    pred_titles = pred["titles"]
    common_titles = gold_titles & pred_titles

    missing_titles = sorted(gold_titles - pred_titles)
    extra_titles = sorted(pred_titles - gold_titles)

    wrong_kinds = []
    for t in sorted(common_titles):
        gk = gold["titles_to_kind"].get(t)
        pk = pred["titles_to_kind"].get(t)
        if gk and pk and gk != pk:
            wrong_kinds.append({"title": t, "gold_kind": gk, "pred_kind": pk})

    missing_contains = sorted(list(gold["contains"] - pred["contains"]))
    extra_contains = sorted(list(pred["contains"] - gold["contains"]))
    missing_sequence = sorted(list(gold["sequence"] - pred["sequence"]))
    extra_sequence = sorted(list(pred["sequence"] - gold["sequence"]))

    page_errors = []
    for t in sorted(common_titles):
        gp = gold["grounding_pages"].get(t, set())
        pp = pred["grounding_pages"].get(t, set())
        if gp and pp != gp:
            page_errors.append(
                {
                    "title": t,
                    "gold_pages": sorted(gp),
                    "pred_pages": sorted(pp),
                }
            )

    return {
        "document": name,
        "gold_path": str(gold_path),
        "pred_path": str(pred_path),
        "counts": {
            "missing_titles": len(missing_titles),
            "extra_titles": len(extra_titles),
            "wrong_node_kinds": len(wrong_kinds),
            "missing_contains_edges": len(missing_contains),
            "extra_contains_edges": len(extra_contains),
            "missing_sequence_edges": len(missing_sequence),
            "extra_sequence_edges": len(extra_sequence),
            "page_grounding_errors": len(page_errors),
        },
        "details": {
            "missing_titles": missing_titles,
            "extra_titles": extra_titles,
            "wrong_node_kinds": wrong_kinds,
            "missing_contains_edges": [{"source": s, "target": t} for (s, t) in missing_contains],
            "extra_contains_edges": [{"source": s, "target": t} for (s, t) in extra_contains],
            "missing_sequence_edges": [{"source": s, "target": t} for (s, t) in missing_sequence],
            "extra_sequence_edges": [{"source": s, "target": t} for (s, t) in extra_sequence],
            "page_grounding_errors": page_errors,
        },
    }


def default_pairs() -> List[Dict[str, str]]:
    project_dir = Path(__file__).resolve().parent
    return [
        {
            "name": "D1",
            "gold": str(project_dir / "gold_ecg_annotation_v1.json"),
            "pred": str(project_dir / "outputs" / "clean_ecg" / "ecg.json"),
        },
        {
            "name": "D2",
            "gold": str(project_dir / "gold_ecg_memento_chef_de_groupe_same_form.json"),
            "pred": str(project_dir / "outputs" / "clean_ecg_2" / "ecg.json"),
        },
    ]


def save_csv(rows: List[Dict[str, Any]], out_path: Path, error_keys: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    header = ["document"] + error_keys
    lines.append(",".join(header))
    for r in rows:
        vals = [r["document"]] + [str(r["counts"][k]) for k in error_keys]
        lines.append(",".join(vals))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_markdown(rows: List[Dict[str, Any]], out_path: Path, error_keys: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pretty = {
        "missing_titles": "missing titles",
        "extra_titles": "extra titles",
        "wrong_node_kinds": "wrong node kinds",
        "missing_contains_edges": "missing contains edges",
        "extra_contains_edges": "extra contains edges",
        "missing_sequence_edges": "missing sequence edges",
        "extra_sequence_edges": "extra sequence edges",
        "page_grounding_errors": "page-grounding errors",
    }
    lines = []
    lines.append("# Structural Error Report")
    lines.append("")
    lines.append("| document | " + " | ".join(pretty[k] for k in error_keys) + " |")
    lines.append("|---|" + "|".join(["---"] * len(error_keys)) + "|")
    for r in rows:
        vals = [str(r["counts"][k]) for k in error_keys]
        lines.append(f"| {r['document']} | " + " | ".join(vals) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_heatmap(rows: List[Dict[str, Any]], out_path: Path, error_keys: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    docs = [r["document"] for r in rows]
    data = np.array([[r["counts"][k] for k in error_keys] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(11, max(4, len(docs) * 0.6)))
    im = ax.imshow(data, cmap="Reds", aspect="auto")

    ax.set_xticks(range(len(error_keys)))
    ax.set_xticklabels(
        [
            "missing\n titles",
            "extra\n titles",
            "wrong\n kinds",
            "missing\n contains",
            "extra\n contains",
            "missing\n sequence",
            "extra\n sequence",
            "page\n errors",
        ],
        fontsize=9,
    )
    ax.set_yticks(range(len(docs)))
    ax.set_yticklabels(docs, fontsize=9)
    ax.set_title("Heatmap of Structural Error Modes by Document", fontsize=12)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = int(data[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Error count")

    plt.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-document structural error report and heatmap."
    )
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parent / "eval_out" / "structural_errors"),
        help="Output directory for report artifacts.",
    )
    parser.add_argument(
        "--pairs_json",
        default=None,
        help="Optional JSON list of {name, gold, pred} pairs. If omitted, local defaults are used.",
    )
    args = parser.parse_args()

    if args.pairs_json:
        pairs = json.loads(Path(args.pairs_json).read_text(encoding="utf-8"))
    else:
        pairs = default_pairs()
    rows = []
    for p in pairs:
        rows.append(
            compare_one(
                gold_path=Path(p["gold"]),
                pred_path=Path(p["pred"]),
                name=p["name"],
            )
        )

    error_keys = [
        "missing_titles",
        "extra_titles",
        "wrong_node_kinds",
        "missing_contains_edges",
        "extra_contains_edges",
        "missing_sequence_edges",
        "extra_sequence_edges",
        "page_grounding_errors",
    ]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "structural_error_report.json").write_text(
        json.dumps({"documents": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_csv(rows, outdir / "structural_error_report.csv", error_keys)
    save_markdown(rows, outdir / "structural_error_report.md", error_keys)
    save_heatmap(rows, outdir / "structural_error_heatmap.png", error_keys)

    print("Saved:")
    print(outdir / "structural_error_report.json")
    print(outdir / "structural_error_report.csv")
    print(outdir / "structural_error_report.md")
    print(outdir / "structural_error_heatmap.png")


if __name__ == "__main__":
    main()
