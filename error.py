import os
import json
import re
import tempfile
import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image
from graphviz import Digraph

# ============================================================
# CONFIG
# ============================================================
DEFAULT_OUT_NAME = "figure_qualitative_error_analysis"

FIG_W = 18
FIG_H = 6.2
DPI = 300

DEFAULT_FOCUS_1_PARENT = "definition du langage cartographique"
DEFAULT_FOCUS_2_PARENT = "1.2 configuration de la zone d'intervention"
DEFAULT_FOCUS_2_CONTEXT_PARENT = "i - analyse de la zone d'intervention"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 20,
    "axes.titlesize": 17,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
})

# ============================================================
# COLORS
# ============================================================
C_MISSING_NODE = "#fee2e2"
C_MISSING_EDGE = "#dc2626"

C_EXTRA_NODE = "#dbeafe"
C_EXTRA_EDGE = "#2563eb"

C_WRONG_KIND_NODE = "#fef3c7"
C_WRONG_KIND_EDGE = "#d97706"

C_CONTEXT_NODE = "#f3f4f6"
C_CONTEXT_BORDER = "#6b7280"

C_MISSING_SEQ = "#e11d48"
C_EXTRA_SEQ = "#0891b2"

# ============================================================
# HELPERS
# ============================================================
def norm(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = s.replace("’", "'").replace("`", "'")
    s = re.sub(r"\s+", " ", s)
    return s


def safe_id(s: str) -> str:
    s = norm(s)
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    return f"n_{s}"


def wrap_label(text: str, width: int = 26) -> str:
    words = text.split()
    lines = []
    cur = []
    cur_len = 0
    for w in words:
        extra = len(w) + (1 if cur else 0)
        if cur_len + extra <= width:
            cur.append(w)
            cur_len += extra
        else:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_graphviz_to_png(graph: Digraph, out_no_ext: str) -> str:
    graph.format = "png"
    graph.render(out_no_ext, cleanup=True)
    return out_no_ext + ".png"


# ============================================================
# DATA EXTRACTION
# ============================================================
def extract_error_summary(diff_data: dict):
    diffs = diff_data["details"]["differences"]
    kind_per_title = diff_data["details"]["kind_agreement_per_title"]

    wrong_kind_titles = [
        t for t, info in kind_per_title.items()
        if not info.get("match", True)
    ]

    summary = Counter()
    summary["Missing titles"] = len(diffs.get("missing_titles", []))
    summary["Extra titles"] = len(diffs.get("extra_titles", []))
    summary["Wrong kind"] = len(wrong_kind_titles)
    summary["Missing contains"] = len(diffs.get("missing_contains", []))
    summary["Extra contains"] = len(diffs.get("extra_contains", []))
    summary["Missing sequence"] = len(diffs.get("missing_sequence", []))
    summary["Extra sequence"] = len(diffs.get("extra_sequence", []))
    return summary


def extract_wrong_kind_map(diff_data: dict):
    kind_per_title = diff_data["details"]["kind_agreement_per_title"]
    wrong_kind = {}
    for t, info in kind_per_title.items():
        if not info.get("match", True):
            wrong_kind[norm(t)] = {
                "gold": ", ".join(info.get("gold_kinds", [])),
                "pred": ", ".join(info.get("pred_kinds", []))
            }
    return wrong_kind


def extract_differences(diff_data: dict):
    diffs = diff_data["details"]["differences"]
    return {
        "missing_titles": set(norm(x) for x in diffs.get("missing_titles", [])),
        "extra_titles": set(norm(x) for x in diffs.get("extra_titles", [])),
        "missing_contains": [(norm(a), norm(b)) for a, b in diffs.get("missing_contains", [])],
        "extra_contains": [(norm(a), norm(b)) for a, b in diffs.get("extra_contains", [])],
        "missing_sequence": [(norm(a), norm(b)) for a, b in diffs.get("missing_sequence", [])],
        "extra_sequence": [(norm(a), norm(b)) for a, b in diffs.get("extra_sequence", [])],
    }


# ============================================================
# PANEL A
# ============================================================
def make_error_bar_chart(ax, summary: Counter):
    labels = list(summary.keys())
    values = [summary[k] for k in labels]

    colors = [
        C_MISSING_EDGE,
        C_EXTRA_EDGE,
        C_WRONG_KIND_EDGE,
        C_MISSING_EDGE,
        C_EXTRA_EDGE,
        C_MISSING_SEQ,
        C_EXTRA_SEQ,
    ]

    x = range(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)

    ax.set_title("(a) Error counts by type")
    ax.set_ylabel("Count")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    for rect, v in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.15,
            str(v),
            ha="center",
            va="bottom",
            fontsize=9
        )


# ============================================================
# NODE DRAWING
# ============================================================
def node_style(title_n: str, wrong_kind_map: dict, missing_titles: set, extra_titles: set):
    if title_n in missing_titles:
        return {
            "fillcolor": C_MISSING_NODE,
            "color": C_MISSING_EDGE,
            "penwidth": "2.2",
            "style": "rounded,filled"
        }
    if title_n in extra_titles:
        return {
            "fillcolor": C_EXTRA_NODE,
            "color": C_EXTRA_EDGE,
            "penwidth": "2.2",
            "style": "rounded,filled"
        }
    if title_n in wrong_kind_map:
        return {
            "fillcolor": C_WRONG_KIND_NODE,
            "color": C_WRONG_KIND_EDGE,
            "penwidth": "2.3",
            "style": "rounded,filled,bold"
        }
    return {
        "fillcolor": C_CONTEXT_NODE,
        "color": C_CONTEXT_BORDER,
        "penwidth": "1.3",
        "style": "rounded,filled"
    }


def add_node(g, title_n: str, wrong_kind_map: dict, missing_titles: set, extra_titles: set, width: int = 26):
    st = node_style(title_n, wrong_kind_map, missing_titles, extra_titles)
    label = title_n

    if title_n in wrong_kind_map:
        gold_kind = wrong_kind_map[title_n]["gold"]
        pred_kind = wrong_kind_map[title_n]["pred"]
        label = f"{wrap_label(label, width)}\nkind: {gold_kind} → {pred_kind}"
    else:
        label = wrap_label(label, width)

    g.node(
        safe_id(title_n),
        label=label,
        shape="box",
        fontname="Helvetica",
        fontsize="20",
        margin="0.14,0.10",
        **st
    )


# ============================================================
# PANEL B
# ============================================================
def make_local_leaf_omission_graph(diff_data: dict, parent_title: str) -> Digraph:
    d = extract_differences(diff_data)
    wrong_kind_map = extract_wrong_kind_map(diff_data)

    parent_n = norm(parent_title)
    missing_children = [c for p, c in d["missing_contains"] if p == parent_n]
    extra_children = [c for p, c in d["extra_contains"] if p == parent_n]

    g = Digraph("leaf_omission")
    g.attr(rankdir="TB", layout="dot", nodesep="0.70", ranksep="1.05", splines="ortho")
    g.attr(label="(b) Local example: missing fine-grained leaves", labelloc="t", fontsize="26")

    add_node(g, parent_n, wrong_kind_map, d["missing_titles"], d["extra_titles"], width=24)

    for child in missing_children:
        add_node(g, child, wrong_kind_map, d["missing_titles"], d["extra_titles"], width=20)
        g.edge(
            safe_id(parent_n), safe_id(child),
            color=C_MISSING_EDGE, style="dashed", penwidth="2.3"
        )

    for child in extra_children:
        add_node(g, child, wrong_kind_map, d["missing_titles"], d["extra_titles"], width=20)
        g.edge(
            safe_id(parent_n), safe_id(child),
            color=C_EXTRA_EDGE, style="dotted", penwidth="2.3"
        )

    return g


# ============================================================
# PANEL C
# ============================================================
def make_local_attachment_graph(diff_data: dict, gold_parent: str, wrong_parent: str) -> Digraph:
    d = extract_differences(diff_data)
    wrong_kind_map = extract_wrong_kind_map(diff_data)

    gold_parent_n = norm(gold_parent)
    wrong_parent_n = norm(wrong_parent)

    missing_children = [c for p, c in d["missing_contains"] if p == gold_parent_n]
    extra_children_gold_parent = [c for p, c in d["extra_contains"] if p == gold_parent_n]
    extra_children_wrong_parent = [c for p, c in d["extra_contains"] if p == wrong_parent_n]

    focus_children = sorted(set(missing_children) | set(extra_children_gold_parent) | set(extra_children_wrong_parent))

    g = Digraph("attachment_error")
    g.attr(rankdir="TB", layout="dot", nodesep="0.40", ranksep="1.05", splines="ortho")
    g.attr(label="(c) Local example: typing and parent--child attachment error", labelloc="t", fontsize="26")

    add_node(g, gold_parent_n, wrong_kind_map, d["missing_titles"], d["extra_titles"], width=20)
    add_node(g, wrong_parent_n, wrong_kind_map, d["missing_titles"], d["extra_titles"], width=20)

    for child in focus_children:
        add_node(g, child, wrong_kind_map, d["missing_titles"], d["extra_titles"], width=20)

        if (gold_parent_n, child) in d["missing_contains"]:
            g.edge(
                safe_id(gold_parent_n), safe_id(child),
                color=C_MISSING_EDGE, style="dashed", penwidth="2.3"
            )

        if (gold_parent_n, child) in d["extra_contains"]:
            g.edge(
                safe_id(gold_parent_n), safe_id(child),
                color=C_EXTRA_EDGE, style="dotted", penwidth="2.3"
            )

        if (wrong_parent_n, child) in d["extra_contains"]:
            g.edge(
                safe_id(wrong_parent_n), safe_id(child),
                color=C_EXTRA_EDGE, style="dotted", penwidth="2.3"
            )

    return g


# ============================================================
# LEGEND
# ============================================================
def add_graphical_legend(fig):
    legend_handles = [
        Patch(facecolor=C_MISSING_NODE, edgecolor=C_MISSING_EDGE, label="Missing node"),
        Patch(facecolor=C_EXTRA_NODE, edgecolor=C_EXTRA_EDGE, label="Extra node"),
        Patch(facecolor=C_WRONG_KIND_NODE, edgecolor=C_WRONG_KIND_EDGE, label="Wrong kind"),
        Patch(facecolor=C_CONTEXT_NODE, edgecolor=C_CONTEXT_BORDER, label="Matched node"),
        Line2D([0], [0], color=C_MISSING_EDGE, lw=3, linestyle="--", label="Missing contains"),
        Line2D([0], [0], color=C_EXTRA_EDGE, lw=3, linestyle=":", label="Extra contains"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
        fontsize=12,
        handlelength=2.2,
        columnspacing=1.5
    )


# ============================================================
# MAIN FIGURE
# ============================================================
def build_article_figure(
    diff_json_path: str,
    out_dir: str,
    out_name: str,
    focus_1_parent: str,
    focus_2_parent: str,
    focus_2_context_parent: str,
):
    diff_data = load_json(diff_json_path)
    summary = extract_error_summary(diff_data)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        g1 = make_local_leaf_omission_graph(diff_data, focus_1_parent)
        panel_b_path = render_graphviz_to_png(g1, os.path.join(tmpdir, "panel_b"))

        g2 = make_local_attachment_graph(diff_data, focus_2_parent, focus_2_context_parent)
        panel_c_path = render_graphviz_to_png(g2, os.path.join(tmpdir, "panel_c"))

        img_b = Image.open(panel_b_path)
        img_c = Image.open(panel_c_path)

        fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
        gs = GridSpec(
            1, 3,
            figure=fig,
            width_ratios=[0.80, 1.25, 1.45],
            wspace=0.05
        )

        ax0 = fig.add_subplot(gs[0, 0])
        make_error_bar_chart(ax0, summary)

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(img_b)
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 2])
        ax2.imshow(img_c)
        ax2.axis("off")

        fig.suptitle(
            "Qualitative structural error analysis for D1",
            fontsize=18,
            y=0.99
        )

        add_graphical_legend(fig)

        png_path = os.path.join(out_dir, out_name + ".png")
        pdf_path = os.path.join(out_dir, out_name + ".pdf")

        fig.savefig(png_path, bbox_inches="tight", dpi=DPI)
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        print("Saved:")
        print(" -", png_path)
        print(" -", pdf_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build qualitative ECG structural error figure.")
    parser.add_argument("--diff_json", required=True, help="Path to evaluation_results.json")
    parser.add_argument("--outdir", required=True, help="Output directory for figure files")
    parser.add_argument("--out_name", default=DEFAULT_OUT_NAME, help="Output filename prefix")
    parser.add_argument("--focus_1_parent", default=DEFAULT_FOCUS_1_PARENT, help="Panel B parent title")
    parser.add_argument("--focus_2_parent", default=DEFAULT_FOCUS_2_PARENT, help="Panel C gold parent title")
    parser.add_argument(
        "--focus_2_context_parent",
        default=DEFAULT_FOCUS_2_CONTEXT_PARENT,
        help="Panel C incorrect/context parent title",
    )
    args = parser.parse_args()

    build_article_figure(
        diff_json_path=args.diff_json,
        out_dir=args.outdir,
        out_name=args.out_name,
        focus_1_parent=args.focus_1_parent,
        focus_2_parent=args.focus_2_parent,
        focus_2_context_parent=args.focus_2_context_parent,
    )


if __name__ == "__main__":
    main()
