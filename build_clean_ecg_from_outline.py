import os
import re
import csv
import json
import hashlib
import argparse
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------
def clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\u00a0", " ").split()).strip()


def normalize_key(value: str) -> str:
    value = clean_text(value).lower().replace("’", "'")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def stable_id(prefix: str, *parts: str, size: int = 12) -> str:
    raw = "||".join(clean_text(p) for p in parts if p is not None)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:size]
    return f"{prefix}_{digest}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ------------------------------------------------------------
# Outline normalization
# ------------------------------------------------------------
def normalize_section(section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    title = clean_text(section.get("title"))
    if not title:
        return None

    number = clean_text(section.get("number")) or None
    role = clean_text(section.get("role")) or None

    start_page = section.get("start_page")
    end_page = section.get("end_page")
    if not isinstance(start_page, int):
        start_page = None
    if not isinstance(end_page, int):
        end_page = None

    children_out: List[Dict[str, Any]] = []
    seen = set()

    for child in section.get("sections", []) or []:
        child_norm = normalize_section(child)
        if not child_norm:
            continue

        key = (
            normalize_key(child_norm.get("number") or ""),
            normalize_key(child_norm.get("title") or ""),
            child_norm.get("start_page"),
            child_norm.get("end_page"),
        )
        if key in seen:
            continue
        seen.add(key)
        children_out.append(child_norm)

    return {
        "title": title,
        "number": number,
        "role": role,
        "start_page": start_page,
        "end_page": end_page,
        "sections": children_out,
    }


def recompute_spans_from_children(section: Dict[str, Any]) -> None:
    children = section.get("sections", []) or []

    for child in children:
        recompute_spans_from_children(child)

    own_start = section.get("start_page") if isinstance(section.get("start_page"), int) else None
    own_end = section.get("end_page") if isinstance(section.get("end_page"), int) else None

    child_starts = [c.get("start_page") for c in children if isinstance(c.get("start_page"), int)]
    child_ends = [c.get("end_page") for c in children if isinstance(c.get("end_page"), int)]

    starts = ([own_start] if own_start is not None else []) + child_starts
    ends = ([own_end] if own_end is not None else []) + child_ends

    if starts:
        section["start_page"] = min(starts)
    else:
        section["start_page"] = None

    if ends:
        section["end_page"] = max(ends)
    else:
        section["end_page"] = None

    start_page = section.get("start_page")
    end_page = section.get("end_page")
    if isinstance(start_page, int) and isinstance(end_page, int) and start_page > end_page:
        section["end_page"] = start_page


def normalize_outline(raw_outline: Dict[str, Any]) -> Dict[str, Any]:
    title = clean_text(raw_outline.get("title")) or "Untitled Course"

    sections_out: List[Dict[str, Any]] = []
    seen = set()

    for sec in raw_outline.get("sections", []) or []:
        sec_norm = normalize_section(sec)
        if not sec_norm:
            continue

        key = (
            normalize_key(sec_norm.get("number") or ""),
            normalize_key(sec_norm.get("title") or ""),
            sec_norm.get("start_page"),
            sec_norm.get("end_page"),
        )
        if key in seen:
            continue
        seen.add(key)
        sections_out.append(sec_norm)

    outline = {
        "title": title,
        "sections": sections_out,
    }

    root_wrapper = {
        "title": title,
        "number": None,
        "role": None,
        "start_page": raw_outline.get("start_page") if isinstance(raw_outline.get("start_page"), int) else None,
        "end_page": raw_outline.get("end_page") if isinstance(raw_outline.get("end_page"), int) else None,
        "sections": sections_out,
    }

    recompute_spans_from_children(root_wrapper)

    outline["start_page"] = root_wrapper.get("start_page")
    outline["end_page"] = root_wrapper.get("end_page")
    outline["sections"] = root_wrapper.get("sections", [])

    return outline


# ------------------------------------------------------------
# ECG typing + construction
# ------------------------------------------------------------
ROMAN_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
DECIMAL_RE = re.compile(r"^\d+(?:\.\d+)+$")


def page_list(start_page: Optional[int], end_page: Optional[int]) -> List[int]:
    if isinstance(start_page, int) and isinstance(end_page, int) and start_page <= end_page:
        return list(range(start_page, end_page + 1))
    if isinstance(start_page, int):
        return [start_page]
    if isinstance(end_page, int):
        return [end_page]
    return []


def classify_section_kind(section: Dict[str, Any], parent_kind: str) -> str:
    has_children = bool(section.get("sections") or [])

    if parent_kind == "COURSE":
        return "MODULE" if has_children else "KC"

    if has_children:
        return "UNIT"

    return "KC"


def make_node(
    node_id: str,
    title: str,
    kind: str,
    number: Optional[str],
    role: Optional[str],
    page_start: Optional[int],
    page_end: Optional[int],
    path: List[str],
) -> Dict[str, Any]:
    return {
        "id": node_id,
        "label": title,
        "kind": kind,
        "title": title,
        "number": number,
        "role": role,
        "page_start": page_start,
        "page_end": page_end,
        "evidence_pages": page_list(page_start, page_end),
        "path": path,
    }


def make_edge(source: str, target: str, edge_type: str) -> Dict[str, Any]:
    return {"source": source, "target": target, "type": edge_type}


def build_ecg_from_outline_clean(outline: Dict[str, Any]) -> Dict[str, Any]:
    course_title = clean_text(outline.get("title")) or "Untitled Course"
    course_start = outline.get("start_page") if isinstance(outline.get("start_page"), int) else None
    course_end = outline.get("end_page") if isinstance(outline.get("end_page"), int) else None

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    course_id = stable_id("COURSE", course_title)
    nodes.append(
        make_node(
            node_id=course_id,
            title=course_title,
            kind="COURSE",
            number=None,
            role=None,
            page_start=course_start,
            page_end=course_end,
            path=[course_title],
        )
    )

    def walk_sections(sections, parent_id, parent_kind, parent_path):
        ordered_child_ids = []

        for sec in sections:
            sec_title = clean_text(sec.get("title"))
            if not sec_title:
                continue

            sec_number = clean_text(sec.get("number"))
            sec_role = clean_text(sec.get("role"))
            page_start = sec.get("start_page") if isinstance(sec.get("start_page"), int) else None
            page_end = sec.get("end_page") if isinstance(sec.get("end_page"), int) else None
            children = sec.get("sections", []) or []

            kind = classify_section_kind(sec, parent_kind=parent_kind)
            path = parent_path + [sec_title]

            node_id = stable_id(
                kind,
                *path,
                sec_number or "",
                str(page_start or ""),
                str(page_end or "")
            )

            nodes.append(
                make_node(
                    node_id=node_id,
                    title=sec_title,
                    kind=kind,
                    number=sec_number,
                    role=sec_role,
                    page_start=page_start,
                    page_end=page_end,
                    path=path,
                )
            )

            edges.append(make_edge(parent_id, node_id, "contains"))
            ordered_child_ids.append(node_id)

            if children:
                walk_sections(
                    sections=children,
                    parent_id=node_id,
                    parent_kind=kind,
                    parent_path=path,
                )

        for i in range(len(ordered_child_ids) - 1):
            edges.append(make_edge(ordered_child_ids[i], ordered_child_ids[i + 1], "sequence"))

    walk_sections(
        sections=outline.get("sections", []) or [],
        parent_id=course_id,
        parent_kind="COURSE",
        parent_path=[course_title],
    )

    ecg = {
        "metadata": {
            "source": "outline_clean.json",
            "course_title": course_title,
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
        "nodes": nodes,
        "edges": edges,
    }
    ecg["validation"] = validate_ecg(ecg)
    ecg["metadata"]["validation_ok"] = ecg["validation"]["ok"]
    ecg["metadata"]["violation_count"] = len(ecg["validation"]["violations"])
    return ecg


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------
def validate_ecg(ecg: Dict[str, Any]) -> Dict[str, Any]:
    nodes = ecg.get("nodes", []) or []
    edges = ecg.get("edges", []) or []

    violations: List[Dict[str, Any]] = []

    allowed_node_kinds = {"COURSE", "MODULE", "UNIT", "KC"}
    allowed_edge_types = {"contains", "sequence"}
    allowed_contains = {
        ("COURSE", "MODULE"),
        ("COURSE", "KC"),
        ("MODULE", "UNIT"),
        ("MODULE", "KC"),
        ("UNIT", "UNIT"),
        ("UNIT", "KC"),
    }

    node_by_id: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        node_id = n.get("id")
        if node_id in node_by_id:
            violations.append({"type": "duplicate_node_id", "node_id": node_id})
        elif node_id:
            node_by_id[node_id] = n

    contains_by_child: Dict[str, List[str]] = {}
    contains_by_parent: Dict[str, List[str]] = {}

    for e in edges:
        src = e.get("source")
        tgt = e.get("target")
        etype = e.get("type")

        if etype not in allowed_edge_types:
            violations.append({"type": "invalid_edge_type", "edge": e})

        if src not in node_by_id:
            violations.append({"type": "missing_source_node", "edge": e})
        if tgt not in node_by_id:
            violations.append({"type": "missing_target_node", "edge": e})

        if etype == "contains" and src in node_by_id and tgt in node_by_id:
            contains_by_child.setdefault(tgt, []).append(src)
            contains_by_parent.setdefault(src, []).append(tgt)

            src_kind = node_by_id[src].get("kind")
            tgt_kind = node_by_id[tgt].get("kind")
            if (src_kind, tgt_kind) not in allowed_contains:
                violations.append({
                    "type": "invalid_contains_hierarchy",
                    "edge": e,
                    "source_kind": src_kind,
                    "target_kind": tgt_kind,
                })

    roots = [n for n in nodes if n.get("kind") == "COURSE"]
    if len(roots) != 1:
        violations.append({"type": "course_root_count", "expected": 1, "actual": len(roots)})

    for n in nodes:
        node_id = n.get("id")
        title = clean_text(n.get("title"))
        kind = n.get("kind")
        start_page = n.get("page_start")
        end_page = n.get("page_end")
        path = n.get("path")
        child_ids = contains_by_parent.get(node_id, [])

        if not node_id:
            violations.append({"type": "missing_node_id", "node": n})

        if kind not in allowed_node_kinds:
            violations.append({"type": "invalid_node_kind", "node_id": node_id, "kind": kind})

        if not title:
            violations.append({"type": "empty_title", "node_id": node_id})

        if kind != "COURSE":
            parents = contains_by_child.get(node_id, [])
            if len(parents) != 1:
                violations.append({"type": "parent_count", "node_id": node_id, "parent_count": len(parents)})
        else:
            parents = contains_by_child.get(node_id, [])
            if parents:
                violations.append({"type": "course_has_parent", "node_id": node_id, "parent_count": len(parents)})

        if isinstance(start_page, int) and isinstance(end_page, int) and start_page > end_page:
            violations.append({"type": "invalid_page_range", "node_id": node_id, "start": start_page, "end": end_page})

        if kind == "KC" and child_ids:
            violations.append({
                "type": "kc_has_children",
                "node_id": node_id,
                "child_count": len(child_ids),
            })

        if kind == "UNIT" and not child_ids:
            violations.append({
                "type": "empty_unit",
                "node_id": node_id,
            })

        if kind == "MODULE" and not child_ids:
            violations.append({
                "type": "empty_module",
                "node_id": node_id,
            })

        if kind == "KC":
            pages = n.get("evidence_pages") or []
            has_grounding = bool(pages) or isinstance(start_page, int) or isinstance(end_page, int)
            if not has_grounding:
                violations.append({
                    "type": "kc_missing_page_grounding",
                    "node_id": node_id,
                })

        if not isinstance(path, list) or not path:
            violations.append({"type": "invalid_path", "node_id": node_id})
        else:
            if clean_text(path[-1]) != title:
                violations.append({
                    "type": "path_title_mismatch",
                    "node_id": node_id,
                    "path_last": path[-1],
                    "title": title,
                })

    for parent_id, child_ids in contains_by_parent.items():
        seen_titles = set()
        parent = node_by_id[parent_id]
        parent_start = parent.get("page_start")
        parent_end = parent.get("page_end")

        for child_id in child_ids:
            child = node_by_id[child_id]
            child_title_key = normalize_key(child.get("title") or "")
            if child_title_key in seen_titles:
                violations.append({"type": "duplicate_sibling_title", "parent_id": parent_id, "child_id": child_id})
            seen_titles.add(child_title_key)

            child_start = child.get("page_start")
            child_end = child.get("page_end")

            if isinstance(parent_start, int) and isinstance(child_start, int) and child_start < parent_start:
                violations.append({"type": "child_starts_before_parent", "parent_id": parent_id, "child_id": child_id})
            if isinstance(parent_end, int) and isinstance(child_end, int) and child_end > parent_end:
                violations.append({"type": "child_ends_after_parent", "parent_id": parent_id, "child_id": child_id})

    for child_id, parents in contains_by_child.items():
        if len(parents) == 1:
            parent_id = parents[0]
            parent_path = node_by_id[parent_id].get("path", [])
            child_path = node_by_id[child_id].get("path", [])
            if not isinstance(parent_path, list) or not isinstance(child_path, list):
                continue
            if child_path[:len(parent_path)] != parent_path:
                violations.append({
                    "type": "path_parent_prefix_mismatch",
                    "parent_id": parent_id,
                    "child_id": child_id,
                })

    seen_sequence_edges = set()
    sequence_edges_by_parent: Dict[str, List[tuple]] = {}

    for e in edges:
        if e.get("type") != "sequence":
            continue

        src = e.get("source")
        tgt = e.get("target")

        if src == tgt:
            violations.append({"type": "sequence_self_loop", "edge": e})

        seq_key = (src, tgt)
        if seq_key in seen_sequence_edges:
            violations.append({"type": "duplicate_sequence_edge", "edge": e})
        seen_sequence_edges.add(seq_key)

        if src not in node_by_id or tgt not in node_by_id:
            continue

        src_parents = contains_by_child.get(src, [])
        tgt_parents = contains_by_child.get(tgt, [])

        if len(src_parents) != 1 or len(tgt_parents) != 1 or src_parents[0] != tgt_parents[0]:
            violations.append({"type": "sequence_not_between_siblings", "edge": e})
            continue

        parent_id = src_parents[0]
        siblings = contains_by_parent.get(parent_id, [])

        try:
            src_index = siblings.index(src)
            tgt_index = siblings.index(tgt)
        except ValueError:
            violations.append({"type": "sequence_nodes_not_in_parent_children", "edge": e})
            continue

        if tgt_index != src_index + 1:
            violations.append({
                "type": "sequence_not_adjacent",
                "edge": e,
                "parent_id": parent_id,
                "source_index": src_index,
                "target_index": tgt_index,
            })

        sequence_edges_by_parent.setdefault(parent_id, []).append((src, tgt))

    def has_cycle_in_sequence(seq_edges: List[tuple]) -> bool:
        graph: Dict[str, List[str]] = {}
        nodes_in_graph = set()

        for a, b in seq_edges:
            graph.setdefault(a, []).append(b)
            nodes_in_graph.add(a)
            nodes_in_graph.add(b)

        visited = set()
        stack = set()

        def dfs(node: str) -> bool:
            if node in stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            stack.add(node)
            for nei in graph.get(node, []):
                if dfs(nei):
                    return True
            stack.remove(node)
            return False

        for node in nodes_in_graph:
            if dfs(node):
                return True
        return False

    for parent_id, seq_edges in sequence_edges_by_parent.items():
        if has_cycle_in_sequence(seq_edges):
            violations.append({
                "type": "sequence_cycle",
                "parent_id": parent_id,
            })

    return {"ok": len(violations) == 0, "violations": violations}


# ------------------------------------------------------------
# Exports
# ------------------------------------------------------------
def export_graphviz_dot(ecg: Dict[str, Any], out_path: str) -> None:
    kind_style = {
        "COURSE": {"shape": "box", "color": "lightblue", "style": "filled,bold"},
        "MODULE": {"shape": "box", "color": "lightgreen", "style": "filled"},
        "UNIT": {"shape": "ellipse", "color": "gold", "style": "filled"},
        "KC": {"shape": "note", "color": "lightpink", "style": "filled"},
    }

    lines = []
    lines.append("digraph ECG {")
    lines.append('  rankdir=TB;')
    lines.append('  graph [fontsize=10, overlap=false, splines=true];')
    lines.append('  node [fontsize=10];')
    lines.append('  edge [fontsize=9];')

    for node in ecg.get("nodes", []):
        node_id = node["id"]
        label = clean_text(node.get("label", node_id)).replace('"', '\\"')
        kind = node.get("kind", "KC")
        style = kind_style.get(kind, {"shape": "box", "color": "white", "style": "filled"})
        lines.append(
            f'  "{node_id}" [label="{label}", shape="{style["shape"]}", style="{style["style"]}", fillcolor="{style["color"]}"];'
        )

    for edge in ecg.get("edges", []):
        src = edge["source"]
        tgt = edge["target"]
        etype = edge.get("type", "contains")
        if etype == "contains":
            lines.append(f'  "{src}" -> "{tgt}" [label="contains"];')
        else:
            lines.append(f'  "{src}" -> "{tgt}" [style=dashed, label="sequence"];')

    lines.append("}")
    write_text(out_path, "\n".join(lines))


def export_nodes_csv(ecg: Dict[str, Any], out_path: str) -> None:
    fields = ["id", "kind", "title", "number", "page_start", "page_end", "path"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for n in ecg.get("nodes", []):
            writer.writerow({
                "id": n.get("id"),
                "kind": n.get("kind"),
                "title": n.get("title"),
                "number": n.get("number"),
                "page_start": n.get("page_start"),
                "page_end": n.get("page_end"),
                "path": " > ".join(n.get("path", [])),
            })


def export_edges_csv(ecg: Dict[str, Any], out_path: str) -> None:
    fields = ["source", "target", "type"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in ecg.get("edges", []):
            writer.writerow({
                "source": e.get("source"),
                "target": e.get("target"),
                "type": e.get("type"),
            })


def build_summary(ecg: Dict[str, Any], outline_clean: Dict[str, Any]) -> Dict[str, Any]:
    nodes = ecg.get("nodes", []) or []
    edges = ecg.get("edges", []) or []

    counts_by_edge_type: Dict[str, int] = {}
    for e in edges:
        etype = e.get("type", "unknown")
        counts_by_edge_type[etype] = counts_by_edge_type.get(etype, 0) + 1

    path_lengths = [
        len(n.get("path", []))
        for n in nodes
        if isinstance(n.get("path"), list) and n.get("path")
    ]
    max_hierarchy_depth = max(path_lengths) if path_lengths else 0

    depth_by_kind: Dict[str, Dict[str, float]] = {}
    for kind in ["COURSE", "MODULE", "UNIT", "KC"]:
        kind_depths = [
            len(n.get("path", []))
            for n in nodes
            if n.get("kind") == kind and isinstance(n.get("path"), list) and n.get("path")
        ]
        if kind_depths:
            depth_by_kind[kind] = {
                "min": min(kind_depths),
                "max": max(kind_depths),
                "avg": round(sum(kind_depths) / len(kind_depths), 2),
            }
        else:
            depth_by_kind[kind] = {
                "min": 0,
                "max": 0,
                "avg": 0.0,
            }

    kc_nodes = [n for n in nodes if n.get("kind") == "KC"]
    grounded_kcs = 0
    for n in kc_nodes:
        pages = n.get("evidence_pages") or []
        start_page = n.get("page_start")
        end_page = n.get("page_end")

        has_grounding = bool(pages) or isinstance(start_page, int) or isinstance(end_page, int)
        if has_grounding:
            grounded_kcs += 1

    total_kcs = len(kc_nodes)
    kc_grounding_percent = round((grounded_kcs / total_kcs) * 100, 2) if total_kcs > 0 else 0.0

    return {
        "outline_top_sections": len(outline_clean.get("sections", [])),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "counts_by_kind": {
            "COURSE": sum(1 for n in nodes if n.get("kind") == "COURSE"),
            "MODULE": sum(1 for n in nodes if n.get("kind") == "MODULE"),
            "UNIT": sum(1 for n in nodes if n.get("kind") == "UNIT"),
            "KC": sum(1 for n in nodes if n.get("kind") == "KC"),
        },
        "counts_by_edge_type": counts_by_edge_type,
        "hierarchy_depth": {
            "max_path_length": max_hierarchy_depth,
            "depth_by_kind": depth_by_kind,
        },
        "kc_grounding_coverage": {
            "grounded_kcs": grounded_kcs,
            "total_kcs": total_kcs,
            "percent": kc_grounding_percent,
        },
        "validation_ok": bool(ecg.get("validation", {}).get("ok")),
        "violation_count": len(ecg.get("validation", {}).get("violations", [])),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def build_clean_ecg(outline_json: str, outdir: str) -> None:
    ensure_dir(outdir)

    print("[1/4] Loading outline")
    raw_outline = load_json(outline_json)

    print("[2/4] Normalizing outline and recomputing spans upward")
    outline_clean = normalize_outline(raw_outline)
    write_json(os.path.join(outdir, "outline_clean_normalized.json"), outline_clean)

    print("[3/4] Building clean ECG")
    ecg = build_ecg_from_outline_clean(outline_clean)
    write_json(os.path.join(outdir, "ecg.json"), ecg)
    write_json(os.path.join(outdir, "validation.json"), ecg.get("validation", {}))

    print("[4/4] Exporting bundle")
    export_graphviz_dot(ecg, os.path.join(outdir, "ecg.dot"))
    export_nodes_csv(ecg, os.path.join(outdir, "nodes.csv"))
    export_edges_csv(ecg, os.path.join(outdir, "edges.csv"))
    write_json(os.path.join(outdir, "summary.json"), build_summary(ecg, outline_clean))

    print("\nDone.")
    print(f"Output folder: {outdir}")
    print(f"Nodes: {len(ecg['nodes'])}")
    print(f"Edges: {len(ecg['edges'])}")
    print(f"Validation OK: {ecg['validation']['ok']}")
    print(f"Violations: {len(ecg['validation']['violations'])}")


def main():
    parser = argparse.ArgumentParser(description="Build a clean ECG from outline_clean.json")
    parser.add_argument("--outline_json", required=True, help="Path to outline_clean.json")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args()

    build_clean_ecg(args.outline_json, args.outdir)


if __name__ == "__main__":
    main()