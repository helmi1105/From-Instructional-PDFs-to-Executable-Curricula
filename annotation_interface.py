#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Tuple, Set
from collections import defaultdict, deque


ALLOWED_KINDS = ["COURSE", "MODULE", "UNIT", "KC"]
KIND_COLORS = {"COURSE": "#d6ecff", "MODULE": "#e9f7df", "UNIT": "#fff3cd", "KC": "#f8d7da"}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_pages(text: str) -> List[int]:
    vals = []
    for p in (text or "").split(","):
        p = p.strip()
        if p:
            vals.append(int(p))
    return sorted(set(vals))


def shorten(text: str, n: int = 42) -> str:
    t = (text or "").strip()
    return t if len(t) <= n else t[: n - 1] + "…"


class AnnotationApp:
    def __init__(self, root: tk.Tk, draft_path: Path, output_path: Path, annotator_id: str) -> None:
        self.root = root
        self.draft_path = draft_path
        self.output_path = output_path
        self.annotator_id = annotator_id
        self.data = load_json(output_path)
        self.node_by_id: Dict[str, Dict[str, Any]] = {}
        self.selected_node_id: str | None = None
        self.selected_edge_idx: int | None = None
        self.drag_node_id: str | None = None
        self.drag_offset: Tuple[float, float] = (0.0, 0.0)
        self.is_panning = False
        self.last_pan_xy: Tuple[int, int] = (0, 0)
        self.edge_connect_mode: str | None = None
        self.edge_source_node: str | None = None
        self.edge_relink_mode = False
        self.edge_relink_step = 0
        self.positions: Dict[str, Tuple[float, float]] = {}
        self.node_shape_items: Dict[str, int] = {}
        self.zoom = 1.0

        self.root.title(f"ECG Annotation Interface (Graphical) - {annotator_id}")
        self.root.geometry("1540x900")
        self._build_ui()
        self._refresh_all()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text=f"Draft (pre-annotation, not gold): {self.draft_path}").pack(side=tk.LEFT)
        self.var_status = tk.StringVar(value="Ready")
        ttk.Label(top, textvariable=self.var_status).pack(side=tk.LEFT, padx=16)
        ttk.Button(top, text="Save", command=self.save).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="Save As...", command=self.save_as_prompt).pack(side=tk.RIGHT, padx=4)

        outer = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(outer, padding=8)
        center = ttk.Frame(outer, padding=8)
        right = ttk.Frame(outer, padding=8)
        outer.add(left, weight=1)
        outer.add(center, weight=3)
        outer.add(right, weight=2)

        ttk.Label(left, text="Nodes").pack(anchor=tk.W)
        self.node_list = tk.Listbox(left, exportselection=False)
        self.node_list.pack(fill=tk.BOTH, expand=True)
        self.node_list.bind("<<ListboxSelect>>", self.on_select_node_from_list)
        btns = ttk.Frame(left)
        btns.pack(fill=tk.X, pady=4)
        ttk.Button(btns, text="Add node", command=self.add_node).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Delete node", command=self.delete_node).pack(side=tk.LEFT, padx=2)

        gbox = ttk.LabelFrame(center, text="Graph Canvas (click/drag/edit)", padding=6)
        gbox.pack(fill=tk.BOTH, expand=True)
        tools = ttk.Frame(gbox)
        tools.pack(fill=tk.X, pady=4)
        ttk.Button(tools, text="Connect contains", command=lambda: self.start_connect_mode("contains")).pack(side=tk.LEFT, padx=2)
        ttk.Button(tools, text="Connect sequence", command=lambda: self.start_connect_mode("sequence")).pack(side=tk.LEFT, padx=2)
        ttk.Button(tools, text="Cancel connect", command=self.cancel_connect_mode).pack(side=tk.LEFT, padx=2)
        ttk.Button(tools, text="Relink edge", command=self.start_relink_mode).pack(side=tk.LEFT, padx=8)
        ttk.Button(tools, text="Set edge=contains", command=lambda: self.set_selected_edge_type("contains")).pack(side=tk.LEFT, padx=2)
        ttk.Button(tools, text="Set edge=sequence", command=lambda: self.set_selected_edge_type("sequence")).pack(side=tk.LEFT, padx=2)
        ttk.Button(tools, text="Delete edge", command=self.delete_edge).pack(side=tk.LEFT, padx=2)
        ttk.Button(tools, text="Auto layout", command=self.auto_layout).pack(side=tk.LEFT, padx=12)
        ttk.Button(tools, text="Zoom +", command=lambda: self.apply_zoom(1.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(tools, text="Zoom -", command=lambda: self.apply_zoom(1 / 1.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(tools, text="Fit", command=self.fit_to_view).pack(side=tk.LEFT, padx=2)
        self.var_zoom = tk.StringVar(value="100%")
        ttk.Label(tools, textvariable=self.var_zoom).pack(side=tk.LEFT, padx=8)

        self.canvas = tk.Canvas(gbox, bg="white", width=900, height=740, scrollregion=(0, 0, 2800, 2800))
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)

        editor = ttk.LabelFrame(right, text="Editor", padding=8)
        editor.pack(fill=tk.X)
        self.var_id = tk.StringVar()
        self.var_title = tk.StringVar()
        self.var_kind = tk.StringVar()
        self.var_pages = tk.StringVar()
        ttk.Label(editor, text="Node ID").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(editor, textvariable=self.var_id, state="readonly", width=40).grid(row=0, column=1, sticky=tk.W, padx=4, pady=2)
        ttk.Label(editor, text="Title").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(editor, textvariable=self.var_title, width=48).grid(row=1, column=1, sticky=tk.W, padx=4, pady=2)
        ttk.Label(editor, text="Kind").grid(row=2, column=0, sticky=tk.W)
        ttk.Combobox(editor, values=ALLOWED_KINDS, textvariable=self.var_kind, width=14, state="readonly").grid(row=2, column=1, sticky=tk.W, padx=4, pady=2)
        ttk.Label(editor, text="Grounding pages").grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(editor, textvariable=self.var_pages, width=48).grid(row=3, column=1, sticky=tk.W, padx=4, pady=2)
        ttk.Button(editor, text="Apply node changes", command=self.apply_node_changes).grid(row=4, column=1, sticky=tk.W, pady=4)

        ebox = ttk.LabelFrame(right, text="Edges", padding=8)
        ebox.pack(fill=tk.BOTH, expand=True, pady=8)
        self.edge_list = tk.Listbox(ebox, exportselection=False)
        self.edge_list.pack(fill=tk.BOTH, expand=True)
        self.edge_list.bind("<<ListboxSelect>>", self.on_select_edge)
        eform = ttk.Frame(ebox)
        eform.pack(fill=tk.X, pady=4)
        self.var_edge_type = tk.StringVar(value="contains")
        self.var_edge_src = tk.StringVar()
        self.var_edge_tgt = tk.StringVar()
        ttk.Combobox(eform, values=["contains", "sequence"], textvariable=self.var_edge_type, width=12, state="readonly").grid(row=0, column=0, padx=2)
        ttk.Entry(eform, textvariable=self.var_edge_src, width=18).grid(row=0, column=1, padx=2)
        ttk.Entry(eform, textvariable=self.var_edge_tgt, width=18).grid(row=0, column=2, padx=2)
        ebtn = ttk.Frame(ebox)
        ebtn.pack(fill=tk.X)
        ttk.Button(ebtn, text="Add edge", command=self.add_edge).pack(side=tk.LEFT, padx=2)
        ttk.Button(ebtn, text="Delete selected edge", command=self.delete_edge).pack(side=tk.LEFT, padx=2)

    def _refresh_all(self) -> None:
        self.node_by_id = {n["id"]: n for n in self.data.get("nodes", []) if n.get("id")}
        self._ensure_positions()
        self._refresh_node_list()
        self._refresh_edge_list()
        self.draw_graph()

    def _refresh_node_list(self) -> None:
        self.node_list.delete(0, tk.END)
        for n in self.data.get("nodes", []):
            self.node_list.insert(tk.END, f'{n["id"]} | {n.get("kind", "")} | {shorten(n.get("title", ""))}')

    def _refresh_edge_list(self) -> None:
        self.edge_list.delete(0, tk.END)
        for i, e in enumerate(self.data.get("edges", [])):
            self.edge_list.insert(tk.END, f'{i}: {e.get("type")} | {e.get("source")} -> {e.get("target")}')

    def _ensure_positions(self) -> None:
        layout = self.data.setdefault("metadata", {}).setdefault("ui_layout", {})
        pos = layout.setdefault("positions", {})
        if pos:
            for nid, xy in pos.items():
                if isinstance(xy, list) and len(xy) == 2:
                    self.positions[nid] = (float(xy[0]), float(xy[1]))
        for n in self.data.get("nodes", []):
            nid = n["id"]
            if nid not in self.positions:
                self.positions[nid] = (120.0, 120.0)
        if not pos:
            self.auto_layout()

    def auto_layout(self) -> None:
        nodes = [n["id"] for n in self.data.get("nodes", []) if n.get("id")]
        children: Dict[str, List[str]] = defaultdict(list)
        indeg: Dict[str, int] = {nid: 0 for nid in nodes}
        for e in self.data.get("edges", []):
            if e.get("type") != "contains":
                continue
            s, t = e.get("source"), e.get("target")
            if s in indeg and t in indeg:
                children[s].append(t)
                indeg[t] += 1

        roots = [nid for nid in nodes if indeg[nid] == 0]
        if not roots and nodes:
            roots = [nodes[0]]

        depth: Dict[str, int] = {}
        q = deque()
        for r in roots:
            depth[r] = 0
            q.append(r)
        while q:
            cur = q.popleft()
            for ch in children.get(cur, []):
                nd = depth[cur] + 1
                if ch not in depth or nd < depth[ch]:
                    depth[ch] = nd
                    q.append(ch)
        for nid in nodes:
            if nid not in depth:
                depth[nid] = max(depth.values(), default=0) + 1

        # Top-down tree layout with subtree widths (root centered, real tree look).
        span_unit = 220.0
        y_start = 120.0
        y_gap = 170.0
        cursor_x = 120.0
        assigned: Set[str] = set()

        def subtree_span(nid: str, seen: Set[str]) -> float:
            if nid in seen:
                return span_unit
            seen.add(nid)
            kids = [k for k in children.get(nid, []) if k not in seen]
            if not kids:
                return span_unit
            total = 0.0
            for k in kids:
                total += subtree_span(k, set(seen))
            return max(span_unit, total)

        def place(nid: str, left_x: float, lvl: int, seen: Set[str]) -> float:
            if nid in seen:
                w = span_unit
                cx = left_x + w / 2.0
                self.positions[nid] = (cx, y_start + lvl * y_gap)
                assigned.add(nid)
                return w
            seen.add(nid)
            kids = [k for k in children.get(nid, []) if k not in seen]
            if not kids:
                w = span_unit
                cx = left_x + w / 2.0
                self.positions[nid] = (cx, y_start + lvl * y_gap)
                assigned.add(nid)
                return w
            widths = [subtree_span(k, set(seen)) for k in kids]
            total = max(span_unit, sum(widths))
            run_x = left_x
            child_centers: List[float] = []
            for k, w in zip(kids, widths):
                place(k, run_x, lvl + 1, set(seen))
                child_centers.append(run_x + w / 2.0)
                run_x += w
            cx = (child_centers[0] + child_centers[-1]) / 2.0 if child_centers else (left_x + total / 2.0)
            self.positions[nid] = (cx, y_start + lvl * y_gap)
            assigned.add(nid)
            return total

        for r in roots:
            w = place(r, cursor_x, 0, set())
            cursor_x += w + 80.0

        # Place unassigned/disconnected nodes below the tree.
        leftovers = [nid for nid in nodes if nid not in assigned]
        if leftovers:
            max_depth = max(depth.values(), default=0)
            y_extra = y_start + (max_depth + 2) * y_gap
            for i, nid in enumerate(leftovers):
                self.positions[nid] = (160.0 + i * span_unit, y_extra)
        self._persist_positions()
        self.draw_graph()

    def _persist_positions(self) -> None:
        pos = {}
        for nid, (x, y) in self.positions.items():
            pos[nid] = [round(float(x), 2), round(float(y), 2)]
        self.data.setdefault("metadata", {}).setdefault("ui_layout", {})["positions"] = pos

    def apply_zoom(self, factor: float) -> None:
        new_zoom = self.zoom * factor
        if new_zoom < 0.2 or new_zoom > 4.0:
            return
        self.zoom = new_zoom
        self.var_zoom.set(f"{int(self.zoom * 100)}%")
        self.draw_graph()

    def fit_to_view(self) -> None:
        if not self.positions:
            return
        xs = [p[0] for p in self.positions.values()]
        ys = [p[1] for p in self.positions.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(200.0, max_x - min_x + 200.0)
        span_y = max(200.0, max_y - min_y + 200.0)
        c_w = max(600.0, float(self.canvas.winfo_width()))
        c_h = max(400.0, float(self.canvas.winfo_height()))
        self.zoom = max(0.2, min(4.0, min(c_w / span_x, c_h / span_y)))
        self.var_zoom.set(f"{int(self.zoom * 100)}%")
        self.draw_graph()

    def on_mousewheel(self, event: tk.Event) -> None:
        if getattr(event, "delta", 0) > 0:
            self.apply_zoom(1.1)
        elif getattr(event, "delta", 0) < 0:
            self.apply_zoom(1 / 1.1)

    def draw_graph(self) -> None:
        self.canvas.delete("all")
        self.node_shape_items.clear()
        for idx, e in enumerate(self.data.get("edges", [])):
            s, t, typ = e.get("source"), e.get("target"), e.get("type")
            if s not in self.positions or t not in self.positions:
                continue
            x1, y1 = self.positions[s][0] * self.zoom, self.positions[s][1] * self.zoom
            x2, y2 = self.positions[t][0] * self.zoom, self.positions[t][1] * self.zoom
            color = "#1f77b4" if typ == "contains" else "#ff7f0e"
            width = 3 if self.selected_edge_idx == idx else 2
            arrow_shape = (
                max(10, int(14 * self.zoom)),
                max(12, int(18 * self.zoom)),
                max(5, int(7 * self.zoom)),
            )
            line = self.canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill=color,
                width=width,
                arrow=tk.LAST,
                arrowshape=arrow_shape,
            )
            self.canvas.addtag_withtag(f"edge:{idx}", line)

        for n in self.data.get("nodes", []):
            nid = n["id"]
            x0, y0 = self.positions.get(nid, (100.0, 100.0))
            x, y = x0 * self.zoom, y0 * self.zoom
            w, h = 160 * self.zoom, 56 * self.zoom
            fill = KIND_COLORS.get(n.get("kind", "KC"), "#f0f0f0")
            outline = "#cc0000" if nid == self.selected_node_id else "#444444"
            rect = self.canvas.create_rectangle(x - w / 2, y - h / 2, x + w / 2, y + h / 2, fill=fill, outline=outline, width=2)
            self.node_shape_items[nid] = rect
            self.canvas.addtag_withtag(f"node:{nid}", rect)
            txt = self.canvas.create_text(x, y - 9 * self.zoom, text=shorten(n.get("title", ""), 34), width=150 * self.zoom, font=("Segoe UI", max(7, int(9 * self.zoom))))
            self.canvas.addtag_withtag(f"node:{nid}", txt)
            t2 = self.canvas.create_text(x, y + 17 * self.zoom, text=f'{nid} | {n.get("kind", "")}', font=("Consolas", max(6, int(8 * self.zoom))), fill="#333333")
            self.canvas.addtag_withtag(f"node:{nid}", t2)

    def _node_from_canvas_item(self, item: int) -> str | None:
        for tag in self.canvas.gettags(item):
            if tag.startswith("node:"):
                return tag.split(":", 1)[1]
        return None

    def _edge_from_canvas_item(self, item: int) -> int | None:
        for tag in self.canvas.gettags(item):
            if tag.startswith("edge:"):
                try:
                    return int(tag.split(":", 1)[1])
                except ValueError:
                    return None
        return None

    def select_edge(self, idx: int) -> None:
        edges = self.data.get("edges", [])
        if idx < 0 or idx >= len(edges):
            return
        self.selected_edge_idx = idx
        e = edges[idx]
        self.var_edge_type.set(e.get("type", "contains"))
        self.var_edge_src.set(e.get("source", ""))
        self.var_edge_tgt.set(e.get("target", ""))
        self.edge_list.selection_clear(0, tk.END)
        self.edge_list.selection_set(idx)
        self.edge_list.see(idx)
        self.draw_graph()

    def on_canvas_click(self, event: tk.Event) -> None:
        item = self.canvas.find_withtag("current")
        if not item:
            self.drag_node_id = None
            self.is_panning = True
            self.last_pan_xy = (event.x, event.y)
            return
        eidx = self._edge_from_canvas_item(item[0])
        if eidx is not None:
            self.select_edge(eidx)
            self.var_status.set(f"Selected edge #{eidx}")
            return
        nid = self._node_from_canvas_item(item[0])
        if not nid:
            return
        self.select_node(nid)
        if self.edge_relink_mode and self.selected_edge_idx is not None:
            edges = self.data.get("edges", [])
            if 0 <= self.selected_edge_idx < len(edges):
                if self.edge_relink_step == 0:
                    edges[self.selected_edge_idx]["source"] = nid
                    self.edge_relink_step = 1
                    self.var_status.set(f"Relink edge #{self.selected_edge_idx}: source={nid}, now click target")
                    self._refresh_edge_list()
                    self.select_edge(self.selected_edge_idx)
                else:
                    edges[self.selected_edge_idx]["target"] = nid
                    self.edge_relink_step = 0
                    self.edge_relink_mode = False
                    self.var_status.set(f"Edge #{self.selected_edge_idx} relinked to {edges[self.selected_edge_idx]['source']} -> {nid}")
                    self._refresh_edge_list()
                    self.select_edge(self.selected_edge_idx)
            return
        if self.edge_connect_mode:
            if self.edge_source_node is None:
                self.edge_source_node = nid
                self.var_status.set(f"Connect {self.edge_connect_mode}: source={nid}, now click target")
            else:
                if nid != self.edge_source_node:
                    self.data.setdefault("edges", []).append(
                        {"source": self.edge_source_node, "target": nid, "type": self.edge_connect_mode}
                    )
                    self.var_status.set(f"Edge added: {self.edge_connect_mode} {self.edge_source_node}->{nid}")
                    self.edge_source_node = None
                    self._refresh_edge_list()
                    self.draw_graph()
            return
        x, y = self.positions.get(nid, (event.x / self.zoom, event.y / self.zoom))
        self.drag_node_id = nid
        self.drag_offset = (x - (event.x / self.zoom), y - (event.y / self.zoom))

    def on_canvas_drag(self, event: tk.Event) -> None:
        if self.is_panning:
            dx = event.x - self.last_pan_xy[0]
            dy = event.y - self.last_pan_xy[1]
            for nid, (x, y) in list(self.positions.items()):
                self.positions[nid] = (x + (dx / self.zoom), y + (dy / self.zoom))
            self.last_pan_xy = (event.x, event.y)
            self._persist_positions()
            self.draw_graph()
            return
        if not self.drag_node_id:
            return
        self.positions[self.drag_node_id] = ((event.x / self.zoom) + self.drag_offset[0], (event.y / self.zoom) + self.drag_offset[1])
        self._persist_positions()
        self.draw_graph()

    def on_canvas_release(self, _event: tk.Event) -> None:
        self.drag_node_id = None
        self.is_panning = False

    def start_connect_mode(self, typ: str) -> None:
        self.edge_connect_mode = typ
        self.edge_source_node = None
        self.var_status.set(f"Connect mode ({typ}): click source node then target node")

    def cancel_connect_mode(self) -> None:
        self.edge_connect_mode = None
        self.edge_source_node = None
        self.edge_relink_mode = False
        self.edge_relink_step = 0
        self.var_status.set("Connect mode canceled")

    def start_relink_mode(self) -> None:
        if self.selected_edge_idx is None:
            messagebox.showerror("Error", "Select an edge first (click an edge line).")
            return
        self.edge_connect_mode = None
        self.edge_source_node = None
        self.edge_relink_mode = True
        self.edge_relink_step = 0
        self.var_status.set(f"Relink edge #{self.selected_edge_idx}: click NEW source node, then NEW target node")

    def set_selected_edge_type(self, edge_type: str) -> None:
        if self.selected_edge_idx is None:
            messagebox.showerror("Error", "Select an edge first.")
            return
        if edge_type not in ("contains", "sequence"):
            return
        edges = self.data.get("edges", [])
        if 0 <= self.selected_edge_idx < len(edges):
            edges[self.selected_edge_idx]["type"] = edge_type
            self.var_edge_type.set(edge_type)
            self._refresh_edge_list()
            self.select_edge(self.selected_edge_idx)

    def select_node(self, nid: str) -> None:
        n = self.node_by_id.get(nid)
        if not n:
            return
        self.selected_node_id = nid
        self.var_id.set(nid)
        self.var_title.set(n.get("title", ""))
        self.var_kind.set(n.get("kind", ""))
        pages = self.data.get("grounding", {}).get(nid, {}).get("pages", [])
        self.var_pages.set(",".join(str(p) for p in pages))
        self.draw_graph()

    def on_select_node_from_list(self, _event: Any = None) -> None:
        sel = self.node_list.curselection()
        if not sel:
            return
        n = self.data.get("nodes", [])[sel[0]]
        self.select_node(n["id"])

    def apply_node_changes(self) -> None:
        if not self.selected_node_id:
            messagebox.showerror("Error", "Select a node first.")
            return
        n = self.node_by_id.get(self.selected_node_id)
        if not n:
            return
        kind = self.var_kind.get().strip()
        if kind not in ALLOWED_KINDS:
            messagebox.showerror("Error", f"Invalid kind: {kind}")
            return
        n["title"] = self.var_title.get().strip()
        n["label"] = n["title"]
        n["kind"] = kind
        self.data.setdefault("grounding", {}).setdefault(self.selected_node_id, {})["pages"] = parse_pages(self.var_pages.get())
        self._refresh_node_list()
        self.draw_graph()

    def add_node(self) -> None:
        idx = len(self.data.get("nodes", [])) + 1
        nid = f"MANUAL_{idx:04d}"
        while nid in self.node_by_id:
            idx += 1
            nid = f"MANUAL_{idx:04d}"
        self.data.setdefault("nodes", []).append(
            {
                "id": nid,
                "kind": "KC",
                "label": "New node",
                "title": "New node",
                "number": "",
                "page_start": None,
                "page_end": None,
                "path": [],
                "children": [],
            }
        )
        self.data.setdefault("grounding", {})[nid] = {"pages": [], "heading_path": [], "support_spans": []}
        self.positions[nid] = (260.0, 220.0)
        self._persist_positions()
        self._refresh_all()
        self.select_node(nid)

    def delete_node(self) -> None:
        if not self.selected_node_id:
            messagebox.showerror("Error", "Select a node first.")
            return
        nid = self.selected_node_id
        self.data["nodes"] = [n for n in self.data.get("nodes", []) if n.get("id") != nid]
        self.data["edges"] = [e for e in self.data.get("edges", []) if e.get("source") != nid and e.get("target") != nid]
        self.data.get("grounding", {}).pop(nid, None)
        self.positions.pop(nid, None)
        self.selected_node_id = None
        self.var_id.set("")
        self.var_title.set("")
        self.var_kind.set("")
        self.var_pages.set("")
        self._persist_positions()
        self._refresh_all()

    def on_select_edge(self, _event: Any = None) -> None:
        sel = self.edge_list.curselection()
        if not sel:
            return
        self.select_edge(sel[0])

    def add_edge(self) -> None:
        typ = self.var_edge_type.get().strip()
        src = self.var_edge_src.get().strip()
        tgt = self.var_edge_tgt.get().strip()
        if typ not in ("contains", "sequence"):
            messagebox.showerror("Error", "Edge type must be contains or sequence.")
            return
        if src not in self.node_by_id or tgt not in self.node_by_id:
            messagebox.showerror("Error", "Source/target IDs must exist.")
            return
        self.data.setdefault("edges", []).append({"source": src, "target": tgt, "type": typ})
        self._refresh_edge_list()
        self.draw_graph()

    def delete_edge(self) -> None:
        if self.selected_edge_idx is None:
            messagebox.showerror("Error", "Select an edge first.")
            return
        edges = self.data.get("edges", [])
        if 0 <= self.selected_edge_idx < len(edges):
            edges.pop(self.selected_edge_idx)
        self.edge_relink_mode = False
        self.edge_relink_step = 0
        self.selected_edge_idx = None
        self._refresh_edge_list()
        self.draw_graph()

    def save(self) -> None:
        self._persist_positions()
        meta = self.data.setdefault("metadata", {})
        meta["annotation_mode"] = "human_corrected_from_draft"
        meta["draft_preannotation_path"] = str(self.draft_path)
        meta["annotator_id"] = self.annotator_id
        save_json(self.output_path, self.data)
        self.var_status.set(f"Saved: {self.output_path}")
        messagebox.showinfo("Saved", str(self.output_path))

    def save_as_prompt(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Save As")
        win.geometry("900x120")
        var_path = tk.StringVar(value=str(self.output_path))
        ttk.Label(win, text="Output path").pack(anchor=tk.W, padx=8, pady=4)
        ttk.Entry(win, textvariable=var_path, width=130).pack(fill=tk.X, padx=8)

        def do_save() -> None:
            self.output_path = Path(var_path.get().strip())
            self.save()
            win.destroy()

        ttk.Button(win, text="Save", command=do_save).pack(anchor=tk.E, padx=8, pady=6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Graphical annotation interface for ECG draft correction.")
    parser.add_argument("--draft", required=True, help="Path to draft ECG JSON (pre-annotation).")
    parser.add_argument("--annotator_id", required=True, help="Annotator ID (A1, A2, A3...).")
    parser.add_argument("--output", required=True, help="Path to annotator output ECG JSON.")
    parser.add_argument("--reset_output", action="store_true", help="Reset output from draft before opening UI.")
    args = parser.parse_args()

    draft = Path(args.draft)
    output = Path(args.output)
    if not draft.exists():
        raise SystemExit(f"Draft not found: {draft}")
    if args.reset_output or not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(draft, output)

    root = tk.Tk()
    AnnotationApp(root, draft, output, args.annotator_id)
    root.mainloop()


if __name__ == "__main__":
    main()
