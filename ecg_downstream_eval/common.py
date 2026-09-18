from __future__ import annotations
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

def load_json(path: str | Path) -> Any:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[\]}])", r"\1", raw)
        return json.loads(cleaned)

def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out

def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")

def norm(s: Any) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    # Preserve numbering and word initials; exact downstream scoring is conservative.
    return s.strip()

def page_set(x: Any) -> set[int]:
    if x is None:
        return set()
    if isinstance(x, int):
        return {x}
    if isinstance(x, str):
        vals = set()
        for a, b in re.findall(r"(\d+)(?:\s*[-–]\s*(\d+))?", x):
            aa = int(a); bb = int(b) if b else aa
            vals.update(range(min(aa, bb), max(aa, bb)+1))
        return vals
    if isinstance(x, (list, tuple, set)):
        vals = set()
        for item in x:
            vals |= page_set(item)
        return vals
    return set()

def prf(pred: Sequence[int], gold: Sequence[int]) -> tuple[float,float,float]:
    p, g = set(pred), set(gold)
    if not p and not g:
        return (1.0, 1.0, 1.0)
    precision = len(p & g) / len(p) if p else 0.0
    recall = len(p & g) / len(g) if g else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    return precision, recall, f1

def safe_mean(xs: Sequence[float]) -> float:
    return sum(xs)/len(xs) if xs else 0.0

def estimate_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text.split()) * 1.33))

def extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError(f"No JSON object found in model output: {text[:300]}")
        return json.loads(m.group(0))

def graph_nodes(data: Any) -> list[dict]:
    """Best-effort adapter for the repository gold/predicted ECG JSON formats."""
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("nodes", "items", "sections"):
        val = data.get(key)
        if isinstance(val, list):
            return [x for x in val if isinstance(x, dict)]
    # nested graph wrapper
    for key in ("graph", "ecg", "gold"):
        val = data.get(key)
        if isinstance(val, dict):
            nodes = graph_nodes(val)
            if nodes:
                return nodes
    return []

def graph_edges(data: Any) -> list[tuple[str,str,str]]:
    edges: list[tuple[str,str,str]] = []
    if not isinstance(data, dict):
        return edges

    raw_edges = data.get("edges")
    if isinstance(raw_edges, list):
        for e in raw_edges:
            if not isinstance(e, dict):
                continue
            typ = str(e.get("type") or e.get("relation") or "")
            src = str(e.get("src") or e.get("source") or "")
            dst = str(e.get("dst") or e.get("target") or "")
            if typ and src and dst:
                edges.append((typ.lower(), src, dst))

    for key in ("contains", "sequence"):
        vals = data.get(key)
        if isinstance(vals, list):
            for e in vals:
                if isinstance(e, (list, tuple)) and len(e) >= 2:
                    edges.append((key, str(e[0]), str(e[1])))
                elif isinstance(e, dict):
                    src = str(e.get("src") or e.get("source") or "")
                    dst = str(e.get("dst") or e.get("target") or "")
                    if src and dst:
                        edges.append((key, src, dst))
    for key in ("graph", "ecg", "gold"):
        val = data.get(key)
        if isinstance(val, dict):
            edges.extend(graph_edges(val))
    return edges

def node_id(n: dict, idx: int = 0) -> str:
    return str(n.get("id") or n.get("node_id") or n.get("uid") or f"n{idx}")

def node_title(n: dict) -> str:
    return str(n.get("title") or n.get("name") or n.get("heading") or "").strip()

def node_kind(n: dict) -> str:
    return str(n.get("kind") or n.get("type") or n.get("node_type") or "").strip()

def node_path(n: dict) -> list[str]:
    p = n.get("outline_path") or n.get("path") or n.get("hierarchical_path") or []
    if isinstance(p, str):
        return [x.strip() for x in re.split(r"\s*(?:>|/|→)\s*", p) if x.strip()]
    if isinstance(p, list):
        return [str(x).strip() for x in p if str(x).strip()]
    return []

def node_pages(n: dict) -> list[int]:
    if 'evidence_pages' in n:
        return sorted(page_set(n['evidence_pages']))
    if n.get('page_start') is not None and n.get('page_end') is not None:
        a, b = n['page_start'], n['page_end']
        if type(a) is not int or type(b) is not int or a < 1 or b < a:
            raise ValueError('Invalid explicit page interval')
        return list(range(a, b + 1))
    pages = page_set(n.get("pages"))
    if not pages:
        pages = page_set(n.get("page"))
    if not pages:
        pages = page_set(n.get("page_span"))
    if not pages:
        pages = page_set(n.get("source_pages"))
    return sorted(pages)

def build_indexes(data: dict):
    nodes = graph_nodes(data)
    ids = {node_id(n, i): n for i,n in enumerate(nodes)}
    parent: dict[str,str] = {}
    next_by: dict[str,str] = {}
    for typ, src, dst in graph_edges(data):
        if typ == "contains":
            if dst in parent and parent[dst] != src:
                raise ValueError(f'Multiple parents: {dst}')
            parent[dst] = src
        elif typ == "sequence":
            if src in next_by and next_by[src] != dst:
                raise ValueError(f'Multiple next nodes: {src}')
            next_by[src] = dst
    return ids, parent, next_by

def graph_pages(data: dict) -> dict:
    """Use the project's declared page semantics, including grounding overrides."""
    from page_grounding import decode_pages
    result = {}
    for i, n in enumerate(graph_nodes(data)):
        nid = node_id(n, i)
        entry = (data.get('grounding') or {}).get(nid, {}) or {}
        if 'pages' in entry or 'pages' in n:
            result[nid] = decode_pages(entry.get('pages', n.get('pages', [])), data.get('page_semantics'))
        else:
            result[nid] = node_pages(n)
    return result

def title_for_id(ids: dict[str,dict], nid: str | None) -> str:
    return node_title(ids.get(nid, {})) if nid else ""

def full_path_from_parent(ids: dict[str,dict], parent: dict[str,str], nid: str) -> list[str]:
    # Prefer annotated path when available.
    p = node_path(ids.get(nid, {}))
    title = node_title(ids.get(nid, {}))
    if p:
        if title and norm(p[-1]) != norm(title):
            return p + [title]
        return p
    out = []
    cur = nid
    seen = set()
    while cur and cur not in seen:
        seen.add(cur)
        n = ids.get(cur)
        if not n:
            break
        t = node_title(n)
        if t:
            out.append(t)
        cur = parent.get(cur, "")
    return list(reversed(out))
