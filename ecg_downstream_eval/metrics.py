from __future__ import annotations
from .common import *

def exact_title(pred: str, gold: str) -> float:
    return float(norm(pred) == norm(gold))

def path_accuracy(pred, gold) -> float:
    p = [norm(x) for x in (pred or []) if norm(x)]
    g = [norm(x) for x in (gold or []) if norm(x)]
    return float(p == g)

def score_case(pred: dict, gold: dict) -> dict:
    pp, pr, pf = prf(pred.get("pages") or [], gold.get("pages") or [])
    return {
        "section_accuracy": exact_title(pred.get("section_title",""), gold.get("section_title","")),
        "parent_accuracy": exact_title(pred.get("parent_title",""), gold.get("parent_title","")),
        "next_accuracy": exact_title(pred.get("next_title",""), gold.get("next_title","")),
        "path_accuracy": path_accuracy(pred.get("path") or [], gold.get("path") or []),
        "page_precision": pp,
        "page_recall": pr,
        "page_f1": pf,
        "traceability_exact": float(
            exact_title(pred.get("section_title",""), gold.get("section_title","")) == 1.0
            and path_accuracy(pred.get("path") or [], gold.get("path") or []) == 1.0
            and pf == 1.0
        ),
    }

def summarize(rows: list[dict]) -> dict:
    keys = [
        "section_accuracy","parent_accuracy","next_accuracy","path_accuracy",
        "page_precision","page_recall","page_f1","traceability_exact"
    ]
    out = {}
    for k in keys:
        out[k] = safe_mean([float(r["scores"][k]) for r in rows])
    for k in ("latency_s","input_tokens","output_tokens","estimated_context_tokens","graph_context_tokens_est"):
        vals = [r["meta"].get(k) for r in rows if r.get("meta",{}).get(k) is not None]
        if vals:
            out[f"mean_{k}"] = safe_mean([float(x) for x in vals])
    out["n_cases"] = len(rows)
    return out
