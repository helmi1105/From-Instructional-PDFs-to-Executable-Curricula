"""Instance-preserving ECG evaluation. Legacy evaluators remain separate."""
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import csv
import json
import re
from pathlib import Path
from page_grounding import decode_pages, gold_pages_by_id

VERSION = '2.1-embedded-heading-number'

# Explicit structural labels only: a number occurring in ordinary prose is
# not evidence that the title already carries its section number.
STRUCTURAL_NUMBER_RE = re.compile(
    r'^\s*(?:question|chapter|chapitre|section|part|partie|module|unit|unite|unité|'
    r'champ|annex|annexe|appendix|lesson|lecon|leçon|exercise|exercice)\s+'
    r'(?P<number>\d+(?:\.\d+)*|[ivxlcdm]+)\.?(?=\s|[:)\-–—]|$)', re.IGNORECASE)


def normalize_title(title, number=None):
    # Legacy canonicalization stays untouched. Only suppress redundant metadata
    # numbering when an anchored structural label contains the same number.
    from evaluate_ecg_against_gold import canonical_heading_full
    match = STRUCTURAL_NUMBER_RE.match(str(title or ''))
    if match and str(number or '').strip().rstrip('.').casefold() == match['number'].casefold():
        number = None
    return canonical_heading_full(title, number)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r,
                f1=2*p*r/(p+r) if p+r else 0.0)


def read_graph(data, include_course=False):
    nodes = {}
    issues = []
    pages = gold_pages_by_id(data)
    for n in data.get('nodes', []):
        nid = n.get('id')
        if not isinstance(nid, str) or not nid or nid in nodes:
            raise ValueError(f'Missing or duplicate node ID: {nid!r}')
        evidence = pages[nid]
        if 'evidence_pages' in n and n['evidence_pages']:
            evidence = decode_pages(n['evidence_pages'])
        elif isinstance(n.get('page_start'), int):
            evidence = decode_pages([n['page_start'], n.get('page_end') or n['page_start']],
                                    {'format': '[start_page, end_page]', 'inclusive': True})
        nodes[nid] = dict(n, norm=normalize_title(n.get('title', ''), n.get('number')), pages=evidence)
    edges = []
    if 'edges' in data:
        edges = [(e.get('source'), e.get('target'), e.get('type')) for e in data['edges']]
    else:
        for kind in ('contains', 'sequence'):
            for edge in data.get(kind, []):
                edges.append((edge[0], edge[2] if len(edge) == 3 else edge[1], kind))
    parents = defaultdict(list)
    for s, t, kind in edges:
        if s not in nodes or t not in nodes:
            issues.append({'type': 'dangling_edge', 'edge': [s,t,kind]})
        if kind == 'contains':
            parents[t].append(s)
    def context(nid, visited=()):
        if nid in visited:
            return ('<cycle>',)
        ps = sorted(set(parents[nid]))
        if len(ps) != 1:
            return ('<multiple_parents>',) if ps else ()
        p = ps[0]
        if p not in nodes:
            return ('<missing_parent>',)
        if nodes[p].get('kind') == 'COURSE' and not include_course:
            return ()
        return context(p, visited+(nid,)) + (nodes[p]['norm'],)
    for nid, n in nodes.items():
        n['context'] = context(nid)
        n['parent_ids'] = sorted(set(parents[nid]))
    excluded = {nid for nid,n in nodes.items() if n.get('kind') == 'COURSE' and not include_course}
    return {'nodes': {nid:n for nid,n in nodes.items() if nid not in excluded},
            'edges': [e for e in edges if e[0] not in excluded and e[1] not in excluded], 'issues': issues}


def align(gold, pred, fuzzy_threshold=None):
    if fuzzy_threshold is not None and not 0 < fuzzy_threshold <= 1:
        raise ValueError('fuzzy_threshold must be in (0, 1]')
    g, p = gold['nodes'], pred['nodes']
    mapping, records, ambiguous = {}, [], []
    used = set()
    def candidates(exact):
        result = []
        for pi in sorted(p):
            if pi in mapping: continue
            for gi in sorted(g):
                if gi in used: continue
                same = p[pi]['norm'] == g[gi]['norm']
                if not p[pi]['norm'] or not g[gi]['norm']: continue
                if exact and not same: continue
                similarity = 1.0 if same else SequenceMatcher(None, p[pi]['norm'], g[gi]['norm']).ratio()
                if not exact and (same or similarity < fuzzy_threshold): continue
                pc, gc = p[pi]['context'], g[gi]['context']
                context_score = (pc == gc, bool(pc and gc and pc[-1] == gc[-1]),
                                 sum(a == b for a,b in zip(reversed(pc), reversed(gc))))
                result.append((similarity, context_score, pi, gi))
        return sorted(result, key=lambda x: (-x[0], tuple(-int(v) for v in x[1]), x[2], x[3]))
    for exact in ([True, False] if fuzzy_threshold is not None else [True]):
        choices = candidates(exact)
        counts = Counter(pi for _,_,pi,_ in choices)
        for similarity, context_score, pi, gi in choices:
            if counts[pi] > 1:
                ambiguous.append(dict(pred_id=pi, gold_id=gi, similarity=similarity, context_score=context_score))
            if pi in mapping or gi in used: continue
            mapping[pi] = gi
            used.add(gi)
            records.append(dict(pred_id=pi, gold_id=gi, pred_title=p[pi]['title'], gold_title=g[gi]['title'],
                                method='exact' if exact else 'fuzzy', similarity=similarity))
    return mapping, records, ambiguous


def evaluate_graphs(gold_data, pred_data, include_course=False, fuzzy_threshold=None):
    gold, pred = read_graph(gold_data, include_course), read_graph(pred_data, include_course)
    g,p = gold['nodes'], pred['nodes']
    mapping, matches, ambiguous = align(gold, pred, fuzzy_threshold)
    def count_metric(tp): return prf(tp,len(p)-tp,len(g)-tp)
    metrics = {'node_instance': count_metric(len(mapping)),
               'node_kind': count_metric(sum(p[pi].get('kind') == g[gi].get('kind') for pi,gi in mapping.items()))}
    # Exact path-aware multiset comparison is independent of fuzzy sensitivity.
    gp = Counter(n['context']+(n['norm'],) for n in g.values())
    pp = Counter(n['context']+(n['norm'],) for n in p.values())
    metrics['path_aware_node'] = count_metric(sum((gp & pp).values()))
    differences = {}
    for kind in ('contains','sequence'):
        ge = Counter((s,t) for s,t,k in gold['edges'] if k==kind)
        pe = Counter((mapping.get(s, ('unmatched_pred',s)),mapping.get(t, ('unmatched_pred',t)))
                     for s,t,k in pred['edges'] if k==kind)
        tp=sum((ge & pe).values())
        metrics[kind]=prf(tp,sum(pe.values())-tp,sum(ge.values())-tp)
        differences['missing_'+kind]=[list(e) for e in (ge-pe).elements()]
        differences['extra_'+kind]=[list(e) for e in (pe-ge).elements()]
    tp=fp=fn=0
    for pi,gi in mapping.items():
        a,b=set(g[gi]['pages']),set(p[pi]['pages'])
        tp+=len(a&b); fp+=len(b-a); fn+=len(a-b)
    metrics['grounding_conditional']=prf(tp,fp,fn)
    metrics['grounding_end_to_end']=prf(tp, fp+sum(len(n['pages']) for i,n in p.items() if i not in mapping),
        fn+sum(len(n['pages']) for i,n in g.items() if i not in mapping.values()))
    return {'version':VERSION, 'policy':{'include_course':include_course,'fuzzy_threshold':fuzzy_threshold,
        'grounding':'All included node kinds; explicit reference page semantics; absent pages contribute no pairs.',
        'alignment':'Exact title first; context then lexical IDs break ties; fuzzy greedy pass is opt-in.'},
        'metrics':metrics,'matches':matches,'ambiguous_matches':ambiguous,
        'unmatched_gold':[g[i] for i in sorted(g) if i not in mapping.values()],
        'unmatched_predicted':[p[i] for i in sorted(p) if i not in mapping],
        'differences':differences,'gold_issues':gold['issues'],'pred_issues':pred['issues']}


def export_alignment(result, outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    for name,key in [('matched_nodes','matches'),('ambiguous_matches','ambiguous_matches'),
                     ('unmatched_gold_nodes','unmatched_gold'),('unmatched_predicted_nodes','unmatched_predicted')]:
        rows=result[key]
        fields=sorted({k for row in rows for k in row}) or ['id']
        with (out/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
            writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,tuple,dict)) else v for k,v in row.items()} for row in rows)
