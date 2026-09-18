"""Offline analysis using the shared instance evaluator; no model calls."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
from ecg_evaluation_core import read_graph, evaluate_graphs, VERSION
from experiment_logging import write, sha


def complexity(data, pages=None, has_toc=None):
    graph=read_graph(data,True); nodes=graph['nodes']; children=Counter(s for s,t,k in graph['edges'] if k=='contains')
    counts=Counter(n['norm'] for n in nodes.values() if n.get('kind')!='COURSE')
    headings=sum(n.get('kind')!='COURSE' for n in nodes.values())
    return dict(pages=pages,heading_count=headings,node_count=len(nodes),root_included=True,
        max_depth=max((len(n['context']) for n in nodes.values()),default=0),depth_root=0,
        mean_internal_branching=statistics.mean(children.values()) if children else 0,
        max_branching=max(children.values(),default=0),leaf_count=sum(i not in children for i in nodes),
        repeated_title_groups=sum(v>1 for v in counts.values()),repeated_title_extra_instances=sum(v-1 for v in counts.values()),
        has_toc=has_toc,nodes_per_page=headings/pages if pages else None,
        issues=graph['issues'],contains_edges=sum(children.values()),expected_tree_edges=max(0,len(nodes)-1))


def repair_report(before,after,gold=None,pages=None,has_toc=None):
    result=evaluate_graphs(before,after,fuzzy_threshold=0.9)
    a,b=read_graph(before)['nodes'],read_graph(after)['nodes']
    operations=[dict(operation='remove',node=n) for n in result['unmatched_gold']]+[dict(operation='add',node=n) for n in result['unmatched_predicted']]
    mapping={m['pred_id']:m['gold_id'] for m in result['matches']}
    for pi,gi in mapping.items():
        for field,operation in [('norm','rename'),('context','reparent'),('kind','type_change'),('pages','page_change')]:
            if a[gi].get(field)!=b[pi].get(field):
                operations.append(dict(operation=operation,before_id=gi,after_id=pi,before=a[gi].get(field),after=b[pi].get(field)))
    seq=result['differences']
    if seq['missing_sequence'] or seq['extra_sequence']:
        operations.append(dict(operation='sequence_change',removed=seq['missing_sequence'],added=seq['extra_sequence']))
    out=dict(version=VERSION,operations=operations,no_change=not operations,
        matching_note='Operation attribution uses explicit fuzzy threshold 0.9; ambiguous renames/moves require inspection.',
        complexity=complexity(gold or before,pages,has_toc))
    if gold:
        x,y=evaluate_graphs(gold,before),evaluate_graphs(gold,after)
        out['before_metrics']=x['metrics']; out['after_metrics']=y['metrics']
        out['deltas']={k:y['metrics'][k]['f1']-v['f1'] for k,v in x['metrics'].items()}
        out['outcomes']={k:'beneficial' if v>0 else 'harmful' if v<0 else 'neutral' for k,v in out['deltas'].items()}
    return out


def stability(paths,gold):
    runs=[]; graphs=[]
    for p in paths:
        try:
            data=json.loads(Path(p).read_text(encoding='utf-8-sig'))
            r=evaluate_graphs(gold,data); graphs.append(data)
            runs.append(dict(path=str(p),status='success',metrics=r['metrics']))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            runs.append(dict(path=str(p),status='failed',error_type=type(exc).__name__))
    names=['node_instance','node_kind','path_aware_node','contains','sequence','grounding_conditional','grounding_end_to_end']
    def summary(values):return dict(mean=statistics.mean(values) if values else None,std=statistics.stdev(values) if len(values)>1 else 0.0,n=len(values))
    return dict(version=VERSION,runs=runs,success_rate=sum(r['status']=='success' for r in runs)/len(runs) if runs else 0,
        successful_run_scores={k:summary([r['metrics'][k]['f1'] for r in runs if r['status']=='success']) for k in names},
        all_run_scores_failure_as_zero={k:summary([r['metrics'][k]['f1'] if r['status']=='success' else 0 for r in runs]) for k in names},
        pairwise_instance_overlap=[{'a':i,'b':j,'metrics':evaluate_graphs(graphs[i],graphs[j])['metrics']} for i in range(len(graphs)) for j in range(i+1,len(graphs))])


def batch(pairs,out):
    from evaluate_ecg_against_gold import evaluate
    from ecg_evaluation_core import export_alignment
    out.mkdir(parents=True,exist_ok=True)
    rows=[]; records=[]
    for pair in pairs:
        name=pair.get('name',pair.get('id')); folder=out/name
        try:
            r=evaluate(Path(pair['gold']),Path(pair['pred']))
            write(folder/'evaluation_results.json',r); export_alignment(r['corrected'],folder)
            records.append(dict(pair,status='success',gold_sha256=sha(pair['gold']),pred_sha256=sha(pair['pred'])))
            for family,metrics in [('legacy',r['metrics']),('corrected',r['corrected']['metrics'])]:
                for metric,value in metrics.items():
                    if 'f1' in value:
                        rows.append(dict(document=name,family=family,metric=metric,**{k:value[k] for k in ['tp','fp','fn','precision','recall','f1']}))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            records.append(dict(pair,status='failed',error_type=type(exc).__name__,reason=str(exc)))
    write(out/'evaluation_manifest.json',{'version':VERSION,'runs':records})
    with (out/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['document','family','metric','tp','fp','fn','precision','recall','f1']);w.writeheader();w.writerows(rows)
    lines=['# Re-evaluation results','','| Document | Family | Metric | F1 |','|---|---|---|---:|']
    lines += [f"| {r['document']} | {r['family']} | {r['metric']} | {r['f1']:.6f} |" for r in rows]
    (out/'tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return records


def paired_statistics(rows,seed=2026,replicates=10000):
    """Input: document, metric, before, after. Average runs within documents first."""
    import numpy as np
    from scipy.stats import wilcoxon
    groups={}
    for row in rows: groups.setdefault(row['metric'],{}).setdefault(row['document'],[]).append(float(row['after'])-float(row['before']))
    rng=np.random.default_rng(seed); results=[]
    for metric,docs in sorted(groups.items()):
        values=np.array([statistics.mean(v) for _,v in sorted(docs.items())]); n=len(values)
        boots=np.mean(rng.choice(values,size=(replicates,n),replace=True),axis=1)
        p=float(wilcoxon(values).pvalue) if np.any(values) else 1.0
        from scipy.stats import rankdata
        nz=values[values!=0]; ranks=rankdata(abs(nz))
        effect=float(sum(ranks*np.sign(nz))/sum(ranks)) if len(nz) else 0.0
        results.append(dict(metric=metric,n_documents=n,mean_delta=float(values.mean()),median_delta=float(np.median(values)),
            bootstrap_95_ci=[float(x) for x in np.quantile(boots,[.025,.975])],rank_biserial=effect,p=p))
    running=0
    for rank,row in enumerate(sorted(results,key=lambda r:r['p'])):
        running=max(running,min(1,row['p']*(len(results)-rank)));row['p_holm']=running
    return dict(seed=seed,bootstrap_replicates=replicates,unit='document; repeated rows averaged within document',results=results)


def main():
    p=argparse.ArgumentParser();p.add_argument('--outdir',required=True)
    sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('batch');b.add_argument('--pairs',required=True)
    c=sub.add_parser('complexity');c.add_argument('--manifest',required=True)
    r=sub.add_parser('repair');r.add_argument('--before',required=True);r.add_argument('--after',required=True);r.add_argument('--gold')
    s=sub.add_parser('statistics');s.add_argument('--rows',required=True)
    a=p.parse_args();out=Path(a.outdir)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if a.command=='batch':batch(load(a.pairs),out)
    elif a.command=='repair':write(out/'repair.json',repair_report(load(a.before),load(a.after),load(a.gold) if a.gold else None))
    elif a.command=='statistics':write(out/'statistics.json',paired_statistics(load(a.rows)))
    else:
        root=Path(a.manifest).parent
        write(out/'complexity.json',{d['id']:complexity(load(root/d['gold']),d['pages'],d.get('has_toc')) for d in load(a.manifest)['documents']})


if __name__=='__main__':main()
