"""Batch existing TOC/Docling adapters, compiler, and frozen-reference evaluation."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--systems',nargs='+',choices=['toc','docling'],default=['toc','docling'])
    p.add_argument('--documents',nargs='+',help='Optional IDs, e.g. D01 D07; default all 20')
    p.add_argument('--outdir',required=True)
    p.add_argument('--docling-ocr',action='store_true',help='Enable Docling local OCR for every selected document')
    p.add_argument('--dry-run',action='store_true',help='Check source files/settings without conversion or output writes')
    a=p.parse_args();snapshot=ROOT/'experiments/reviewer_report_20260913_final'
    frozen=load(snapshot/'report_manifest.json')
    docs=load(snapshot/'corpus_manifest_snapshot.json')['documents']
    if a.documents:
        unknown=set(a.documents)-{d['id'] for d in docs}
        if unknown:raise ValueError(f'Unknown document IDs: {unknown}')
        docs=[d for d in docs if d['id'] in a.documents]
    for name in ['evaluate_ecg_against_gold.py','ecg_evaluation_core.py','page_grounding.py','build_clean_ecg_from_outline.py']:
        if sha(ROOT/name)!=frozen['source_hashes'][name]:raise ValueError(f'{name} differs from the frozen comparison version')
    for d in docs:
        assert sha(ROOT/'data'/d['pdf'])==d['pdf_sha256'],d['id']+' PDF differs from frozen corpus'
        assert (snapshot/'gold_snapshot'/d['id']/'gold.json').is_file()
    versions={}
    for name in ['pymupdf','docling','docling-core','torch','transformers']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    if 'docling' in a.systems and not versions['docling']:raise RuntimeError('Docling is not installed in this Python environment')
    if a.dry_run:
        print(json.dumps(dict(documents=[d['id'] for d in docs],systems=a.systems,docling_ocr=a.docling_ocr,versions=versions,evaluator=frozen['evaluator_version']),indent=2))
        return
    out=Path(a.outdir).resolve()
    if out.exists():raise ValueError('Use a fresh output directory; existing runs are preserved')
    out.mkdir(parents=True)
    sources={}
    for path in list(ROOT.glob('*.py'))+list((ROOT/'toc').glob('*.py'))+list((ROOT/'docling').glob('*.py')):
        rel=path.relative_to(ROOT);target=out/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);sources[str(rel)]=sha(path)
    save(out/'manifest.json',dict(timestamp=datetime.now(timezone.utc).isoformat(),config=vars(a),versions=versions,
        evaluator=frozen['evaluator_version'],source_hashes=sources,reference_snapshot=str(snapshot),
        toc_policy='embedded bookmarks first; automatic printed TOC detection in first 20 pages; no manual page overrides',
        docling_adapter='existing defaults: exclude table-like areas, remove isolated unnumbered headings',
        notes='No paid LLM/OCR API calls. Docling model downloads may require network access. TOC page-label offsets are not explicitly corrected by the current adapter.'))
    statuses=[];metric_rows=[]
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    for d in docs:
        did=d['id'];pdf=ROOT/'data'/d['pdf'];gold=out/did/'gold_snapshot.json'
        gold.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(snapshot/'gold_snapshot'/did/'gold.json',gold)
        for system in a.systems:
            folder=out/did/system;folder.mkdir(parents=True);stages=[]
            def execute(name,arguments):
                cmd=[sys.executable,*map(str,arguments)];start=time.perf_counter()
                print(f'{did} {system}: {name}',flush=True)
                with (folder/(name+'.log')).open('w',encoding='utf-8') as f:
                    code=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
                stages.append(dict(stage=name,command=cmd,seconds=time.perf_counter()-start,exit_code=code))
                save(folder/'stages.json',stages)
                if code:raise RuntimeError(f'{name} failed with exit code {code}; see {folder/(name+".log")}')
            result=dict(document=did,system=system,status='running',pdf_sha256=sha(pdf),gold_sha256=sha(gold))
            try:
                outline=folder/'outline.json'
                if system=='toc':execute('extract',[ROOT/'toc/toc_outline.py','--pdf',pdf,'--out_json',outline,'--max_scan_pages','20'])
                else:
                    args=[ROOT/'docling/run_docling_one.py','--pdf',pdf,'--outdir',folder/'conversion']
                    if a.docling_ocr:args.append('--do_ocr')
                    execute('convert',args)
                    conversion=load(folder/'conversion/conversion_report.json')
                    result['conversion_status']=conversion['status']
                    result['processed_page_count']=len(conversion.get('processed_pages',[]))
                    if not conversion.get('complete'):
                        raise RuntimeError('Docling returned incomplete conversion; adaptation/evaluation stopped')
                    execute('adapt',[ROOT/'docling/adapter.py','--input',folder/'conversion'/(pdf.stem+'_docling.json'),'--output',outline,'--total-pages',d['pages']])
                result['empty_outline']=not load(outline).get('sections')
                execute('compile',[ROOT/'build_clean_ecg_from_outline.py','--outline_json',outline,'--outdir',folder/'graph'])
                pred=folder/'graph/ecg.json';result['prediction_sha256']=sha(pred)
                for mode,threshold in [('exact',None),('fuzzy_09',.9)]:
                    evaldir=folder/('evaluation_'+mode)
                    args=[ROOT/'evaluate_ecg_against_gold.py','--gold',gold,'--pred',pred,'--outdir',evaldir]
                    if threshold is not None:args+=['--fuzzy_threshold',threshold]
                    execute('evaluate_'+mode,args)
                    r=load(evaldir/'evaluation_results.json')
                    for family,metrics in [('legacy',r['metrics']),('corrected',r['corrected']['metrics'])]:
                        if family=='legacy' and mode!='exact':continue
                        for k,v in metrics.items():
                            if isinstance(v,dict) and 'f1' in v:
                                metric_rows.append(dict(document=did,system=system,mode=mode,family=family,metric=k,**{n:v.get(n) for n in ['precision','recall','f1']}))
                result['status']='success_empty_outline' if result['empty_outline'] else 'success'
                result['validation']=load(folder/'graph/validation.json')
            except Exception as exc:
                result.update(status='failed',error=str(exc));print(str(exc),flush=True)
            result['total_stage_seconds']=sum(s['seconds'] for s in stages)
            save(folder/'run_manifest.json',result);statuses.append(result)
            save(out/'status.json',statuses)
            if metric_rows:
                with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(metric_rows[0]));w.writeheader();w.writerows(metric_rows)
    print(f'Results: {out}; failed: {sum(r["status"]=="failed" for r in statuses)}/{len(statuses)}')
    if any(r['status']=='failed' for r in statuses):sys.exit(1)

if __name__=='__main__':main()
