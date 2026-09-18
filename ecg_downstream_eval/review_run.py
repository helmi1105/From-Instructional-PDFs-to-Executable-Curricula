"""Produce a new offline report from saved calls without modifying the original run."""
import argparse
from pathlib import Path
from .common import load_json
from .run_pilot import report, save, sha

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    source, target = Path(args.run), Path(args.outdir)
    target.mkdir(parents=True, exist_ok=False)
    cases, requests = load_json(source/'cases.json'), load_json(source/'requests.json')
    files = [source/n for n in ('manifest.json','cases.json','requests.json','evaluation.json','REPORT.md')]
    files += sorted((source/'calls').glob('*/*.json'))
    sources = {str(p.resolve()):sha(p) for p in files}
    success = report(source, cases, requests, destination=target)
    save(target/'review_manifest.json', {'source_run':str(source.resolve()),'source_hashes':sources,
         'report_code_hashes':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
         'note':'Offline rescore. Original manifest, responses and reports unchanged. No paid calls.'})
    print(f'{success}/{len(requests)} successful saved responses. Report: {target.resolve()}')

if __name__ == '__main__':
    main()
