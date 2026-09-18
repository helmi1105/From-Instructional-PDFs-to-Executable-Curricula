"""Instrumented pipeline used by both public CLI entry points."""
import argparse
import json
import os
from pathlib import Path
from experiment_logging import RunLog, write, sha


def compile_branches(raw, repaired, outdir, mode):
    from build_clean_ecg_from_outline import build_clean_ecg
    from outline_only_pipeline import normalize_outline
    if mode not in ('off','on','both'): raise ValueError('Invalid repair mode')
    if mode in ('on','both') and repaired is None: raise ValueError('Missing repaired outline')
    branches=[]
    for name,outline,enabled in [('without_repair',raw,mode in ('off','both')),
                                  ('with_repair',repaired,mode in ('on','both'))]:
        if not enabled: continue
        folder=Path(outdir)/name
        if folder.exists() and any(folder.iterdir()): raise ValueError(f'Branch already exists: {folder}')
        write(folder/'outline_input.json',outline)
        clean=normalize_outline(outline)
        write(folder/'outline_clean.json',clean)
        build_clean_ecg(str(folder/'outline_clean.json'),str(folder))
        write(folder/'provenance.json',{'source_outline_sha256':sha(folder/'outline_input.json'),
                                      'normalized_outline_sha256':sha(folder/'outline_clean.json'),'branch':name})
        branches.append(folder)
    return branches


def main(direct=False):
    parser=argparse.ArgumentParser(description='Instrumented ECG pipeline; requires a fresh output directory')
    parser.add_argument('--pdf')
    parser.add_argument('--outline_json',help='Reuse one extracted outline')
    parser.add_argument('--repaired_outline_json',help='Reuse the repair of that extraction')
    parser.add_argument('--ocr_pages',help='Explicit OCR cache to reuse; numbered markdown pages JSON')
    parser.add_argument('--outdir',required=True)
    parser.add_argument('--repair',choices=['off','on','both'],default='both')
    parser.add_argument('--variant',choices=['original','revision'],default='original')
    parser.add_argument('--provider',choices=['openai','mistral'],default=os.getenv('LLM_PROVIDER','mistral'))
    parser.add_argument('--chat_model')
    parser.add_argument('--ocr_model',default='mistral-ocr-2505')
    budget=parser.add_mutually_exclusive_group()
    budget.add_argument('--max_output_tokens',type=int,default=16000,help='Per-call cap; each retry uses this cap')
    budget.add_argument('--provider_default_output',action='store_true',help='Omit the output-token limit from API requests; provider limits still apply')
    parser.add_argument('--max_retries',type=int,default=0)
    parser.add_argument('--experiment_id',default='manual')
    parser.add_argument('--document_id')
    parser.add_argument('--run_id',default='run01')
    parser.add_argument('--gold',help='Optional reference for provenance only')
    parser.add_argument('--pricing_json',help='Optional dated rates: input_per_million, output_per_million, ocr_per_page, currency')
    args=parser.parse_args()
    if args.provider_default_output: args.max_output_tokens=None
    if (args.max_output_tokens is not None and args.max_output_tokens<=0) or args.max_retries<0: parser.error('Invalid token/retry budget')
    if args.repaired_outline_json and not args.outline_json: parser.error('A saved repaired outline requires its source --outline_json')
    if not args.pdf and not args.ocr_pages and not args.outline_json: parser.error('Provide PDF, OCR pages or an outline')
    if args.repair!='off' and args.outline_json and not args.repaired_outline_json and not (args.pdf or args.ocr_pages):
        parser.error('Repair requires OCR context or an existing repaired outline')
    if not args.chat_model:
        args.chat_model=os.getenv('OPENAI_CHAT_MODEL','gpt-4.1-2025-04-14') if args.provider=='openai' else os.getenv('MISTRAL_CHAT_MODEL','mistral-large-latest')
    log=RunLog(args.outdir,vars(args),{k:getattr(args,k) for k in ['pdf','ocr_pages','outline_json','repaired_outline_json','gold','pricing_json']})
    try:
        run(args,log,direct)
    except Exception:
        log.finish('failed'); raise
    log.finish('success')


def run(args,log,direct=False):
    import outline_only_pipeline as outline
    from build_clean_ecg_from_outline import validate_ecg
    def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
    pages=None
    if args.ocr_pages:
        pages=log.stage('ocr_cache',lambda:load(args.ocr_pages))
    elif args.pdf:
        def ocr():
            client=outline.Mistral(api_key=os.environ['MISTRAL_API_KEY'])
            uploaded=outline.to_plain(outline.upload_pdf_for_ocr(client,args.pdf))
            url=outline.get_signed_url(client,uploaded['id'])
            response=client.ocr.process(model=args.ocr_model,document={'type':'document_url','document_url':url},include_image_base64=False)
            write(log.out/'ocr_raw.json',outline.to_plain(response))
            return outline.ocr_pages_to_dicts(response)
        pages=log.stage('ocr',ocr)
    if pages is not None: write(log.out/'ocr_pages.json',pages)
    log.manifest['ocr_cached']=bool(args.ocr_pages)
    log.manifest['ocr_page_count']=len(pages) if pages is not None else 0
    client=None
    def ask(prompt,stage):
        nonlocal client
        if client is None:
            if args.provider=='openai':
                if outline.OpenAI is None: raise RuntimeError('OpenAI SDK unavailable')
                client=outline.OpenAI(api_key=os.environ['OPENAI_API_KEY'],max_retries=0)
            else: client=outline.Mistral(api_key=os.environ['MISTRAL_API_KEY'])
        return log.stage(stage,lambda:log.ask(client,args.provider,args.chat_model,prompt,stage,args.max_output_tokens,args.max_retries))
    context=outline.pages_to_markdown_blob(pages) if pages is not None else ''
    if direct:
        if pages is None: raise ValueError('Direct generation requires OCR pages')
        from baseline_direct_ecg import DIRECT_ECG_RULES
        graph=ask(DIRECT_ECG_RULES+'\nOCR PAGES:\n'+context,'direct')
        write(log.out/'ecg_raw_direct.json',graph)
        if args.variant=='revision':
            graph=ask(DIRECT_ECG_RULES+'\nReview and minimally revise this draft against OCR evidence. Return the complete graph.\nDRAFT:\n'+json.dumps(graph,ensure_ascii=False)+'\nOCR PAGES:\n'+context,'direct_revision')
        validation=log.stage('validation',lambda:validate_ecg(graph))
        graph['validation']=validation
        write(log.out/'ecg.json',graph); write(log.out/'validation.json',validation)
        log.manifest['validation_status']='pass' if validation['ok'] else 'fail'
        log.manifest['budget_policy']={'calls':2 if args.variant=='revision' else 1,'per_call_output_cap':args.max_output_tokens,'retries_per_call':args.max_retries,'matching':'Comparable configured output opportunity, not equal actual token cost.'}
        return
    raw=load(args.outline_json) if args.outline_json else ask(outline.OUTLINE_RULES+'\nOCR PAGES:\n'+context,'extraction')
    write(log.out/'extraction'/'outline_raw.json',raw)
    repaired=None
    if args.repair in ('on','both'):
        repaired=load(args.repaired_outline_json) if args.repaired_outline_json else ask(outline.REPAIR_RULES+'\nCURRENT OUTLINE:\n'+json.dumps(raw,ensure_ascii=False)+'\nOCR PAGES:\n'+context,'repair')
        write(log.out/'extraction'/'outline_repaired.json',repaired)
    branches=log.stage('compilation',lambda:compile_branches(raw,repaired,log.out,args.repair))
    log.manifest['validation_status']={f.name:'pass' if load(f/'validation.json')['ok'] else 'fail' for f in branches}
    log.manifest['branch_parent_extraction_sha256']=sha(log.out/'extraction'/'outline_raw.json')
