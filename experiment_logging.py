"""Local run provenance and per-attempt JSON logging; no credentials serialized."""
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_json_response(text):
    """Accept a complete JSON object, optionally in one leading Markdown fence.

    Allow trailing prose after that fence, but reject additional structured
    payloads. Never repair malformed JSON or search through leading prose.
    """
    if not isinstance(text, str):
        raise ValueError('Expected response text')
    candidate = text.strip()
    match = re.fullmatch(r'```(?:json)?[ \t]*\r?\n(.*?)\r?\n```[ \t]*(?:\r?\n(.*))?', candidate, flags=re.DOTALL | re.IGNORECASE)
    wrapper = 'markdown_fence_removed' if match else 'none'
    if match:
        trailing = (match.group(2) or '').strip()
        if '```' in match.group(1) or '```' in trailing or any(c in trailing for c in '{}[]'):
            raise ValueError('Ambiguous fenced JSON response')
        if trailing:
            wrapper = 'markdown_fence_and_trailing_commentary_removed'
    value = json.loads(match.group(1) if match else candidate)
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object')
    return value, wrapper


def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')


class RunLog:
    def __init__(self, outdir, config, inputs):
        self.out=Path(outdir)
        if self.out.exists() and any(self.out.iterdir()):
            raise ValueError(f'Use a fresh output directory: {self.out}')
        self.out.mkdir(parents=True,exist_ok=True)
        root=Path(__file__).parent
        def git(*args):
            try: return subprocess.check_output(['git',*args],cwd=root,stderr=subprocess.DEVNULL).decode().strip()
            except (OSError, subprocess.CalledProcessError): return None
        self.manifest=dict(version='1.0', evaluator_version='2.0-instance-alignment',
            timestamp=datetime.now(timezone.utc).isoformat(), config=config,
            inputs={k:{'path':str(v),'sha256':sha(v)} for k,v in inputs.items() if v},
            git_commit=git('rev-parse','HEAD'), dirty_worktree=bool(git('status','--porcelain')),
            source_hashes={p.name:sha(p) for p in root.glob('*.py')}, stages=[],run_status='running')
        self.start=time.perf_counter(); self.save()

    def save(self): write(self.out/'manifest.json', self.manifest)

    def stage(self, name, fn):
        start=time.perf_counter()
        started_at=datetime.now(timezone.utc).isoformat()
        status='failed'
        try:
            value=fn(); status='success'; return value
        except Exception:
            status='failed'; raise
        finally:
            self.manifest['stages'].append(dict(stage=name,started_at=started_at,ended_at=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-start,status=status)); self.save()

    def finish(self, status):
        attempts=[json.loads(p.read_text(encoding='utf-8')) for p in self.out.glob('calls/*/attempt_*.json')]
        self.manifest['usage_summary']={
            'total_attempts':len(attempts),
            'failed_attempts':sum(a['run_status']!='success' for a in attempts),
            'retry_success':any(a['attempt']>1 and a['run_status']=='success' for a in attempts),
            'known_input_tokens':sum(a.get('input_tokens') or 0 for a in attempts),
            'known_output_tokens':sum(a.get('output_tokens') or 0 for a in attempts),
            'attempts_missing_usage':sum(a.get('input_tokens') is None or a.get('output_tokens') is None for a in attempts),
            'sdk_retry_policy':'OpenAI automatic retries disabled; Mistral transport retries are SDK-dependent and not separately measured.'}
        pricing_path=self.manifest['config'].get('pricing_json')
        if pricing_path:
            rates=json.loads(Path(pricing_path).read_text(encoding='utf-8'))
            usage=self.manifest['usage_summary']
            llm=(usage['known_input_tokens']*rates['input_per_million']+usage['known_output_tokens']*rates['output_per_million'])/1_000_000
            ocr=0 if self.manifest.get('ocr_cached') else self.manifest.get('ocr_page_count',0)*rates['ocr_per_page']
            self.manifest['estimated_cost']={'rates':rates,'llm':llm,'ocr':ocr,'total':llm+ocr,
                'complete_usage':usage['attempts_missing_usage']==0,
                'note':'Estimate using supplied rates; caching discounts and provider-side retries are not modeled.'}
        self.manifest.update(run_status=status,total_seconds=time.perf_counter()-self.start,
            artifacts={str(p.relative_to(self.out)):sha(p) for p in self.out.rglob('*') if p.is_file() and p.name!='manifest.json'})
        self.save()

    def ask(self, client, provider, model, prompt, stage, max_output_tokens, retries=0):
        prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()
        (self.out/'calls'/stage).mkdir(parents=True,exist_ok=True)
        (self.out/'calls'/stage/'prompt.txt').write_text(prompt,encoding='utf-8')
        for attempt in range(1,retries+2):
            record=dict(stage=stage,attempt=attempt,requested_model=model,prompt_sha256=prompt_hash,
                        max_output_tokens=max_output_tokens,temperature=0,completion_status='unknown',
                        parse_status='not_attempted',validation_status='not_attempted',run_status='failed',
                        timestamp=datetime.now(timezone.utc).isoformat())
            target=self.out/'calls'/stage/f'attempt_{attempt}.json'; start=time.perf_counter()
            try:
                if provider=='openai':
                    limits={} if max_output_tokens is None else {'max_output_tokens':max_output_tokens}
                    resp=client.responses.create(model=model,input=[{'role':'user','content':prompt}],temperature=0,**limits)
                    text=getattr(resp,'output_text','')
                else:
                    limits={} if max_output_tokens is None else {'max_tokens':max_output_tokens}
                    resp=client.chat.complete(model=model,messages=[{'role':'user','content':prompt}],temperature=0,**limits)
                    text=resp.choices[0].message.content
                raw=resp.model_dump() if hasattr(resp,'model_dump') else vars(resp)
                usage=raw.get('usage') or {}
                record.update(returned_model=raw.get('model'),response_id=raw.get('id'),
                    input_tokens=usage.get('input_tokens',usage.get('prompt_tokens')),
                    output_tokens=usage.get('output_tokens',usage.get('completion_tokens')),
                        usage=usage,api_status='completed',raw_text=text,raw_response=raw)
                status=raw.get('status'); reason=(raw.get('choices') or [{}])[0].get('finish_reason')
                incomplete=raw.get('incomplete_details') or {}
                record.update(provider_status=status,finish_reason=reason,incomplete_details=incomplete)
                if reason=='length' or incomplete.get('reason')=='max_output_tokens': record['completion_status']='truncated'
                elif status=='completed' or reason=='stop': record['completion_status']='complete'
                write(target,record)  # Persist response BEFORE parsing, including invalid/truncated text.
                try:
                    value, wrapper = parse_json_response(text)
                    record['response_normalization'] = wrapper
                except (ValueError,TypeError):
                    record['parse_status']='invalid_json'; raise
                if not isinstance(value,dict):
                    record['parse_status']='invalid_schema'; raise ValueError('Expected JSON object')
                record['parse_status']='valid'
                if record['completion_status']=='truncated': raise ValueError('Truncated response')
                if status in ('failed','incomplete','cancelled'): raise ValueError('Incomplete provider response')
                record['run_status']='success'
                return value
            except Exception as exc:
                record.setdefault('api_status','failed')
                record['error_type']=type(exc).__name__  # Avoid exception messages containing request credentials.
                if attempt==retries+1: raise
            finally:
                record['seconds']=time.perf_counter()-start; write(target,record)
