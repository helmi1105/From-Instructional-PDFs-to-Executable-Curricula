import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ecg_evaluation_core import evaluate_graphs
from revision_pipeline import compile_branches
from experiment_logging import RunLog
from experiment_analysis import paired_statistics, stability
from unittest.mock import patch


def graph():
    return {'nodes':[{'id':i,'title':title,'kind':kind,'pages':[1]} for i,title,kind in
        [('r','Course','COURSE'),('a','A','MODULE'),('b','B','MODULE'),('x','Conclusion','KC'),('y','Conclusion','KC')]],
        'contains':[['r','a'],['r','b'],['a','x'],['b','y']], 'sequence':[]}


class RevisionTests(unittest.TestCase):
    def test_duplicate_missing(self):
        g=graph();p=copy.deepcopy(g);p['nodes']=p['nodes'][:-1];p['contains']=p['contains'][:-1]
        r=evaluate_graphs(g,p)
        self.assertEqual(r['metrics']['node_instance']['fn'],1)
        self.assertEqual(len(r['matches']),3)
        self.assertEqual(len({m['gold_id'] for m in r['matches']}),3)

    def test_wrong_parent_keeps_node_credit(self):
        g=graph();g['nodes'][-1]['title']='Unique';p=copy.deepcopy(g);p['contains'][-1]=['a','y']
        r=evaluate_graphs(g,p)['metrics']
        self.assertEqual(r['node_instance']['f1'],1)
        self.assertLess(r['contains']['f1'],1)
        self.assertLess(r['path_aware_node']['f1'],1)

    def test_kind_and_pages_do_not_change_matching(self):
        g=graph();p=copy.deepcopy(g);p['nodes'][-1].update(kind='UNIT',pages=[3])
        r=evaluate_graphs(g,p)['metrics']
        self.assertEqual(r['node_instance']['f1'],1)
        self.assertLess(r['node_kind']['f1'],1)
        self.assertEqual(r['grounding_conditional']['fp'],1)
        self.assertEqual(r['grounding_conditional']['fn'],1)

    def test_end_to_end_missing_and_extra(self):
        g={'nodes':[{'id':str(i),'title':str(i),'kind':'KC','pages':[1]} for i in range(10)]}
        p={'nodes':g['nodes'][:2]}
        r=evaluate_graphs(g,p)['metrics']
        self.assertEqual(r['grounding_conditional']['f1'],1)
        self.assertAlmostEqual(r['grounding_end_to_end']['f1'],1/3)
        self.assertEqual(evaluate_graphs(p,g)['metrics']['grounding_end_to_end']['fp'],8)

    def test_fuzzy_is_opt_in(self):
        g={'nodes':[{'id':'g','title':'Introduction','kind':'KC'}]}
        p={'nodes':[{'id':'p','title':'Introductiom','kind':'KC'}]}
        self.assertEqual(evaluate_graphs(g,p)['metrics']['node_instance']['tp'],0)
        self.assertEqual(evaluate_graphs(g,p,fuzzy_threshold=.9)['metrics']['node_instance']['tp'],1)

    def test_repair_branches_and_determinism(self):
        raw={'title':'Course','sections':[{'title':'A','start_page':1,'end_page':1,'sections':[]}]}
        fixed=copy.deepcopy(raw);fixed['sections'].append({'title':'B','start_page':2,'end_page':2,'sections':[]})
        with tempfile.TemporaryDirectory() as t:
            folders=compile_branches(raw,fixed,t,'both')
            a,b=[json.loads((f/'ecg.json').read_text(encoding='utf-8')) for f in folders]
            self.assertEqual(len(a['nodes']),2);self.assertEqual(len(b['nodes']),3)
            other=compile_branches(raw,None,Path(t)/'repeat','off')
            self.assertEqual(a,json.loads((other[0]/'ecg.json').read_text(encoding='utf-8')))

    def test_retry_persists_invalid_and_truncated_responses(self):
        class Response:
            def __init__(self,text,status):self.output_text=text;self.status=status
            def model_dump(self):return {'model':'fake','status':self.status,'usage':{'input_tokens':10,'output_tokens':5},'incomplete_details':{'reason':'max_output_tokens'} if self.status=='incomplete' else None}
        responses=iter([Response('{','incomplete'),Response('{}','completed')])
        client=SimpleNamespace(responses=SimpleNamespace(create=lambda **kw:next(responses)))
        with tempfile.TemporaryDirectory() as t:
            log=RunLog(Path(t)/'run',{},{});self.assertEqual(log.ask(client,'openai','fake','prompt','extract',100,1),{})
            first=json.loads((log.out/'calls/extract/attempt_1.json').read_text())
            self.assertEqual(first['raw_text'],'{');self.assertEqual(first['parse_status'],'invalid_json')
            self.assertEqual(first['completion_status'],'truncated')
            self.assertEqual(json.loads((log.out/'calls/extract/attempt_2.json').read_text())['run_status'],'success')

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            RunLog(Path(t)/'run',{}, {})
            with self.assertRaises(ValueError):RunLog(Path(t)/'run',{}, {})

    def test_api_failure_is_saved(self):
        def fail(**kwargs): raise ConnectionError('test transport error')
        client=SimpleNamespace(responses=SimpleNamespace(create=fail))
        with tempfile.TemporaryDirectory() as t:
            log=RunLog(Path(t)/'run',{}, {})
            with self.assertRaises(ConnectionError):log.ask(client,'openai','fake','prompt','direct',100)
            record=json.loads((log.out/'calls/direct/attempt_1.json').read_text())
            self.assertEqual(record['api_status'],'failed')
            self.assertEqual(record['parse_status'],'not_attempted')

    def test_pipeline_extracts_once_for_both_branches(self):
        from revision_pipeline import run
        raw={'title':'Course','sections':[{'title':'A','start_page':1,'end_page':1,'sections':[]}]}
        repaired=copy.deepcopy(raw);repaired['sections'][0]['title']='B'
        with tempfile.TemporaryDirectory() as t:
            cache=Path(t)/'ocr.json';cache.write_text('[{"page_number":1,"markdown":"A B"}]')
            log=RunLog(Path(t)/'run',{}, {})
            args=SimpleNamespace(ocr_pages=str(cache),pdf=None,provider='openai',chat_model='fake',
                max_output_tokens=100,max_retries=0,outline_json=None,repaired_outline_json=None,repair='both')
            calls=[]
            def ask(*a):
                calls.append(a[4]);return raw if a[4]=='extraction' else repaired
            with patch.object(log,'ask',side_effect=ask),patch('outline_only_pipeline.OpenAI'),patch.dict('os.environ',{'OPENAI_API_KEY':'fake'}):
                run(args,log)
            self.assertEqual(calls,['extraction','repair'])
            self.assertTrue((log.out/'without_repair/ecg.json').exists())
            self.assertTrue((log.out/'with_repair/ecg.json').exists())

    def test_failed_run_is_not_silently_excluded(self):
        with tempfile.TemporaryDirectory() as t:
            r=stability([Path(t)/'missing.json'],graph())
            self.assertEqual(r['success_rate'],0)
            self.assertEqual(r['runs'][0]['status'],'failed')
            self.assertEqual(r['all_run_scores_failure_as_zero']['node_instance']['mean'],0)

    def test_direct_revision_uses_validator_without_compiler(self):
        from revision_pipeline import run
        draft={'nodes':[{'id':'r','title':'Course','kind':'COURSE','path':['Course'],'page_start':1,'page_end':1}], 'edges':[]}
        with tempfile.TemporaryDirectory() as t:
            cache=Path(t)/'ocr.json';cache.write_text('[{"page_number":1,"markdown":"Course"}]')
            log=RunLog(Path(t)/'run',{}, {})
            args=SimpleNamespace(ocr_pages=str(cache),pdf=None,provider='openai',chat_model='fake',max_output_tokens=100,max_retries=0,variant='revision')
            calls=[]
            def ask(*a):calls.append(a[4]);return copy.deepcopy(draft)
            with patch.object(log,'ask',side_effect=ask),patch('outline_only_pipeline.OpenAI'),patch.dict('os.environ',{'OPENAI_API_KEY':'fake'}),patch('revision_pipeline.compile_branches',side_effect=AssertionError('Compiler must not be used')):
                run(args,log,direct=True)
            self.assertEqual(calls,['direct','direct_revision'])
            self.assertTrue(json.loads((log.out/'ecg.json').read_text())['validation']['ok'])

    def test_document_level_statistics(self):
        r=paired_statistics([dict(document='a',metric='node',before=0,after=1),dict(document='a',metric='node',before=0,after=1),dict(document='b',metric='node',before=0,after=0)],replicates=100)
        self.assertEqual(r['results'][0]['n_documents'],2)
        self.assertEqual(r['results'][0]['mean_delta'],.5)


if __name__=='__main__':unittest.main()
