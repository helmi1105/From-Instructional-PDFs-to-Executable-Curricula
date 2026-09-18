import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from .common import graph_pages, norm
from .run_pilot import build_requests, graph_context, call_one, report, path_without_root, check_reuse

class PilotTests(unittest.TestCase):
    def test_reuse_rejects_changed_payload_or_model(self):
        import copy
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); folder=p/'calls'/'x';folder.mkdir(parents=True)
            prediction={'section_title':'a','parent_title':'p','next_title':'','path':['p','a'],'pages':[1]}
            config={k:0 for k in ('model','temperature','max_output_tokens','sdk_retries','timeout_seconds','openai_sdk_version','prompt_sha256','retrieval')}
            config['inputs']={k:{'sha256':'test'} for k in ('ocr','predicted')}
            request={'request_id':'x','input':'unchanged'}
            for name,data in [('manifest.json',{'config':config}),('requests.json',[request]),('cases.json',[])]:
                (p/name).write_text(json.dumps(data))
            for name,data in [('result.json',{'request_id':'x','status':'success','prediction':prediction}),('raw_response.json',{'status':'completed'}),('output_text.json',{'text':json.dumps(prediction)})]:
                (folder/name).write_text(json.dumps(data))
            self.assertEqual(len(check_reuse(p,[request],config)[0]),1)
            with self.assertRaises(ValueError):check_reuse(p,[dict(request,input='changed')],config)
            changed=copy.deepcopy(config);changed['model']='other'
            with self.assertRaises(ValueError):check_reuse(p,[request],changed)

    def test_traceability_requires_path_and_exact_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); cases=[{'case_id':'x','gold':{'section_title':'a','parent_title':'p','next_title':'','path':['root','a'],'pages':[1,2]}}]
            requests=[]
            for system,predpath,pages in [('full_context',['alias','a'],[1,2]),('flat_rag',['root','a'],[1]),('ecg_guided_rag',['root','a'],[1,2])]:
                requests.append({'request_id':system,'case_id':'x','system':system})
                folder=out/'calls'/system;folder.mkdir(parents=True)
                pred=dict(cases[0]['gold'],path=predpath,pages=pages)
                (folder/'result.json').write_text(json.dumps({'status':'success','prediction':pred}))
            report(out,cases,requests)
            systems=json.loads((out/'evaluation.json').read_text())['systems']
            self.assertEqual(systems['full_context']['metrics']['traceability_strict'],0)
            self.assertEqual(systems['full_context']['metrics']['traceability_without_root'],1)
            self.assertEqual(systems['flat_rag']['metrics']['traceability_strict'],0)
            self.assertEqual(systems['ecg_guided_rag']['metrics']['traceability_strict'],1)
    def test_root_diagnostic_keeps_internal_structure(self):
        gold={'path':['Document','Parent','Heading']}
        self.assertEqual(path_without_root({'path':['Alias','Parent','Heading']},gold),1)
        self.assertEqual(path_without_root({'path':['Alias','Extra','Parent','Heading']},gold),0)
        self.assertEqual(path_without_root({'path':['Parent','Heading']},gold),0)
        self.assertEqual(path_without_root({'path':[]},gold),0)
    def test_declared_page_intervals_and_prediction(self):
        g = {'nodes':[{'id':'a','pages':[10,15]}], 'page_semantics':{'format':'[start_page,end_page]','inclusive':True}}
        self.assertEqual(graph_pages(g)['a'], list(range(10,16)))
        del g['page_semantics']
        self.assertEqual(graph_pages(g)['a'], [10,15])
        self.assertEqual(graph_pages({'nodes':[{'id':'a','evidence_pages':[3,4]}]})['a'], [3,4])
        self.assertEqual(norm('Introduction'), 'introduction')

    def test_missing_duplicate_and_cross_parent_graph(self):
        g = {'nodes':[{'id':'a','title':'Conclusion'},{'id':'b','title':'Conclusion'}]}
        self.assertEqual(graph_context(g,'Conclusion')['match_status'],'ambiguous')
        self.assertEqual(graph_context(g,'Unknown')['match_status'],'missing')
        g = {'nodes':[{'id':'a','title':'First'},{'id':'b','title':'Second'}],
             'contains':[['p','a'],['q','b']], 'sequence':[['a','b']]}
        self.assertEqual(graph_context(g,'First')['next_title'],'')

    def test_equal_retrieval_no_gold_payload(self):
        cases = [{'case_id':'x','query':'Locate Heading','anchor_title':'Heading','gold':{'secret':'GOLD_ONLY'}}]
        ocr = [{'page_number':1,'markdown':'Heading content'}]
        req = build_requests(cases,ocr,{'nodes':[]})
        self.assertEqual(req[1]['chunk_ids'],req[2]['chunk_ids'])
        self.assertNotIn('GOLD_ONLY',json.dumps(req))

    def test_invalid_and_truncated_raw_preserved(self):
        for status, text in [('completed','not json'),('incomplete','{}')]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                response = SimpleNamespace(output_text=text,model_dump=lambda **kw: {
                    'status':status,'usage':{'input_tokens':10,'output_tokens':2},
                    'incomplete_details':{'reason':'max_output_tokens'} if status=='incomplete' else None})
                client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw:response))
                folder=Path(tmp)/'call'
                result=call_one(client,{'request_id':'x','instructions':'x','input':'x'},folder,
                                {'model':'mock','max_output_tokens':20})
                self.assertTrue((folder/'raw_response.json').exists())
                self.assertEqual(result['status'],'truncated' if status=='incomplete' else 'invalid_output')
                self.assertEqual(result['usage']['input_tokens'],10)

    def test_failure_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            gold={'section_title':'a','parent_title':'p','next_title':'b','path':['p','a'],'pages':[1]}
            cases=[{'case_id':'x','gold':gold},{'case_id':'y','gold':gold}]
            requests=[]
            for system in ('full_context','flat_rag','ecg_guided_rag'):
                for cid in ('x','y'):
                    rid=cid+'_'+system
                    requests.append({'case_id':cid,'system':system,'request_id':rid})
                    folder=out/'calls'/rid;folder.mkdir(parents=True)
                    value={'status':'success','prediction':gold} if cid=='x' else {'status':'api_failure'}
                    (folder/'result.json').write_text(json.dumps(value))
            self.assertEqual(report(out,cases,requests),3)
            data=json.loads((out/'evaluation.json').read_text())
            self.assertEqual(data['systems']['full_context']['metrics']['page_f1'],.5)

if __name__ == '__main__':
    unittest.main()
