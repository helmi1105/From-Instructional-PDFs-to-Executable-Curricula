import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from page_grounding import decode_pages, gold_pages_by_id
from evaluate_ecg_against_gold import evaluate

INTERVAL = {'format': '[start_page, end_page]', 'inclusive': True}


class PageGroundingTests(unittest.TestCase):
    def test_explicit_interval(self):
        self.assertEqual(decode_pages([10, 15], INTERVAL), list(range(10, 16)))
        self.assertEqual(decode_pages([10, 10], INTERVAL), [10])

    def test_legacy_list_is_not_guessed(self):
        self.assertEqual(decode_pages([10, 15]), [10, 15])
        self.assertEqual(decode_pages([3, 1, 3]), [1, 3])

    def test_bad_intervals(self):
        for pages in ([15, 10], [10], [1, 2, 3], [0, 3], [True, 3], ['1', 3]):
            with self.subTest(pages=pages), self.assertRaises(ValueError):
                decode_pages(pages, INTERVAL)

    def test_empty_and_node_fallback(self):
        data = {'page_semantics': INTERVAL, 'nodes': [{'id': 'a', 'pages': [2, 4]}]}
        self.assertEqual(gold_pages_by_id(data), {'a': [2, 3, 4]})
        data['grounding'] = {'a': {'pages': []}}
        self.assertEqual(gold_pages_by_id(data), {'a': []})

    def test_scores_coexist(self):
        gold = {'page_semantics': INTERVAL,
                'nodes': [{'id': 'g', 'title': 'Heading', 'kind': 'KC'}],
                'grounding': {'g': {'pages': [10, 15]}}}
        pred = {'nodes': [{'id': 'p', 'title': 'Heading', 'kind': 'KC',
                           'page_start': 10, 'page_end': 15}], 'edges': []}
        with tempfile.TemporaryDirectory() as tmp:
            gp, pp = Path(tmp)/'gold.json', Path(tmp)/'pred.json'
            gp.write_text(json.dumps(gold), encoding='utf-8')
            pp.write_text(json.dumps(pred), encoding='utf-8')
            metrics = evaluate(gp, pp)['metrics']
        self.assertEqual(metrics['page_grounding']['f1'], 0.5)
        self.assertEqual(metrics['page_grounding_encoding_aware']['f1'], 1.0)


if __name__ == '__main__':
    unittest.main()
