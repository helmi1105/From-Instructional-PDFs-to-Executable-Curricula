import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_logging import parse_json_response


class ResponseParsingTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_json_response(' {"x": 1} '), ({'x': 1}, 'none'))

    def test_markdown_json(self):
        for fence in ('```json', '```', '```JSON'):
            self.assertEqual(parse_json_response(f'{fence}\n{{"x": 1}}\n```'),
                             ({'x': 1}, 'markdown_fence_removed'))

    def test_reject_prose_and_broken_payloads(self):
        for text in ('Here is JSON: {"x":1}', '```json\n{"x":\n```',
                     '```json\n{}\n```\n{"other": 1}',
                     '```json\n{}\n```\n```json\n{}\n```',
                     '```json\n{}\n```\n[]', '[]', ''):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_json_response(text)

    def test_fenced_json_with_trailing_commentary(self):
        self.assertEqual(
            parse_json_response('```json\n{"sections": []}\n```\n**Key repairs:**\n- Corrected a page span.'),
            ({'sections': []}, 'markdown_fence_and_trailing_commentary_removed'))

    def test_reject_broken_json_even_with_commentary(self):
        with self.assertRaises(ValueError):
            parse_json_response('```json\n{"sections":\n```\n**Key repairs:**\n- Changed outline.')


if __name__ == '__main__':
    unittest.main()
