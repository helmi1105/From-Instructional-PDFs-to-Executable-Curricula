import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('conversion_checks',Path(__file__).resolve().parents[1]/'docling/conversion_checks.py')
checks=importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class ConversionCompletenessTests(unittest.TestCase):
    def test_partial_conversion_is_rejected_even_with_all_page_records(self):
        self.assertFalse(checks.conversion_complete('partial_success',[],[1,2,3],{1,2,3}))

    def test_missing_and_duplicate_processed_pages_are_rejected(self):
        self.assertFalse(checks.conversion_complete('success',[],[1,2],{1,2,3}))
        self.assertFalse(checks.conversion_complete('success',[],[1,2,3,3],{1,2,3}))

    def test_blank_page_is_allowed_when_processing_completed(self):
        # Text item counts are intentionally irrelevant: blank pages are legitimate.
        self.assertTrue(checks.conversion_complete('success',[],[1,2,3],{1,2,3}))

    def test_reported_error_prevents_acceptance(self):
        self.assertFalse(checks.conversion_complete('success',['preprocess failed'],[1,2,3],{1,2,3}))
