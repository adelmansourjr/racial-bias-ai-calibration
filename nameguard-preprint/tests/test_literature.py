"""Keep checked citation artifacts synchronized and important caveats explicit."""
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import literature_tools as lit


class LiteratureIntegrity(unittest.TestCase):
    def test_reference_and_citation_integrity(self):
        self.assertEqual(lit.check()['references'],25)

    def test_range_and_list_citations(self):
        self.assertEqual(lit.cited_ids('text [6-8] and [13-16,25]'),{6,7,8,13,14,15,16,25})

    def test_erratum_is_pinned(self):
        r=lit.references()[6]
        self.assertTrue(r['url'].endswith('2407.20371v3'))
        self.assertIn('inverted',r['version_note'])

    def test_no_invented_documentation_date(self):
        self.assertIsNone(lit.references()[1]['year'])

    def test_no_mismatched_inlp_identifier(self):
        self.assertEqual(lit.references()[12]['url'],'https://aclanthology.org/2020.acl-main.647/')

    def test_search_scope_is_not_systematic(self):
        log=json.loads((ROOT/'literature/search-log.json').read_text())
        self.assertIn('not exhaustive',log['review_type'])
