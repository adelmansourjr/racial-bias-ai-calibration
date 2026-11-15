from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reserve import extract_prompt, witness_matches


class TestCandidateBlindExtraction(unittest.TestCase):
    def test_prompt_does_not_use_candidate_or_offsets(self):
        item = {"passage": "The key is blue.", "question": "Which color is the key?",
                "answer": "blue", "start": 11}
        self.assertEqual(extract_prompt(item), extract_prompt(dict(item, answer="green", start=4)))

    def test_stopped_exact_source_matches(self):
        for text in ('blue', ' "blue" ', '“blue”'):
            self.assertTrue(witness_matches(text, 'blue', True))
        for text in ('Blue', 'blue.', 'blue and green', 'It was blue.', 'blue\nExplanation.'):
            self.assertFalse(witness_matches(text, 'blue', True))
        self.assertFalse(witness_matches('UNKNOWN', 'UNKNOWN', True))
        self.assertFalse(witness_matches('blue', 'blue', False))
