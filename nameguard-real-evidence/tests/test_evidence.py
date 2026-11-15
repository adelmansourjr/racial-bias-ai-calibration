from dataclasses import replace
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence import verify_excerpt, check_provenance
from evidence_v2 import verify_excerpt as expanded
from evidence_v3 import verify_excerpt as guarded
from actor_guard import verify_excerpt as actor_guard


class DocumentEvidence(unittest.TestCase):
    def test_verbatim_binding_and_tamper(self):
        text = "I built reconciliation reports using SQL. I also reviewed notes."
        result = verify_excerpt(text, "SQL")
        self.assertEqual(result.decision, "supported")
        self.assertTrue(check_provenance(text, result))
        self.assertFalse(check_provenance(text.replace("SQL", "XML"), result))
        shifted = replace(result, evidence=replace(result.evidence, end=result.evidence.end + 1))
        self.assertFalse(check_provenance(text, shifted))

    def test_skill_negation_and_hypothesis_refer(self):
        snippets = ("Skills: SQL, Excel, Word.", "I did not build reports using SQL.",
                    "I would build reports using SQL.", "Training on SQL. No practical example supplied.")
        for text in snippets:
            self.assertEqual(verify_excerpt(text, "SQL").decision, "review", text)

    def test_later_grammar_captures_fronted_use(self):
        text = "Using Excel, I created monthly invoices and maintained accounting notes."
        result = expanded(text, "Excel")
        self.assertTrue(check_provenance(text, result))
        self.assertEqual(result.decision, "supported")

    def test_slash_ambiguity_from_fresh_review_round(self):
        text = "Project notes. Tools Suggested/Used spreadsheet tools, SQL, and Python."
        self.assertEqual(expanded(text, "SQL").decision, "supported")
        self.assertEqual(guarded(text, "SQL").decision, "review")

    def test_actor_guard_refs_passive_and_generic_experience(self):
        for text in ("The calculation forms were created using Excel and exported.",
                     "Extensive experience with Excel and common office tools."):
            self.assertEqual(actor_guard(text, "Excel").decision, "review")
        self.assertEqual(actor_guard("I created weekly reports using Excel.", "Excel").decision, "supported")
