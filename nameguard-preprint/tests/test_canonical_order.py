"""Inspect the dependency order without an MLX dependency or fake GPU result."""
import ast
from pathlib import Path
import unittest


class ConstructionOrder(unittest.TestCase):
    def test_original_source_embedding_precedes_canonical_evidence_construction(self):
        # Static wiring regression only; the actual GPU experiment separately
        # observes real named-source embeddings and every public tensor stage.
        source=(Path(__file__).resolve().parents[1]/'src/nameguard/canonical_backend.py').read_text()
        tree=ast.parse(source)
        function=next(node for node in ast.walk(tree)if isinstance(node,ast.FunctionDef)and node.name=='_infer')
        lines={}
        for node in ast.walk(function):
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Name):lines[target.id]=node.lineno
        self.assertLess(lines['raw'],lines['evidence'])
        self.assertLess(lines['evidence'],lines['ids'])
        self.assertLess(lines['ids'],lines['hidden'])
