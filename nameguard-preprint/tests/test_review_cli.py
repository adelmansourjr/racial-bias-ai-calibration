import json
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import unittest
from nameguard.annotations import DualReviewAuthority
from test_annotations import proposal


class OfflineReviewWorkflow(unittest.TestCase):
    def test_candidates_compare_two_endorsements_and_assemble(self):
        script=Path(__file__).resolve().parents[1]/'scripts/review_tool.py'
        with tempfile.TemporaryDirectory(prefix='nameguard-review-test-')as directory:
            root=Path(directory);p=proposal();source=''.join(n['text']for n in p['nodes'])
            (root/'source.txt').write_text(source);(root/'proposal.json').write_text(json.dumps(p))
            keys={'one':secrets.token_bytes(32),'two':secrets.token_bytes(32)}
            def run(*args):
                result=subprocess.run([sys.executable,str(script),*map(str,args)],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
            run('candidates','--source',root/'source.txt','--identity','John Miller','--output',root/'queue.json')
            queue=json.loads((root/'queue.json').read_text());self.assertIs(queue['resolved'],False)
            self.assertEqual(len(queue['candidates']),2)
            run('compare','--first',root/'proposal.json','--second',root/'proposal.json','--output',root/'agreement.json')
            self.assertEqual(json.loads((root/'agreement.json').read_text())['cohen_kappa'],1.)
            for reviewer,key in keys.items():
                (root/f'{reviewer}.key').write_bytes(key)
                run('endorse','--proposal',root/'proposal.json','--reviewer',reviewer,'--key-file',root/f'{reviewer}.key','--output',root/f'{reviewer}.json')
            run('assemble','--proposal',root/'proposal.json','--approvals',root/'one.json',root/'two.json','--output',root/'packet.json')
            doc=DualReviewAuthority(keys).verify((root/'packet.json').read_bytes())
            self.assertEqual(''.join(n.text for n in doc.nodes),source)
