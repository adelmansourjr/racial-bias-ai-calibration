import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from service import CredentialScoringService, load_credential_calibration
from calibration import CredentialCalibration
from nameguard.runtime import PublicScore


class Scorer:
    def __init__(self):
        self.closed = False

    def score(self, wire):
        return PublicScore(5.,.99)

    def close(self):
        self.closed = True


class NormalService(unittest.TestCase):
    def test_service_uses_fixed_task_offset(self):
        service = CredentialScoringService(Scorer(),CredentialCalibration(4.))
        self.assertEqual(service.score(b'fixture').margin,1.)
        with self.assertRaises(TypeError):
            service.score(b'fixture',threshold=0.)

    def test_close_propagates(self):
        scorer = Scorer()
        CredentialScoringService(scorer,CredentialCalibration(1.)).close()
        self.assertTrue(scorer.closed)

    def test_calibration_digest_detects_changed_artifact(self):
        data = {
            'profile':'synthetic-active-credential-v1','threshold':4.,'frozen_before_test':True,
            'fit_rows':32,'fit_correct_uncalibrated':16,'fit_correct_calibrated':32,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'calibration.json'
            path.write_text(json.dumps(data))
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            fitted = load_credential_calibration(path,expected)
            self.assertEqual(fitted.threshold,4.)
            data['threshold'] = 0.
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'digest'):
                load_credential_calibration(path,expected)
