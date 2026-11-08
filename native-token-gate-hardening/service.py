"""Normal entry points for raw scoring and a fixed credential-task readout."""
import hashlib
import json
from pathlib import Path

from gate import SlotScorer
from calibration import CredentialCalibration
from nameguard.mlx_backend import load_pinned


def load_credential_calibration(path, expected_digest):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_digest:
        raise ValueError('Calibration artifact digest mismatch')
    artifact = json.loads(data)
    expected = {'profile','threshold','frozen_before_test','fit_rows',
                'fit_correct_uncalibrated','fit_correct_calibrated'}
    if set(artifact) != expected or artifact['frozen_before_test'] is not True:
        raise ValueError('Unsupported calibration artifact schema')
    return CredentialCalibration(artifact['threshold'],artifact['profile'])


class CredentialScoringService:
    """The operator configures the task profile, not an untrusted request flag."""

    def __init__(self, scorer, calibration):
        if type(calibration) is not CredentialCalibration:
            raise ValueError('Validated fixed credential calibration required')
        self._scorer = scorer
        self._calibration = calibration

    def score(self, wire):
        return self._calibration.apply(self._scorer.score(wire))

    def close(self):
        self._scorer.close()


def create_service(checkpoint, manifest, reviewer_keys, *, calibration=None):
    model, tokenizer, verify_files = load_pinned(checkpoint,manifest)
    from nameguard.annotations import DualReviewAuthority
    raw = SlotScorer(model,tokenizer,DualReviewAuthority(reviewer_keys))
    service = CredentialScoringService(raw,calibration) if calibration is not None else raw

    def finish():
        service.close()
        digest = raw.verify_parameters()
        verify_files()
        return {'parameter_digest':digest,'checkpoint_files_verified':True}
    return service, tokenizer, finish
