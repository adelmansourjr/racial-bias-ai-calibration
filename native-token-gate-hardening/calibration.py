"""Fixed scalar task calibration fitted independently of identity substitutions."""
from dataclasses import dataclass
import math

from nameguard.runtime import PublicScore


@dataclass(frozen=True)
class CredentialCalibration:
    threshold: float
    profile: str = 'synthetic-active-credential-v1'

    def __post_init__(self):
        if type(self.threshold) not in (float, int) or not math.isfinite(self.threshold):
            raise ValueError('Finite fixed threshold required')
        if self.profile != 'synthetic-active-credential-v1':
            raise ValueError('Unsupported task profile')

    def apply(self, score):
        if type(score) is not PublicScore:
            raise ValueError('Expected a score from the protected scorer')
        value = score.margin - self.threshold
        return PublicScore(value, 1 / (1 + math.exp(-max(-700., min(700., value)))))


def fit_credential_threshold(rows):
    """Minimize fitting errors, then choose the least absolute score shift."""
    if not rows or any(type(row['label']) is not bool or not math.isfinite(row['margin']) for row in rows):
        raise ValueError('Finite labeled fitting rows required')
    if {row['label'] for row in rows} != {False, True}:
        raise ValueError('Both qualification labels required')
    distinct = sorted({row['margin'] for row in rows})
    candidates = {0., distinct[0]-1, distinct[-1]+1}
    candidates.update((left+right)/2 for left, right in zip(distinct, distinct[1:]))
    def criterion(threshold):
        mistakes = sum((row['margin'] >= threshold) != row['label'] for row in rows)
        return mistakes, abs(threshold), threshold
    return CredentialCalibration(float(min(candidates, key=criterion)))
