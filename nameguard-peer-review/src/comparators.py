"""Same-policy external retained-ID baseline; identity rows are never embedded."""
import math
import bootstrap
from gate import SlotScorer,ContractError
from nameguard.runtime import PublicScore


class ExternalEvidenceScorer(SlotScorer):
    """Same authentication, layout witness, locks, model guard and scalar output.

    The intervention occurs before the embedding lookup, solely for comparison.
    This is a strong equivalent baseline, not the proposed internal path.
    """
    def _infer_layout(self,layout,mode,observe):
        if mode!='packed' or observe is not None:raise ContractError('Score-only reference')
        import mlx.core as mx
        self._guard()
        state=self._model.model(mx.array(layout.public_ids)[None])
        logits=self._model.model.embed_tokens.as_linear(state[:,-1:].astype(mx.float32)).astype(mx.float32)
        margin=logits[0,0,self._labels[0]]-logits[0,0,self._labels[1]]
        mx.eval(margin)
        self._guard()
        value=float(margin.item())
        return PublicScore(value,1/(1+math.exp(-max(-700.,min(700.,value)))))
