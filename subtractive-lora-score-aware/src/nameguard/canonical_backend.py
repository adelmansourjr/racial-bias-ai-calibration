"""Native-token-count evidence path built INSIDE the model after raw embedding.

This explicitly constructs canonical retained evidence internally; it is not
learned semantic unlearning and is not the earlier scalarwise gate.
"""
import math
from .mlx_backend import QwenBackend, load_pinned
from .serialized_backend import _MODEL_GATE
from .runtime import PublicScore, RuntimeRejected, ScoringService
from .annotations import DualReviewAuthority
from .native_encoding import NativeSourceEncoder, PreparedSource
from .encoding import MAX_PUBLIC_TOKENS


class CanonicalBackend(QwenBackend):
    def infer(self, source):
        if not _MODEL_GATE.acquire(blocking=False): raise RuntimeRejected('Protected model runtime busy')
        try: return self._infer(source)
        finally: _MODEL_GATE.release()
    def _infer(self, source):
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask
        if type(source) is not PreparedSource: raise RuntimeRejected('Original source preparation required')
        self._guard(); model = self._model
        # This MUST happen before canonical evidence construction/tokenization.
        # All original names enter the original embedding module unchanged.
        raw = model.model.embed_tokens(mx.array(source.all_ids)[None]); mx.eval(raw)
        self._emit('raw_embedding', raw)
        evidence = ''.join(node.text for node in source.document.nodes if node.role == 'evidence')
        ids = self._tokenizer.encode(evidence, add_special_tokens=False)
        if not 0 < len(ids) <= MAX_PUBLIC_TOKENS or any(type(i) is not int or i < 0 for i in ids):
            raise RuntimeRejected('Unsupported canonical evidence tokens; no truncation')
        inputs = mx.array(ids)[None]; self._emit('canonical_ids', inputs)
        hidden = model.model.embed_tokens(inputs); self._emit('compact_embedding', hidden)
        mask = create_attention_mask(hidden, None)
        for index, layer in enumerate(model.layers):
            hidden = layer(hidden, mask=mask, cache=None); self._emit(f'layer_{index}', hidden)
        hidden = model.model.norm(hidden); self._emit('final_hidden', hidden)
        logits = model.model.embed_tokens.as_linear(hidden[:, -1:].astype(mx.float32)).astype(mx.float32)
        self._emit('final_vocab', logits)
        margin = logits[0, 0, self._labels[0]] - logits[0, 0, self._labels[1]]; mx.eval(margin)
        self._guard(); value = float(margin.item())
        return PublicScore(value, 1 / (1 + math.exp(-max(-700., min(700., value)))))


def create_native_service(checkpoint, manifest, reviewer_keys):
    model, tokenizer, verify_files = load_pinned(checkpoint, manifest)
    backend = CanonicalBackend(model, tokenizer)
    service = ScoringService(DualReviewAuthority(reviewer_keys), NativeSourceEncoder(tokenizer), backend)
    def verify_close():
        service.close(); digest = backend.verify_parameters(); verify_files()
        return {'parameter_digest': digest, 'checkpoint_files_unchanged': True}
    return service, verify_close
