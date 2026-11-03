"""Trusted, pinned Qwen backend. Importing CPU contracts never imports MLX."""
import hashlib
import json
import math
from pathlib import Path
from .runtime import PublicScore, RuntimeRejected

PARAMETER_DIGEST = 'c6c622310d66489a5a9de5f438c23063bf784b3031f2e8af64bfd60da0394b40'
EXPECTED_ARCHITECTURE = ('mlx_lm.models.qwen2', 'Model', 1536, 28, True)
OVERLAY_FLAGS = ['_active_overlay', '_name_gate_active', '_name_invariance_overlay_active', '_raw_name_adapter_active']


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        while block := source.read(8 * 1024 * 1024): h.update(block)
    return h.hexdigest()


def parameter_digest(model):
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    h = hashlib.sha256()
    for key, value in sorted(tree_flatten(model.parameters())):
        mx.eval(value); h.update(key.encode()); h.update(str(value.shape).encode()); h.update(str(value.dtype).encode())
        h.update(np.array(value.astype(mx.float32) if value.dtype == mx.bfloat16 else value).tobytes())
    return h.hexdigest()


class QwenBackend:
    def __init__(self, model, tokenizer):
        architecture = (type(model).__module__, type(model).__name__, model.args.hidden_size, len(model.layers), model.args.tie_word_embeddings)
        if architecture != EXPECTED_ARCHITECTURE: raise RuntimeRejected('Unsupported architecture')
        model.freeze(); model.eval()
        self._model = model; self._tokenizer = tokenizer
        self._check_flags()
        if parameter_digest(model) != PARAMETER_DIGEST: raise RuntimeRejected('Unrecognized base parameter values')
        self._original_structure = self._structure()
        labels = [tokenizer.encode(label, add_special_tokens=False) for label in ['A', 'B']]
        if any(len(ids) != 1 for ids in labels): raise RuntimeRejected('Unsupported original vocabulary labels')
        self._labels = [ids[0] for ids in labels]
    def _check_flags(self):
        if any(getattr(self._model, flag, False) for flag in OVERLAY_FLAGS): raise RuntimeRejected('Another model overlay is active')
    def _structure(self):
        from mlx.utils import tree_flatten
        modules = tuple((name, type(module).__module__, type(module).__qualname__, id(module))
                        for name, module in self._model.named_modules())
        arrays = tuple((key, id(value), tuple(value.shape), str(value.dtype))
                       for key, value in sorted(tree_flatten(self._model.parameters())))
        return modules, arrays
    def _guard(self):
        self._check_flags()
        if self._structure() != self._original_structure: raise RuntimeRejected('Model structure or parameter objects changed')
    def _emit(self, stage, value):
        """No-op in serving. Trusted experimental subclasses may compare tensors."""
    def infer(self, encoded):
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask
        self._guard()
        model = self._model
        source_ids = mx.array(encoded.all_ids)[None]
        raw = model.model.embed_tokens(source_ids); mx.eval(raw)
        self._emit('raw_embedding', raw)
        # The exclusion is AFTER all original scalar tokens are embedded.
        public_ids = mx.take(source_ids, mx.array(encoded.keep), axis=1)
        hidden = mx.take(raw, mx.array(encoded.keep), axis=1)
        self._emit('compact_embedding', hidden)
        mask = create_attention_mask(hidden, None)
        for index, layer in enumerate(model.layers):
            hidden = layer(hidden, mask=mask, cache=None)
            self._emit(f'layer_{index}', hidden)
        hidden = model.model.norm(hidden); self._emit('final_hidden', hidden)
        logits = model.model.embed_tokens.as_linear(hidden[:, -1:].astype(mx.float32)).astype(mx.float32)
        self._emit('final_vocab', logits)
        margin = logits[0, 0, self._labels[0]] - logits[0, 0, self._labels[1]]
        mx.eval(margin)
        self._guard()
        value = float(margin.item())
        probability = 1 / (1 + math.exp(-max(-700., min(700., value))))
        return PublicScore(value, probability)
    def verify_parameters(self):
        self._guard()
        if parameter_digest(self._model) != PARAMETER_DIGEST: raise RuntimeRejected('Model parameter content changed')
        return PARAMETER_DIGEST


def load_pinned(checkpoint, manifest):
    """Manifest is trusted operator configuration, never supplied in a request."""
    import os
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    checkpoint = Path(checkpoint); files = json.loads(Path(manifest).read_text())['files']
    def verify_files():
        current = {str(p.relative_to(checkpoint)): file_digest(p) for p in checkpoint.rglob('*') if p.is_file()}
        expected = {key: value['sha256'] for key, value in files.items()}
        if current != expected: raise RuntimeRejected('Checkpoint manifest mismatch')
    verify_files()
    from mlx_lm import load
    import mlx.core as mx
    model, tokenizer = load(str(checkpoint), lazy=True)
    mx.set_cache_limit(128 * 1024 * 1024)
    return model, tokenizer, verify_files
