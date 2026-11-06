"""Route original-token activation slots in a pinned, cache-free Qwen scorer."""
from dataclasses import dataclass
import math
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / 'nameguard-validation/src'))
from nameguard.annotations import ConsensusDocument
from nameguard.contracts import ContractError
from nameguard.mlx_backend import QwenBackend
from nameguard.runtime import PublicScore
from nameguard.serialized_backend import _MODEL_GATE


@dataclass(frozen=True)
class Layout:
    ids: tuple[int, ...]
    public: tuple[int, ...]

    @property
    def public_ids(self):
        return tuple(self.ids[i] for i in self.public)


def native_layout(document, tokenizer, limit=2048):
    """Verify a role-boundary factorization and use the full-source IDs for inference."""
    if type(document) is not ConsensusDocument:
        raise ContractError('Verified role document required')
    runs = []
    for node in document.nodes:
        if not node.text:
            continue
        if runs and runs[-1][0] == node.role:
            runs[-1] = (node.role, runs[-1][1] + node.text)
        else:
            runs.append((node.role, node.text))
    source = ''.join(text for _, text in runs)
    ids = tuple(tokenizer.encode(source, add_special_tokens=False))
    if not 0 < len(ids) <= limit or any(type(token) is not int or token < 0 for token in ids):
        raise ContractError('Unsupported native sequence length or token')
    pieces = []
    public = []
    for role, text in runs:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            raise ContractError('Nonempty role run tokenized to an empty sequence')
        if role == 'evidence':
            public.extend(range(len(pieces), len(pieces) + len(tokens)))
        elif role != 'identity':
            raise ContractError('Unsupported role')
        pieces.extend(tokens)
    if tuple(pieces) != ids:
        raise ContractError('Native tokenizer crosses a reviewed role boundary')
    if not public or public[-1] != len(ids) - 1:
        raise ContractError('A public final readout slot is required')
    return Layout(ids, tuple(public))


class SlotScorer(QwenBackend):
    """Fixed logical evidence coordinates with full-length zeroed identity storage."""

    def __init__(self, model, tokenizer, authority):
        super().__init__(model, tokenizer)
        self.authority = authority

    def score(self, wire):
        return self._run(wire, 'packed', None)

    def _run(self, wire, mode, observe):
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        if mode not in ('packed', 'dense-logical', 'dense-physical'):
            raise ValueError('Unsupported execution mode')
        if not _MODEL_GATE.acquire(blocking=False):
            raise ContractError('Protected model runtime busy')
        try:
            self._guard()
            layout = native_layout(self.authority.verify(wire), self._tokenizer)
            model = self._model
            ids = mx.array(layout.ids)[None]
            keep = mx.array(layout.public)
            n = len(layout.ids)
            # For each original slot, give the ordered index of its public token.
            mapping = {slot: rank for rank, slot in enumerate(layout.public)}
            ranks = mx.array([mapping.get(slot, 0) for slot in range(n)])
            public = mx.array([slot in mapping for slot in range(n)])
            full_mask = public[None, :, None]
            def scatter(values):
                return mx.where(full_mask, mx.take(values, ranks, axis=1), 0)
            def emit(stage, values):
                if observe is not None:
                    observe(stage, values)

            emit('native_ids', ids)
            raw = model.model.embed_tokens(ids)
            # Evaluate the original named embedding lookup before the boundary.
            mx.eval(raw)
            emit('raw_embedding', raw)
            state = mx.where(full_mask, raw, 0)
            emit('boundary', state)
            dense = mode != 'packed'
            if dense:
                positions = mx.arange(n)
                causal = positions[:, None] >= positions[None, :]
                allowed = public[:, None] & public[None, :] & causal
                # Each private query has a dummy self-edge. Its output is discarded.
                allowed = allowed | (~public[:, None] & (positions[:, None] == positions[None, :]))
            else:
                mask = create_attention_mask(mx.take(state, keep, axis=1), None)
            for index, layer in enumerate(model.layers):
                if mode == 'packed':
                    active = mx.take(state, keep, axis=1)
                    state = scatter(layer(active, mask=mask, cache=None))
                elif mode == 'dense-physical':
                    state = mx.where(full_mask, layer(state, mask=allowed, cache=None), 0)
                else:
                    state = self._dense_logical(layer, state, keep, ranks, public, allowed)
                emit(f'block_{index+1}', state)
            if dense:
                state = mx.where(full_mask, model.model.norm(state), 0)
            else:
                state = scatter(model.model.norm(mx.take(state, keep, axis=1)))
            emit('final_hidden', state)
            logits = model.model.embed_tokens.as_linear(state[:, -1:].astype(mx.float32)).astype(mx.float32)
            emit('vocabulary', logits)
            margin = logits[0, 0, self._labels[0]] - logits[0, 0, self._labels[1]]
            mx.eval(margin)
            self._guard()
            value = float(margin.item())
            return PublicScore(value, 1 / (1 + math.exp(-max(-700., min(700., value)))))
        finally:
            _MODEL_GATE.release()

    @staticmethod
    def _dense_logical(layer, state, keep, ranks, public, mask):
        """Diagnostic dense kernel with the same logical RoPE and private key mask."""
        import mlx.core as mx
        from mlx_lm.models.base import scaled_dot_product_attention

        attention = layer.self_attn
        normalized = layer.input_layernorm(state)
        batch, length, _ = state.shape

        def projection(operator, heads):
            return operator(normalized).reshape(batch, length, heads, -1).transpose(0, 2, 1, 3)

        def logical_rope(values):
            rotated = attention.rope(mx.take(values, keep, axis=2))
            return mx.where(public[None, None, :, None], mx.take(rotated, ranks, axis=2), 0)

        q = logical_rope(projection(attention.q_proj, attention.n_heads))
        k = logical_rope(projection(attention.k_proj, attention.n_kv_heads))
        v = projection(attention.v_proj, attention.n_kv_heads)
        out = scaled_dot_product_attention(q, k, v, cache=None, scale=attention.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        residual = state + attention.o_proj(out)
        updated = residual + layer.mlp(layer.post_attention_layernorm(residual))
        return mx.where(public[None, :, None], updated, 0)
