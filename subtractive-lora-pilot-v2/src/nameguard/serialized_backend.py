"""Conservative process-wide serialization even across multiple service objects."""
import threading
from .mlx_backend import QwenBackend, load_pinned
from .runtime import RuntimeRejected, ScoringService
from .contracts import ReviewAuthority
from .encoding import ScalarEncoder

_MODEL_GATE = threading.Lock()


class SerialQwenBackend(QwenBackend):
    def infer(self, encoded):
        if not _MODEL_GATE.acquire(blocking=False): raise RuntimeRejected('Protected GPU runtime busy')
        try: return super().infer(encoded)
        finally: _MODEL_GATE.release()


def create_service(checkpoint, manifest, reviewer_keys):
    """Operator-only construction; manifests/keys are never request parameters.

    The returned verify_close callable should be used once inference finishes.
    It checks loaded parameter content and original checkpoint files.
    """
    model, tokenizer, verify_files = load_pinned(checkpoint, manifest)
    backend = SerialQwenBackend(model, tokenizer)
    service = ScoringService(ReviewAuthority(reviewer_keys), ScalarEncoder(tokenizer), backend)
    def verify_close():
        # Stop accepting new requests before the final integrity check.
        service.close()
        parameter_digest = backend.verify_parameters(); verify_files()
        return {'parameter_digest': parameter_digest, 'checkpoint_files_unchanged': True}
    return service, verify_close
