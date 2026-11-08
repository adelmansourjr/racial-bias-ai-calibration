"""Preserve valid original roles before considering optional formatting repairs."""
from review import propose_whitespace_repair as _propose
from gate import ContractError


def propose_whitespace_repair(source, spans, registry, tokenizer, *, allow_formatting=False):
    try:
        return _propose(source,spans,registry,tokenizer,allow_formatting=False)
    except ContractError:
        if not allow_formatting:
            raise
    return _propose(source,spans,registry,tokenizer,allow_formatting=True)
