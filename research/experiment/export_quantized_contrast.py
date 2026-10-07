"""Recover the demo's exact affine4-bit A/B contrast using independent row groups."""
from pathlib import Path
import argparse
import json
import os
import numpy as np
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import mlx.core as mx
from transformers import AutoTokenizer

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model',type=Path,required=True)
snapshot=parser.parse_args().model
tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
ids=[tok.encode(x,add_special_tokens=False)[0] for x in ['A','B']]
weights=mx.load(str(snapshot/'model.safetensors'))
rows=weights['model.embed_tokens.weight'][mx.array(ids)]
quantized,scales,biases=mx.quantize(rows,group_size=64,bits=4,mode='affine')
dequantized=mx.dequantize(quantized,scales,biases,group_size=64,bits=4,mode='affine')
contrast=(dequantized[0]-dequantized[1]).astype(mx.float32)
mx.eval(contrast)
out=Path(__file__).parent/'results'/'qwen_qlora'
np.save(out/'contrast.npy',np.array(contrast))
old=np.load(out/'activations.npz')
np.savez_compressed(out/'activations-with-contrast.npz',before=old['before'],after=old['after'],contrast=np.array(contrast))
(out/'contrast-provenance.json').write_text(json.dumps({'source_checkpoint':str(snapshot), 'token_ids':ids,
    'method':'MLX affine4bit group64 quantization independently applied to original tied embedding rows',
    'reason_row_equivalent':'MLX affine quantization scales/biases are per row-group, not across vocabulary rows'},indent=2)+'\n')
print('Saved independently reconstructed quantized contrast.')
