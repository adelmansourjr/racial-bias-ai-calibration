"""Load the locally fitted adapter and score an intact named A/B prompt."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiment import ROOT, TAPS, load_pinned, parameter_digest, score_record, sha


def score_prompt(prompt, checkpoint):
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    selection = json.loads((ROOT/'audit/selection.json').read_text())
    path = ROOT/'local-private/adapter.npz'
    if sha(path.read_bytes()) != selection['adapter_sha256']:
        raise ValueError('Adapter digest mismatch')
    with np.load(path, allow_pickle=False) as data:
        mean, weights, direction = [data[key] for key in ('mean', 'weights', 'direction')]
        mode = str(data['mode'])
    if mode != selection['selected']['mode'] or mode not in ('final', 'taps'):
        raise ValueError('Unexpected adapter mode')
    width = len(TAPS)*1536 if mode == 'taps' else 1536
    if mean.shape != (width,) or weights.shape != (width,) or direction.shape != (1536,):
        raise ValueError('Unexpected adapter shape')
    for values in (mean, weights, direction):
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError('Expected finite float32 adapter values')
    manifest = ROOT/'audit/checkpoint-manifest.json'
    model, tokenizer, verify_files = load_pinned(checkpoint, manifest)
    model.freeze()
    model.eval()
    expected = json.loads(manifest.read_text())['parameter_digest']
    if parameter_digest(model) != expected:
        raise ValueError('Model parameter digest mismatch')
    serialized = tokenizer.apply_chat_template([{'role':'user','content':prompt}],
                                               tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(serialized, add_special_tokens=False)
    labels = [tokenizer.encode(letter,add_special_tokens=False) for letter in 'AB']
    if any(len(label)!=1 for label in labels):
        raise ValueError('A/B readout requires single-token choices')
    labels = [label[0] for label in labels]
    hidden = model.model.embed_tokens(mx.array(ids)[None])
    mask = create_attention_mask(hidden,None)
    taps = []
    for index, layer in enumerate(model.layers,1):
        hidden = layer(hidden,mask=mask,cache=None)
        if index in TAPS:
            taps.append(model.model.norm(hidden[:, -1:])[0,0].astype(mx.float32))
    mx.eval(*taps)
    features = np.concatenate([np.array(tap) for tap in taps]) if mode=='taps' else np.array(taps[-1])
    correction = mx.sum((mx.array(features)-mx.array(mean))*mx.array(weights))
    edited = taps[-1] - mx.array(direction)*correction
    scores = []
    for h in (taps[-1], edited):
        logits = model.model.embed_tokens.as_linear(h[None,None])[0,0].astype(mx.float32)
        mx.eval(h,logits)
        z = np.array(logits)
        scores.append(score_record(float(z[labels[0]]-z[labels[1]]),z,np.array(h),labels))
    if parameter_digest(model)!=expected:
        raise ValueError('Model parameter digest changed')
    verify_files()
    return dict(native=scores[0],adapted=scores[1],input_tokens=len(ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--prompt-file',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(score_prompt(args.prompt_file.read_text(),args.checkpoint),indent=2))


if __name__=='__main__':
    main()
