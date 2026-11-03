"""Run the learned subtractive rank-one adapter on an unchanged named prompt."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nameguard.mlx_backend import load_pinned,parameter_digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--prompt-file',required=True)
    args=p.parse_args()
    import mlx.core as mx
    import numpy as np
    with np.load(ROOT/'adapters/subtractive-rank-one.npz',allow_pickle=False)as data:
        a,v,mu=[data[k]for k in ['a','v','mu']]
    if any(x.shape!=(1536,)or not np.isfinite(x).all()for x in [a,v,mu]):raise ValueError('Invalid adapter')
    model,tok,verify_files=load_pinned(args.checkpoint,ROOT/'audit/checkpoint-manifest.json')
    content=Path(args.prompt_file).read_text()
    text=tok.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
    ids=tok.encode(text,add_special_tokens=False)
    h=model.model(mx.array(ids)[None])[0,-1].astype(mx.float32)
    edited=h-mx.array(a)*mx.sum(mx.array(v)*(h-mx.array(mu)))
    raw=model.model.embed_tokens.as_linear(h[None,None])[0,0].astype(mx.float32)
    corrected=model.model.embed_tokens.as_linear(edited[None,None])[0,0].astype(mx.float32)
    labels=[tok.encode(c,add_special_tokens=False)[0]for c in 'AB'];mx.eval(raw,corrected)
    digest=parameter_digest(model);verify_files()
    if digest!=json.loads((ROOT/'audit/checkpoint-manifest.json').read_text())['parameter_digest']:raise ValueError('Base drift')
    print(json.dumps({'raw_margin':float((raw[labels[0]]-raw[labels[1]]).item()),
      'adapted_margin':float((corrected[labels[0]]-corrected[labels[1]]).item()),'tokens':len(ids),
      'input_ids_sha256':hashlib.sha256(np.array(ids,dtype=np.int64).tobytes()).hexdigest(),
      'original_named_text_retained':True,'base_parameters_and_files_unchanged':True,
      'scope':'Experimental rank-one final-state activation adapter; no universal invariance or calibrated hiring accuracy guarantee.'},indent=2))


if __name__=='__main__':main()
