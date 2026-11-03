"""Apply the selected subtractive activation adapter to an intact raw prompt."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nameguard.mlx_backend import load_pinned,parameter_digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--prompt-file',required=True);p.add_argument('--basis',default=str(ROOT/'adapters/subtractive-basis.npz'))
    args=p.parse_args()
    import mlx.core as mx
    import numpy as np
    data=np.load(args.basis,allow_pickle=False);u=data['U'];mu=data['mu'];scale=float(data['scale'])
    if u.ndim!=2 or u.shape[0]!=1536 or mu.shape!=(1536,)or not 0<scale<=1:raise ValueError('Unsupported adapter dimensions/scale')
    if not np.isfinite(u).all()or not np.isfinite(mu).all():raise ValueError('Nonfinite adapter')
    if not np.allclose(u.T@u,np.eye(u.shape[1]),atol=1e-5):raise ValueError('Directions must be orthonormal')
    model,tok,verify_files=load_pinned(args.checkpoint,ROOT/'audit/checkpoint-manifest.json')
    text=Path(args.prompt_file).read_text()
    template=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
    ids=tok.encode(template,add_special_tokens=False)
    h=model.model(mx.array(ids)[None])[0,-1].astype(mx.float32)
    edited=h-scale*((h-mx.array(mu))@mx.array(u))@mx.array(u).T
    logits=model.model.embed_tokens.as_linear(h[None,None])[0,0].astype(mx.float32)
    adapted=model.model.embed_tokens.as_linear(edited[None,None])[0,0].astype(mx.float32)
    mx.eval(h,edited,logits,adapted)
    indices=[tok.encode(c,add_special_tokens=False)[0]for c in 'AB']
    digest=parameter_digest(model);verify_files()
    if digest!=json.loads((ROOT/'audit/checkpoint-manifest.json').read_text())['parameter_digest']:raise ValueError('Base parameter drift')
    print(json.dumps({'raw_margin':float((logits[indices[0]]-logits[indices[1]]).item()),
      'adapted_margin':float((adapted[indices[0]]-adapted[indices[1]]).item()),
      'input_tokens':len(ids),'input_ids_sha256':hashlib.sha256(np.array(ids,dtype=np.int64).tobytes()).hexdigest(),
      'rank':u.shape[1],'scale':scale,'no_name_detection_or_input_rewrite':True,
      'checkpoint_and_parameters_unchanged':True,'scope':'Experimental final-state adapter; not a calibrated hiring score or universal name-invariance guarantee.'},indent=2))


if __name__=='__main__':main()
