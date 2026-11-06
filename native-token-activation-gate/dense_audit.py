"""Localize dense-route length sensitivity on fixed synthetic evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import numpy as np
from gate import SlotScorer, native_layout
from nameguard.annotations import DualReviewAuthority, endorse, packet, proposal_from_spans
from nameguard.mlx_backend import load_pinned
ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args()
    target=ROOT/'results/dense-audit.json'
    if target.exists():
        raise FileExistsError(target)
    import mlx.core as mx
    from mlx_lm.models.base import scaled_dot_product_attention
    model,tok,verify_files=load_pinned(args.checkpoint,ROOT/'audit/checkpoint-manifest.json')
    keys={key:secrets.token_bytes(32) for key in ('a','b')}
    authority=DualReviewAuthority(keys)
    gate=SlotScorer(model,tok,authority)
    runs=[]
    prefix='Applicant name:\n'
    suffix='\nThe applicant has a current CPA credential. Answer A if qualified. Otherwise B.'
    for name in ('Ai','Inês de Albuquerque-Silva'):
        text=tok.apply_chat_template([{'role':'user','content':prefix+name+suffix}],
                                     tokenize=False,add_generation_prompt=True)
        start=text.index(prefix)+len(prefix)
        payload=proposal_from_spans(text,[{'start':start,'end':start+len(name),'text':name}],[name],resolved=True)
        request=packet(payload,[endorse(payload,reviewer=k,key=v) for k,v in keys.items()])
        layout=native_layout(authority.verify(request),tok)
        ids=mx.array(layout.ids)[None]
        keep=mx.array(layout.public)
        mapping={slot:rank for rank,slot in enumerate(layout.public)}
        ranks=mx.array([mapping.get(slot,0) for slot in range(len(layout.ids))])
        public=mx.array([slot in mapping for slot in range(len(layout.ids))])
        slots=mx.arange(len(layout.ids))
        causal=slots[:,None]>=slots[None,:]
        mask=(public[:,None]&public[None,:]&causal)|(~public[:,None]&(slots[:,None]==slots[None,:]))
        h=mx.where(public[None,:,None],model.model.embed_tokens(ids),0)
        layer=model.layers[0]
        attention=layer.self_attn
        arrays={}
        def trace(key,value,axis=1):
            selected=mx.take(value,keep,axis=axis)
            mx.eval(selected)
            arrays[key]=np.array(selected.astype(mx.float32))
            return value
        trace('boundary',h)
        x=trace('input_norm',layer.input_layernorm(h))
        batch,length,_=x.shape
        def project(key,operator,heads):
            value=trace(key,operator(x))
            return value.reshape(batch,length,heads,-1).transpose(0,2,1,3)
        q=project('q_proj',attention.q_proj,attention.n_heads)
        k=project('k_proj',attention.k_proj,attention.n_kv_heads)
        v=project('v_proj',attention.v_proj,attention.n_kv_heads)
        def rope(key,value):
            rotated=attention.rope(mx.take(value,keep,axis=2))
            full=mx.where(public[None,None,:,None],mx.take(rotated,ranks,axis=2),0)
            return trace(key,full,2)
        q,k=rope('q_rope',q),rope('k_rope',k)
        out=scaled_dot_product_attention(q,k,v,cache=None,scale=attention.scale,mask=mask)
        out=trace('attention',out.transpose(0,2,1,3).reshape(batch,length,-1))
        residual=trace('residual',h+trace('o_proj',attention.o_proj(out)))
        normalized=trace('post_norm',layer.post_attention_layernorm(residual))
        mlp=trace('mlp',layer.mlp(normalized))
        trace('block_1',mx.where(public[None,:,None],residual+mlp,0))
        runs.append(arrays)
    comparisons={}
    for stage,left in runs[0].items():
        right=runs[1][stage]
        if left.shape!=right.shape:
            raise ValueError('Evidence geometry mismatch')
        comparisons[stage]={'exact':left.tobytes()==right.tobytes(),
                            'max_absolute_error':float(np.max(np.abs(left-right)))}
    result={'synthetic_case':True,'first_block_stage_comparison':comparisons,
            'first_unequal_stage':next((k for k,v in comparisons.items() if not v['exact']),None),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'parameter_digest':gate.verify_parameters()}
    verify_files()
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
