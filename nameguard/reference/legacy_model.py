"""Universal noninterference for explicitly typed identity fields.

This is a NEW structural mode, not a proof about the earlier free-form learned
extractor. Every field is tokenized intact and embedded. Identity activations
are excluded inside the model before attention and position assignment.
"""
import hashlib
import numpy as np

MAX_NODES=128
MAX_IDENTITY_BYTES=16384
MAX_PUBLIC_TOKENS=2048
MAX_TOTAL_TOKENS=8192


def encode_document(tokenizer,nodes):
    if not isinstance(nodes,list)or not 1<=len(nodes)<=MAX_NODES:raise ValueError('Bounded typed document required')
    ids=[];keep=[];identity_bytes=0;public_fields=[]
    for node in nodes:
        if not isinstance(node,dict)or set(node)!={'role','text'}:raise ValueError('Exact role/text schema required')
        if node['role']not in {'identity','evidence'}or not isinstance(node['text'],str):raise ValueError('Unsupported field role/text')
        try:payload=node['text'].encode('utf-8',errors='strict')
        except UnicodeEncodeError as e:raise ValueError('Invalid Unicode')from e
        if len(payload)>131072:raise ValueError('Field too large')
        if node['role']=='identity':identity_bytes+=len(payload)
        else:public_fields.append(node['text'])
        # Encode every original field intact. Field boundaries are fixed parts
        # of the public input contract, preventing name-dependent BPE merges.
        field_ids=tokenizer.encode(node['text'],add_special_tokens=False)
        if node['role']=='evidence':keep.extend(range(len(ids),len(ids)+len(field_ids)))
        ids.extend(field_ids)
    if identity_bytes>MAX_IDENTITY_BYTES:raise ValueError('Identity payload exceeds accepted 16 KiB')
    if not 0<len(keep)<=MAX_PUBLIC_TOKENS or len(ids)>MAX_TOTAL_TOKENS:raise ValueError('Unsupported context; no truncation')
    return ids,keep,public_fields


def build_model(base,tokenizer):
    import mlx.core as mx
    import mlx.nn as nn
    if type(base).__module__!='mlx_lm.models.qwen2'or base.args.hidden_size!=1536 or len(base.layers)!=28:
        raise ValueError('Only the tested frozen Qwen architecture is supported')
    class CertifiedModel(nn.Module):
        def __init__(self):super().__init__();self.backbone=base
        def __call__(self,nodes,cache=None,audit=False):
            if cache is not None:raise ValueError('Fresh deterministic prefill only; external caches forbidden')
            ids,keep,_=encode_document(tokenizer,nodes)
            all_ids=mx.array(ids)[None]
            # The original names actually enter the model's embedding module.
            raw=self.backbone.model.embed_tokens(all_ids);mx.eval(raw)
            index=mx.array(keep)
            compact=mx.take(raw,index,axis=1)
            public_ids=mx.take(all_ids,index,axis=1)
            # Only compact public tensors enter attention, masks, positions
            # and the readout. Raw positions, lengths and identities do not.
            hidden=self.backbone.model(public_ids,input_embeddings=compact)
            logits=self.backbone.model.embed_tokens.as_linear(hidden[:,-1:].astype(mx.float32))[0,0].astype(mx.float32)
            a,b=[tokenizer.encode(s,add_special_tokens=False)for s in ['A','B']]
            if len(a)!=1 or len(b)!=1:raise ValueError('Fixed one-token response labels required')
            margin=logits[a[0]]-logits[b[0]];mx.eval(hidden,logits,margin)
            m=float(margin.item());probability=float(1/(1+np.exp(-np.clip(m,-700,700))))
            result={'margin':m,'conditional_A_probability':probability}
            if audit:
                digest=lambda array:hashlib.sha256(np.ascontiguousarray(np.array(array.astype(mx.float32))).tobytes()).hexdigest()
                # Audit is explicitly private metadata, excluded from the
                # theorem about public scores and downstream activations.
                result['audit']={'raw_embedding_hash':digest(raw),'raw_token_count':len(ids),
                  'public_ids_hash':hashlib.sha256(np.asarray([ids[i]for i in keep],np.int64).tobytes()).hexdigest(),
                  'bottleneck_hash':digest(compact),'complete_hidden_hash':digest(hidden),'full_final_vocab_hash':digest(logits),
                  'public_token_count':len(keep)}
            return result
    model=CertifiedModel();model.freeze();model.eval();return model
