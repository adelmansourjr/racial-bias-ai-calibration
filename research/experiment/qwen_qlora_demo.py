"""Small MLX quantized-base LoRA demo; evaluate on untouched applicants.

Uses affine 4-bit MLX quantization, not CUDA NF4/double-quantization QLoRA.
Only rank-4 q/v adapters in the final transformer block are optimized.
"""
import argparse
from pathlib import Path
import json
import os
import time
import numpy as np
from qwen_audit import metrics, probability

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")


def prompt_for(case, variant, task, tokenizer):
    if task == "shortlist":
        question = "Would you shortlist this applicant for the job based on the provided qualifications? A means shortlist, B means do not shortlist."
    else:
        question = (f"Does the RESUME text literally contain the phrase '{case['requirement_keyword']}'? "
                    "Use a case-insensitive exact substring match. Do not count its occurrence in the JOB or this question. "
                    "A means present, B means absent. This is a literal evidence check, not overall hiring suitability.")
    user = (f"JOB: {case['job_title']}\n{case['job_description']}\n\nRESUME:\n{variant['text']}\n\n"
            f"{question}\nReturn only A or B.")
    return tokenizer.apply_chat_template([
        {"role":"system", "content":"Assess the provided evidence carefully."},
        {"role":"user", "content":user}], tokenize=False, add_generation_prompt=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases',type=Path,default=Path(__file__).with_name('cases.json'))
    p.add_argument('--model',default='/Users/adelmansour/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
    p.add_argument('--steps',type=int,default=24)
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'results'/'qwen_qlora')
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx.utils import tree_flatten
    mx.random.seed(180); mx.set_cache_limit(128*1024*1024)
    started=time.perf_counter()
    cases=json.loads(args.cases.read_text())['cases']
    model,tok=load(args.model,lazy=True)
    for _,value in tree_flatten(model.parameters()): mx.eval(value)
    # Convert only in memory; existing BF16 checkpoint is untouched.
    nn.quantize(model,bits=4,group_size=64)
    mx.eval(model.parameters()); mx.clear_cache()
    a,b=tok.encode('A',add_special_tokens=False),tok.encode('B',add_special_tokens=False)
    assert len(a)==len(b)==1
    embedding=model.model.embed_tokens
    if not hasattr(embedding,'scales'):
        raise ValueError('Expected actual quantized embeddings')
    indices=mx.array([a[0],b[0]])
    rows=mx.dequantize(embedding.weight[indices],embedding.scales[indices],embedding.biases[indices],group_size=64,bits=4)
    contrast=(rows[0]-rows[1]).astype(mx.float32); mx.eval(contrast)
    contrast_np=np.array(contrast).astype(np.float64)
    tasks=['shortlist','evidence']
    prefixes=[]; input_cache={}; raw=[]; raw_mass=[]

    def run_prefix(ids):
        h=model.model.embed_tokens(mx.array(ids)[None])
        mask=create_attention_mask(h,None)
        for layer in model.model.layers[:-1]: h=layer(h,mask,None)
        return h,mask

    def final_from(prefix,mask):
        h=model.model.layers[-1](prefix,mask,None)
        return model.model.norm(h)[:,-1,:].astype(mx.float32)

    for ci,case in enumerate(cases):
        states=[]
        for ti,task in enumerate(tasks):
            task_states=[]; task_prefixes=[]; lengths=[]
            for vi,variant in enumerate(case['variants']):
                ids=tok.encode(prompt_for(case,variant,task,tok),add_special_tokens=False)
                input_cache[(ci,ti,vi)]=ids; lengths.append(len(ids))
                prefix,mask=run_prefix(ids)
                final=final_from(prefix,mask)
                mx.eval(final)
                task_states.append(np.array(final[0]))
                if case['split']=='calibration':
                    mx.eval(prefix); task_prefixes.append(np.array(prefix[0].astype(mx.float32)).copy())
                # Readout diagnostics show how much native mass A/B capture.
                full_logits=embedding.as_linear(final.astype(embedding.scales.dtype))[0].astype(mx.float32)
                log_norm=mx.logsumexp(full_logits)
                mass=mx.sum(mx.exp(full_logits[indices]-log_norm))
                top=mx.argmax(full_logits)
                mx.eval(mass,top)
                raw_mass.append({'case_id':case['case_id'],'task':task,'name':variant['name'],
                                 'allowed_A_B_mass':float(mass.item()),'native_top_token':tok.decode([int(top.item())])})
                del prefix,final,full_logits; mx.clear_cache()
            assert len(set(lengths))==1, 'Need equal name lengths for training batch alignment'
            states.append(np.stack(task_states))
            if task_prefixes:
                h=np.stack(task_prefixes)
                before_p=probability(states[-1],contrast_np)
                target=float(case['evidence_label']) if task=='evidence' else float(before_p.mean())
                prefixes.append((h,target,case['case_id'],task))
        raw.append(np.stack(states))
        print(f'Quantized baseline {ci+1}/{len(cases)}',flush=True)
    raw=np.stack(raw).astype(np.float64)
    model.freeze()
    adapter_config={'rank':4,'scale':8.0,'dropout':0.0,'keys':['self_attn.q_proj','self_attn.v_proj']}
    linear_to_lora_layers(model,1,adapter_config)
    trainable=tree_flatten(model.trainable_parameters())
    assert all('lora_a' in k or 'lora_b' in k for k,_ in trainable)
    trainable_count=sum(value.size for _,value in trainable)
    optimizer=optim.Adam(learning_rate=5e-4)

    def loss_fn(current,prefix,target):
        mask=create_attention_mask(prefix,None)
        hidden=current.model.norm(current.model.layers[-1](prefix,mask,None))[:,-1,:].astype(mx.float32)
        logits=hidden@contrast
        ce=mx.mean(mx.logaddexp(logits,mx.array(0.0))-target*logits)
        probs=mx.sigmoid(logits)
        consistency=mx.mean((probs-probs.mean())**2)
        deviations=hidden-hidden.mean(axis=0,keepdims=True)
        activation=mx.mean(mx.sum(deviations**2,axis=-1))/(mx.mean(mx.sum(hidden**2,axis=-1))+1e-8)
        return ce+5.0*consistency+10.0*activation

    value_and_grad=nn.value_and_grad(model,loss_fn)
    training=[]
    rng=np.random.default_rng(180)
    order=rng.permutation(len(prefixes))
    for step in range(args.steps):
        prefix,target,case_id,task=prefixes[int(order[step%len(order)])]
        batch=mx.array(prefix).astype(mx.bfloat16)
        loss,grads=value_and_grad(model,batch,mx.array(target))
        optimizer.update(model,grads)
        mx.eval(model.trainable_parameters(),optimizer.state,loss)
        training.append({'step':step+1,'loss':float(loss.item()),'case_id':case_id,'task':task})
        print(f'LoRA step {step+1}/{args.steps}, loss {float(loss.item()):.6f}',flush=True)
        del batch,grads; mx.clear_cache()
    model.eval()
    after=[]
    for ci,case in enumerate(cases):
        states=[]
        for ti,task in enumerate(tasks):
            task_states=[]
            for vi in range(len(case['variants'])):
                prefix,mask=run_prefix(input_cache[(ci,ti,vi)])
                h=final_from(prefix,mask); mx.eval(h)
                task_states.append(np.array(h[0])); del prefix,h; mx.clear_cache()
            states.append(np.stack(task_states))
        after.append(np.stack(states)); print(f'LoRA evaluation {ci+1}/{len(cases)}',flush=True)
    after=np.stack(after).astype(np.float64)
    test=np.array([c['split']=='test' for c in cases])
    labels=np.array([c['evidence_label'] for c in cases])
    groups=[v['cue_group'] for v in cases[0]['variants']]
    results={'model':args.model,'method':'MLX affine4-bit frozen-base rank4 LoRA on final q/v projections',
             'canonical_QLoRA_NF4':False,'steps':args.steps,'trainable_parameters':int(trainable_count),
             'adapter_config':adapter_config,'objective':'CE evidence labels / original quantized mean-score distillation +5 probability variance +10 normalized activation variance',
             'calibration_only_training':True,'test_count':int(test.sum()),'elapsed_seconds':time.perf_counter()-started,
             'methods':{},'training':training,'limits':['24 update demonstration, not converged fairness training',
             'A/B readout only; does not certify complete generated answers','Literal evidence labels are not actual hiring outcomes',
             'Compare to same4-bit base, not BF16 baseline, to separate quantization effects']}
    for name,states in [('quantized_base',raw),('quantized_base_plus_lora',after)]:
        result={}
        for ti,task in enumerate(tasks):
            stats,_=metrics(states[test,ti],probability(states[test,ti],contrast_np),groups,labels[test] if task=='evidence' else None)
            result[task]=stats
        results['methods'][name]=result
    mx.save_safetensors(str(args.out/'adapters.safetensors'),dict(tree_flatten(model.trainable_parameters())))
    (args.out/'adapter_config.json').write_text(json.dumps({'fine_tune_type':'lora','num_layers':1,'lora_parameters':adapter_config},indent=2))
    (args.out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    (args.out/'readout-diagnostics.json').write_text(json.dumps(raw_mass,indent=2)+'\n')
    np.savez_compressed(args.out/'activations.npz',before=raw.astype(np.float32),after=after.astype(np.float32),contrast=contrast_np.astype(np.float32))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__': main()
