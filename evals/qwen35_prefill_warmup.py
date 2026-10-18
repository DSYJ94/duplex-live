"""Opt-in synthetic long-prefill warmup before any controller session.

Same frozen model, dynamic cache and math attention. This generates one unused
text token at each bounded length, never speech or a dispatched action. It does
not pad, edit, cache across or otherwise transform actual user requests.
"""
from pathlib import Path
import time
from evals.moshi_model_download import sha256

LENGTHS=(2048,4096)


def warm_long_prefill(executor):
    if executor.counter or executor.history or executor.warmup.get('long_prefill') is not None:raise ValueError('Long prefill warmup must precede actual controller requests and run once')
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    model=executor.model
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Warmup requires the completely frozen evaluated model')
    if getattr(model.generation_config,'cache_implementation',None) not in (None,'dynamic'):raise ValueError('Warmup preserves the verified dynamic cache')
    ids=executor.tokenizer('warm',add_special_tokens=False)['input_ids'];special=set(executor.tokenizer.all_special_ids)
    token=next((i for i in ids if i not in special),None)
    if token is None:raise ValueError('No ordinary synthetic warmup token')
    versions={n:p._version for n,p in model.named_parameters()};rows=[]
    with torch.inference_mode(),torch.cuda.device(model.device),sdpa_kernel(SDPBackend.MATH):
        for length in LENGTHS:
            inputs=torch.full((1,length),token,dtype=torch.long,device=model.device);mask=torch.ones_like(inputs);before=time.monotonic()
            result=model.generate(input_ids=inputs,attention_mask=mask,max_new_tokens=1,do_sample=False,use_cache=True,pad_token_id=executor.tokenizer.pad_token_id,eos_token_id=executor.tokenizer.eos_token_id)
            torch.cuda.synchronize(model.device)
            if result.shape!=(1,length+1) or not torch.equal(result[:,:length],inputs):raise ValueError('Synthetic warmup changed its input or failed to generate one token')
            rows.append(dict(input_tokens=length,generated_tokens=1,generated_token_id=int(result[0,-1]),seconds=time.monotonic()-before))
    if versions!={n:p._version for n,p in model.named_parameters()}:raise ValueError('Warmup changed a parameter version')
    if executor.counter or executor.history:raise ValueError('Synthetic warmup changed controller conversation state')
    proof=dict(schema='dl-synthetic-long-prefill-warmup-v1',lengths=list(LENGTHS),results=rows,synthetic_only=True,parameter_versions_unchanged=True,conversation_state_unchanged=True,user_requests_transformed=False,persistent_conversation_cache=False,actions_dispatched=False,speech_generated=False,training=False,code_sha256=sha256(Path(__file__)))
    executor.warmup=dict(original=executor.warmup,long_prefill=proof)
    return proof
