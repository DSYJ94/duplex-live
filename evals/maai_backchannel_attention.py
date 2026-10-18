"""Explicit research-only equivalent vectorized ALiBi sliding causal mask.

The pinned publisher implementation constructs a full square mask and applies
its context bound with nested scalar assignments at every growing cache length.
This computes only the query rows actually requested. Weights, cache semantics,
relative bias, finite lower-triangle +1 and context limit remain unchanged.
"""
import argparse
import json
from pathlib import Path
import types
from evals.moshi_model_download import sha256


def mask_scores(self,qk,mask=None):
    import torch
    if mask is not None:raise RuntimeError('Vectorized diagnostic supports the model default mask only')
    if qk.dtype!=torch.float32:raise RuntimeError('Only the proved float32 detector attention is supported')
    keys=qk.shape[-1];queries=qk.shape[-2]
    k=torch.arange(keys,device=qk.device,dtype=qk.dtype)
    q=torch.arange(keys-queries,keys,device=qk.device)
    allowed=k.unsqueeze(0)<=q.unsqueeze(1)
    if self.context_limit>0:allowed=allowed & (k.unsqueeze(0)>=q.unsqueeze(1)-self.context_limit+1)
    bias=k.view(1,1,1,keys)*self.m.to(qk.device).view(1,self.num_heads,1,1)+1.
    return qk+bias.masked_fill(~allowed.view(1,1,queries,keys),float('-inf'))


def install(model):
    changed=[]
    for name,module in model.named_modules():
        if type(module).__name__=='MultiHeadAttentionAlibi':
            module.mask_scores=types.MethodType(mask_scores,module);changed.append(name)
    if not changed:raise RuntimeError('No pinned ALiBi attention modules found')
    return changed


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--inventory',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    import torch
    from evals.maai_backchannel_head import load_head
    model,load=load_head(args.inventory);torch.manual_seed(481516)
    module=next(m for m in model.modules() if type(m).__name__=='MultiHeadAttentionAlibi');original=module.mask_scores
    cases=[]
    with torch.inference_mode():
        for context in (8,250):
            module.context_limit=context
            for length,queries in ((1,1),(17,1),(17,17),(261,1),(261,7)):
                module.mask=None;qk=torch.randn(1,module.num_heads,queries,length)
                reference=original(qk.clone());candidate=mask_scores(module,qk.clone())
                exact=torch.equal(reference,candidate)
                if not exact:raise RuntimeError('Changed finite bias or causal/context mask')
                cases.append(dict(context=context,length=length,query_rows=queries,exact=True))
        module.context_limit=250;module.mask=None
        raw=torch.randn(2,37,512);features=torch.relu(model.decrease_dimension(raw));baseline=[];cache=None
        for t in range(features.shape[1]):
            result,cache=model(features[:1,t:t+1],features[1:,t:t+1],cache=cache);baseline.append(result['p_bc_det'])
        changed=install(model);candidate=[];cache=None
        for t in range(features.shape[1]):
            result,cache=model(features[:1,t:t+1],features[1:,t:t+1],cache=cache);candidate.append(result['p_bc_det'])
        error=float((torch.tensor(baseline)-torch.tensor(candidate)).abs().max())
        if error>1e-6:raise RuntimeError('Changed cached model outputs')
    report=dict(schema='dl-maai-vectorized-mask-proof-v1',passed=True,load=load,mask_cases=cases,
        cached_model_probability_max_error=error,changed_modules=changed,inventory_sha256=sha256(args.inventory),
        code_sha256=sha256(Path(__file__)),head_code_sha256=sha256(Path(__file__).with_name('maai_backchannel_head.py')),
        random_features_only=True,audio_behavior_qualified=False,production_approved=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
