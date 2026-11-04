"""One forward-only shape-confound probe; never changes the training path."""
from pathlib import Path
import torch
from .large_inputs import sha

ORIGINAL = Path('/workspace/migration/venus-training-v1/native-cache-v8/venus-control-v1-train-ordinary-0131.pt')
ORIGINAL_SHA = '0dddaf8b69f7b068b09de0427016346e0ce7cd251b29a7f9ce73977b04096907'


def whole_backbone_native_heads(llm, example, device):
    """Full causal backbone, with each original native head GEMM shape."""
    x=example['embeddings'].to(device).unsqueeze(0).detach().requires_grad_(True)
    positions=torch.arange(x.shape[1],device=device).unsqueeze(0)
    output=llm.model(inputs_embeds=x,position_ids=positions,use_cache=False,return_dict=True)
    wanted=set(int(p) for p in example['positions']);logits={};hidden={}
    for segment in example['segments']:
        start,n=segment['start'],segment['length'];last=start+n-1
        if last in wanted:
            h=output.last_hidden_state[:,start:start+n]
            logits[last]=llm.lm_head(h)[0,-1].float();hidden[last]=h[0,-1].float()
    if set(logits)!=wanted:raise ValueError('Every original labelled native position required')
    return torch.stack([logits[int(p)] for p in example['positions']]),torch.stack([hidden[int(p)] for p in example['positions']])


def probe(model):
    """Use only the exact historical failed21, never its changed v24 namesake."""
    base=dict(new_optimizer_updates=0,forward_only=True,changes_training_admission=False,
              original_sha256=ORIGINAL_SHA,whole_training_path_enabled=False)
    if not ORIGINAL.is_file():return dict(base,performed=False,reason='Original historical21 absent; do not substitute current304 example')
    if sha(ORIGINAL)!=ORIGINAL_SHA:return dict(base,performed=False,reason='Historical21 hash differs; do not substitute or relabel')
    example=torch.load(ORIGINAL,map_location='cpu',weights_only=False)
    if len(example['positions'])!=21 or len(example['logit_proofs'])!=21 or list(example['embeddings'].shape)!=[161,4096]:
        raise ValueError('Exact original known-failing21 numerical scene required')
    from .native_cache import training_input_proof
    example=dict(example,forward_path='whole-causal-native-head-shape-probe-v1')
    result=training_input_proof(model,example,forward_fn=whole_backbone_native_heads)
    return dict(base,performed=True,report=result,
                next_action='Future full-gradient/memory/restore/whole-corpus approval required' if result['passed'] else 'Retire this avenue for this phase; no kernel or precision sweep')
