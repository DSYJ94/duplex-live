"""Explicit BF16 weight merge for checked post-branch semantic adapters."""
import json
from evals.moshi_model_download import sha256


def plan(checkpoint_path, proof_path):
    import torch
    proof=json.loads(proof_path.read_text())
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if (checkpoint.get('schema')!='dl-lychee-semantic-adapter-v1' or checkpoint['rank']!=8
            or checkpoint['layers']!=[24,25,26,27] or checkpoint['scale']!=1.
            or not proof.get('passed') or proof['checkpoint_sha256']!=sha256(checkpoint_path)
            or proof['frozen_before_sha256']!=proof['frozen_after_sha256']
            or checkpoint['frozen_sha256']!=proof['frozen_after_sha256']
            or checkpoint['semantic_data_sha256']!=proof['data_sha256']
            or checkpoint['steps']!=proof['steps']
            or any(proof['zero_adapter_errors'].values())
            or any(proof['post_update_raw_branch_errors'].values())
            or any(proof['fresh_adapter_reload_errors'].values())):
        raise RuntimeError('Semantic checkpoint has no complete native implementation proof')
    projections={}
    for layer in range(24,28):
        for group,names in (('self_attn',('q_proj','k_proj','v_proj','o_proj')),
                ('mlp',('gate_proj','up_proj','down_proj'))):
            for name in names:
                stem=f'model.layers.{layer}.{group}.{name}'
                projections[stem+'.weight']=(checkpoint['state'][stem+'.a'],checkpoint['state'][stem+'.b'])
    expected={key[:-7]+'.'+suffix for key in projections for suffix in ('a','b')}
    if set(checkpoint['state'])!=expected:raise RuntimeError('Extra/missing semantic projection tensors')
    if any(not torch.isfinite(value).all() for pair in projections.values() for value in pair):
        raise RuntimeError('Nonfinite native semantic adapter')
    record={'checkpoint_sha256':sha256(checkpoint_path),'proof_sha256':sha256(proof_path),
        'base_assets_manifest_sha256':checkpoint['base_assets_manifest_sha256'],
        'merge':'FP32 B@A added to original checkpoint weight, then BF16 rounding',
        'expected_source_weights':sorted(projections),'applied_source_weights':{},
        'dynamic_adapter_to_merged_logit_parity_proven':False,'quality_qualified':False}
    return projections,record


def merge(name, value, projections, record):
    import torch
    if name not in projections:return value
    if name in record['applied_source_weights']:raise RuntimeError('Semantic source matrix merged twice')
    a,b=projections[name]
    if (value.dtype!=torch.bfloat16 or value.ndim!=2 or a.shape!=(8,value.shape[1])
            or b.shape!=(value.shape[0],8)):
        raise RuntimeError('Semantic projection does not match original source matrix')
    delta=b.to(value.device).float()@a.to(value.device).float()
    result=(value.float()+delta).to(value.dtype)
    if not torch.isfinite(result).all():raise RuntimeError('Nonfinite merged semantic weight')
    record['applied_source_weights'][name]={'shape':list(value.shape),
        'changed_bf16_elements':int((result!=value).sum()),
        'max_abs_applied_delta':float((result.float()-value.float()).abs().max())}
    return result
