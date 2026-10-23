"""Read-only CPU comparison of two saved native adapter/Adam checkpoints.

Hash-bound artifacts, not another model run. Effective LoRA matrix norms use
rank-sized Gram matrices; no full7B load or dense32000x4096 text-head allocation.
These measurements establish neither causality nor behavioral qualification.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def product_norm(b,a):
    """Frobenius norm of B@A in FP64 without constructing the dense product."""
    b=b.double();a=a.double()
    squared=float(((b.T@b)*(a@a.T).T).sum())
    if squared<0 or not math.isfinite(squared):raise ArithmeticError('Invalid finite Gram-product norm')
    return math.sqrt(squared)


def effective_delta(old_a,old_b,new_a,new_b,scaling):
    import torch
    if (old_a.shape!=new_a.shape or old_b.shape!=new_b.shape or old_a.ndim!=2 or old_b.ndim!=2
            or old_a.shape[0]!=old_b.shape[1] or not math.isfinite(scaling) or scaling<=0
            or any(not torch.isfinite(t).all() for t in (old_a,old_b,new_a,new_b))):
        raise ValueError('Finite matching low-rank factors and positive saved scale required')
    a=old_a.double();b=old_b.double();da=new_a.double()-a;db=new_b.double()-b
    # Bnew Anew - Bold Aold = dB Anew + Bold dA. This factored
    # difference avoids cancellation between two large almost-equal norms.
    c=torch.cat((db,b),dim=1);d=torch.cat((new_a.double(),da),dim=0)
    before=scaling*product_norm(b,a);after=scaling*product_norm(new_b,new_a)
    delta=scaling*product_norm(c,d)
    dot=scaling**2*float(((b.T@c)*(d@a.T).T).sum())
    return dict(parent_effective_LoRA_frobenius=before,candidate_effective_LoRA_frobenius=after,
                delta_effective_LoRA_frobenius=delta,
                relative_to_parent_LoRA_component=delta/before if before else None,
                parent_component_delta_cosine=dot/(before*delta) if before and delta else None,
                relative_to_frozen_base_weight_not_measured=True,dense_product_materialized=False)


def load(proof):
    import torch
    from safetensors.torch import load_file
    r=json.loads((proof/'report.json').read_text())
    for name,key in [('adapters.safetensors','checkpoint_sha256'),('training-state.pt','training_state_sha256')]:
        if sha(proof/name)!=r[key]:raise ValueError('Saved proof hash mismatch: '+name)
    tensors=load_file(str(proof/'adapters.safetensors'),device='cpu')
    s=torch.load(proof/'training-state.pt',map_location='cpu',weights_only=True)
    if (len(tensors)!=26 or sum(t.numel() for t in tensors.values())!=1894400
            or set(s['parameter_names'])!=set(tensors) or set(s['adapters'])!=set(tensors)
            or any(not torch.equal(t,s['adapters'][n]) or not torch.isfinite(t).all() for n,t in tensors.items())
            or s['optimizer_class']!='torch.optim.adamw.AdamW' or len(s['optimizer']['param_groups'])!=1):
        raise ValueError('Exact complete finite adapter and matching saved optimizer ownership required')
    g=s['optimizer']['param_groups'][0]
    if len(g['params'])!=26 or len(s['optimizer']['state'])!=26:raise ValueError('Every adapter needs its saved Adam state')
    moments={name:s['optimizer']['state'][pid] for name,pid in zip(s['parameter_names'],g['params'])}
    for name,m in moments.items():
        if (any(m[k].shape!=tensors[name].shape or not torch.isfinite(m[k]).all() for k in ('exp_avg','exp_avg_sq'))
                or not math.isfinite(float(m['step']))):raise ValueError('Missing/nonfinite saved Adam moment')
    identity=dict(proof=str(proof),report_sha256=sha(proof/'report.json'),
                  checkpoint_sha256=r['checkpoint_sha256'],training_state_sha256=r['training_state_sha256'],
                  saved_completed_steps=s['completed_steps'],Adam_parameter_steps=sorted({float(m['step']) for m in moments.values()}),
                  saved_optimizer_lr=g['lr'],betas=list(g['betas']),eps=g['eps'],weight_decay=g['weight_decay'],
                  rank=s['metadata']['recipe']['rank'],scaling=s['metadata']['recipe']['scaling'],
                  last_layers=s['metadata']['recipe']['last_layers'],source_commit=r['repo_commit'])
    return tensors,s,moments,identity


def audit(parent,candidate):
    import torch
    torch.set_num_threads(4)
    old,ps,pm,pi=load(parent);new,cs,cm,ci=load(candidate)
    if (set(old)!=set(new) or pi['rank']!=8 or ci['rank']!=8 or pi['scaling']!=ci['scaling']
            or pi['betas']!=ci['betas'] or ps['parameter_names']!=cs['parameter_names']):
        raise ValueError('Same adapter topology and aligned inherited Adam ownership required')
    steps={float(cm[n]['step'])-float(pm[n]['step']) for n in old}
    if len(steps)!=1 or next(iter(steps))<=0:raise ValueError('A consistent positive saved Adam-clock difference required')
    count=next(iter(steps));beta1,beta2=pi['betas'];old_m_coef=beta1**count;old_v_coef=beta2**count
    rows=[];modules=[]
    for name in sorted(old):
        a=old[name].double();b=new[name].double();delta=b-a;before=float(a.norm());dn=float(delta.norm())
        inherited=old_m_coef*pm[name]['exp_avg'].double();added=cm[name]['exp_avg'].double()-inherited
        inherited_v=old_v_coef*pm[name]['exp_avg_sq'].double();added_v=cm[name]['exp_avg_sq'].double()-inherited_v
        rows.append(dict(name=name,shape=list(a.shape),elements=a.numel(),dtype=str(old[name].dtype),
            parent_parameter_norm=before,candidate_parameter_norm=float(b.norm()),delta_norm=dn,
            relative_parameter_delta=dn/before if before else None,maximum_absolute_delta=float(delta.abs().max()),
            changed_elements=int(torch.count_nonzero(delta)),inherited_first_moment_contribution_norm=float(inherited.norm()),
            new_steps_first_moment_residual_norm=float(added.norm()),
            inherited_second_moment_contribution_norm=float(inherited_v.norm()),
            new_steps_second_moment_residual_norm=float(added_v.norm()),
            moment_residual_includes_saved_FP32_rounding=True))
    for name in sorted(n.removesuffix('.lora_A.weight') for n in old if n.endswith('.lora_A.weight')):
        modules.append(dict(module=name,**effective_delta(old[name+'.lora_A.weight'],old[name+'.lora_B.weight'],
                   new[name+'.lora_A.weight'],new[name+'.lora_B.weight'],pi['scaling'])))
    global_parent=math.sqrt(sum(r['parent_parameter_norm']**2 for r in rows));global_delta=math.sqrt(sum(r['delta_norm']**2 for r in rows))
    return dict(schema='dl-native-adapter-saved-delta-audit-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256=sha(Path(__file__)),parent=pi,candidate=ci,trainable_parameters=1894400,tensors=26,modules=13,
        changed_tensors=sum(r['changed_elements']>0 for r in rows),changed_elements=sum(r['changed_elements'] for r in rows),
        global_parent_parameter_norm=global_parent,global_parameter_delta_norm=global_delta,
        global_relative_parameter_delta=global_delta/global_parent,actual_Adam_step_difference=count,
        inherited_first_moment_coefficient=old_m_coef,inherited_second_moment_coefficient=old_v_coef,
        moment_coefficients_are_not_percentages_of_update_influence=True,
        saved_frozen_parameter_hash_records_equal=ps['metadata']['frozen_hashes']==cs['metadata']['frozen_hashes'],
        full_frozen_base_weights_rehashed_by_this_audit=False,
        parameter_rows=rows,effective_module_rows=modules,weights_modified=False,new_optimizer_updates=0,
        GPU_used=False,API_calls=0,behavior_reproduced=False,loss_group_causality_established=False,
        candidate_adopted=False,scope='Saved-file numerical measurements only; not a native generation or convergence test')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('parent','candidate','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();r=audit(a.parent,a.candidate)
    with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:r[k] for k in ('changed_tensors','global_relative_parameter_delta','actual_Adam_step_difference','inherited_first_moment_coefficient','inherited_second_moment_coefficient')}))
