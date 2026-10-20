"""Complete current-recipe gradients and counterfactual Adam arithmetic; no updates."""
import json
import os
from pathlib import Path


def schedule(kinds, old_kinds, new_kinds):
    from evals.context_component_kind import balanced_index, retention_index
    from evals.context_native_teacher_objective import teacher_index
    from evals.context_native_teacher_coverage import complete_indices
    first={n:[0.]*size for n,size in [('actual',28),('reference',4),('old25',25),('new24',24),('audio49',49),('word',10),('clock',14)]}
    mean={n:[0.]*len(v) for n,v in first.items()}
    assert len(kinds)==28 and len(old_kinds)==25 and len(new_kinds)==24
    for step in range(8):
        ri=retention_index(kinds,step,4,True)
        chosen={'actual':[(balanced_index(kinds,step),1.)], 'reference':[(ri,1.)],
                'old25':[(teacher_index(kinds,step,ri,old_kinds),1.)],
                'word':[(i,64/10) for i in range(10)],'clock':[(i,64/14) for i in range(14)],
                'audio49':[(i,8/49) for i in range(step,49,8)]}
        indices=complete_indices(kinds,step,ri,new_kinds)
        chosen['new24']=[(i,1/len(indices)) for i in indices]
        for group,items in chosen.items():
            for i,w in items:
                mean[group][i]+=w/8
                if step==0:first[group][i]+=w
    return first,mean


def run(lm,learned,actual,retention,references,old,new,kinds,boundaries,audits,
        warm,clock,audio,policy,args,metadata,forward):
    import torch
    from torch.nn import functional as F
    from evals.moshi_training_smoke import tensor_hashes
    from evals.moshi_model_download import sha256
    from evals.personaplex_adapters import checkpoint_language_layers
    from evals.context_teacher_text_balance import loss as teacher_loss
    from evals.context_decision_loss import weights
    from evals.context_warm_ordinary_retention import loss as anchor_loss
    from evals.context_native_audio49_training import forward as audio_forward
    from evals.context_native_audio49_loss import loss as audio_loss
    from evals.context_native_gradient_audit import cosine
    assert args.native_audio49_preflight_only and not args.reload and not args.resume_training
    assert [len(x) for x in (actual,retention,old,new,warm,clock,audio)]==[28,4,25,24,10,14,49]
    before=tensor_hashes(lm.named_parameters());plans=schedule(kinds,[x[3]['kind'] for x in old],[x[3]['kind'] for x in new])
    width=sum(p.numel() for p in learned.values())
    vectors=[{g:torch.zeros(width) for g in plan} for plan in plans];rows=[]
    def measure(group,index,identity,make):
        for p in learned.values():p.grad=None
        with checkpoint_language_layers(lm):
            value=make();assert torch.isfinite(value);value.backward()
        pieces=[]
        for p in learned.values():
            if p.grad is not None:assert torch.isfinite(p.grad).all()
            pieces.append(torch.zeros(p.numel()) if p.grad is None else p.grad.detach().float().flatten().cpu())
        assert all(p.grad is None for n,p in lm.named_parameters() if n not in learned)
        vector=torch.cat(pieces)
        for plan,total in zip(plans,vectors):total[group].add_(vector,alpha=plan[group][index])
        row=dict(group=group,index=index,id=identity,loss=float(value.detach()),gradient_norm=float(vector.double().norm()))
        rows.append(row);print(json.dumps(row),flush=True)
    for i,(x,m,y) in enumerate(actual):
        def objective(x=x,m=m,y=y,i=i):
            z=forward(x)[1][0,0,m].float();target=y[m];w=weights(target,args.actual_response_first_word_weight,boundaries[i])
            return (F.cross_entropy(z,target,reduction='none')*w).sum()/w.sum()
        measure('actual',i,audits[i]['scenario'],objective)
    for i,(x,m,r) in enumerate(retention):
        def objective(x=x,m=m,i=i):
            h,z=forward(x);bh,prob=references[i]
            return F.kl_div(z[0,0,m].float().log_softmax(-1),prob,reduction='batchmean')+10*(h[:,m].float()-bh.float()).square().mean()
        measure('reference',i,r['id'],objective)
    for group,batches in [('old25',old),('new24',new)]:
        for i,(x,m,y,r) in enumerate(batches):measure(group,i,r['id'],lambda x=x,m=m,y=y:teacher_loss(forward(x)[1][0,0],y,m)[0])
    for group,batches in [('word',warm),('clock',clock)]:
        for i,(x,m,r,identity) in enumerate(batches):
            def objective(x=x,m=m,r=r):
                h,z=forward(x);return anchor_loss(h,z,m,r)[0]
            measure(group,i,identity,objective)
    for i,row in enumerate(audio):
        def objective(row=row):
            codes=row['codes'].cuda();return audio_loss(audio_forward(lm,codes),codes,row['audio_mask'].cuda())[0]
        measure('audio49',i,row['record']['teacher']['id'],objective)
    # Load, hash and validate the original state, but never install or mutate it.
    parent=args.warm_start_proof;report=json.loads((parent/'report.json').read_text());state_path=parent/'training-state.pt'
    assert sha256(state_path)==report['training_state_sha256']
    state=torch.load(state_path,map_location='cpu',weights_only=False)
    assert state['completed_steps']==520 and state['parameter_names']==list(learned)
    assert tensor_hashes(state['adapters'].items())==tensor_hashes((n,p.detach().cpu()) for n,p in learned.items())
    opt=state['optimizer'];assert len(opt['param_groups'])==1
    params=opt['param_groups'][0];ids=params['params'];assert len(ids)==len(learned)
    beta1,beta2=params['betas'];eps=params['eps'];analyses={}
    for horizon,total in zip(('first_update','eight_update_parent_frozen_mean'),vectors):
        gradient=sum(total.values());norm=float(gradient.double().norm());clipped=gradient*min(1.,1./(norm+1e-6))
        fresh=clipped/(clipped.square().sqrt()+eps);continued=[];offset=0;adam_steps=[]
        for key,(name,p) in zip(ids,learned.items()):
            s=opt['state'][key];n=p.numel();g=clipped[offset:offset+n].reshape(p.shape);offset+=n
            assert s['exp_avg'].shape==p.shape and s['exp_avg_sq'].shape==p.shape
            assert torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all() and (s['exp_avg_sq']>=0).all()
            t=int(s['step'])+1;adam_steps.append(t)
            m=beta1*s['exp_avg']+(1-beta1)*g;v=beta2*s['exp_avg_sq']+(1-beta2)*g.square()
            continued.append((m/(1-beta1**t)/(v.sqrt()/(1-beta2**t)**.5+eps)).flatten())
        old=torch.cat(continued)
        analyses[horizon]=dict(combined_gradient_norm=norm,group_gradient_norms={g:float(v.double().norm()) for g,v in total.items()},
            group_cosine_with_fresh_adam_descent={g:cosine(v,fresh) for g,v in total.items()},
            group_cosine_with_parent_adam_descent={g:cosine(v,old) for g,v in total.items()},
            pairwise_gradient_cosine={a+'__'+b:cosine(total[a],total[b]) for a in total for b in total if a<b},
            fresh_vs_parent_adam_direction_cosine=cosine(fresh,old),parent_adam_next_step_numbers=sorted(set(adam_steps)),
            interpretation='Positive gradient/direction cosine predicts local loss decrease under subtraction; this is frozen-parent first-order arithmetic, not native behavioral qualification or sequential training')
    for p in learned.values():p.grad=None
    assert before==tensor_hashes(lm.named_parameters()) and len(rows)==154
    result=dict(schema='dl-native-audio49-complete-gradient-probe-v1',source_commit=os.environ['DL_TASK_COMMIT'],metadata=metadata,policy=policy,
        parent_optimizer_state_sha256=sha256(state_path),all154_cases_measured=True,cases=rows,schedule_first=plans[0],schedule_mean8=plans[1],analyses=analyses,
        all_model_parameters_unchanged=True,optimizer_updates=0,optimizer_state_imported=False,new_checkpoint_written=False,native_behavior_evaluated=False,
        training_approved=False,candidate_adopted=False,diagnostic_only=True)
    (args.out/'audio49-gradient-probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(analyses),flush=True)
