"""Free CPU-only proof of actual upstream OPAL, never a native model run.

Uses a tiny frozen linear layer plus LoRA. Proves 100-step keep parity, actual
generated-code swapping, and both program/Adam restores in distinct processes.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import random
import numpy as np
import torch
from .lora import LoRALinear
from .opal_adapter import Options, UPSTREAM_COMMIT, checkpoint_manifest, load_upstream, wrap
from .state import exact_state, rng_state, restore_rng


def probe():
    return dict(python=random.random(),numpy=float(np.random.rand()),torch=torch.rand(4).tolist())


def make(steps):
    torch.manual_seed(20261028)
    model=LoRALinear(torch.nn.Linear(4,2),rank=2,alpha=2)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    def multiplier(step):
        import math
        return (step+1)/16 if step<16 else .5*(1+math.cos(math.pi*(step-16)/max(1,steps-16)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,multiplier)
    return model,opt,scheduler


def step(model,opt,scheduler,wrapped):
    x=torch.tensor([[.5,1.,-1.,2.],[-.5,.25,1.,0.]])
    target=torch.tensor([[.2,-.1],[.4,.3]])
    opt.zero_grad(set_to_none=True)
    loss=(model(x)-target).square().mean();loss.backward()
    if wrapped:opt.step(loss=loss)
    else:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        opt.step()
    scheduler.step()
    return float(loss.detach())


def child(args):
    saved=torch.load(args.resume,map_location='cpu',weights_only=False)
    model,base,scheduler=make(saved['planned_steps'])
    model.load_state_dict(saved['model'])
    opt=wrap(base,model,source=args.source,run_dir=Path(args.output)/'restored-run',
        options=Options(total_steps=saved['planned_steps'],goal='Tiny offline LoRA integration; optimizer state equivalence only',
                        decision_updates=tuple(saved['decision_updates']),every=1),
        scheduler=scheduler,clip_grad_norm=1.)
    opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
    assert probe()==saved['expected_rng_probe'],'Independent RNG restore differs'
    restored=checkpoint_manifest(opt)
    assert restored==saved['manifest'],'Restored code/version/update differs before continuation'
    losses=[]
    for _ in range(saved['completed_steps'],saved['planned_steps']):
        losses.append(step(model,opt,scheduler,True))
    result=dict(model=model.state_dict(),adam=base.state_dict(),scheduler=scheduler.state_dict(),
                manifest=checkpoint_manifest(opt),losses=losses)
    opt.close();torch.save(result,Path(args.output)/'child-final.pt')


def run_case(source,out,*,rewrite=False):
    steps=24 if rewrite else 100
    model,base,scheduler=make(steps)
    options=Options(total_steps=steps,goal='Tiny offline LoRA integration; optimizer state equivalence only',
                    decision_updates=(4,) if rewrite else (16,64,96),every=1)
    if rewrite:
        opal=load_upstream(source)
        def fixture(workspace,message,system):
            path=workspace/'updater.py';text=path.read_text()
            marker='        for o in self.opts:\n            o.step()'
            assert text.count(marker)==1
            text=text.replace(marker,"        for group in self.param_groups:\n            group['weight_decay'] = .02\n"+marker)
            path.write_text(text)
            return 'Offline fixture changes weight decay; no LLM or paid service involved.'
        opt=opal.wrap(base,model,scheduler=scheduler,total_steps=steps,goal=options.goal,
            backend=fixture,schedule=[4],every=1,max_decisions=1,blocking=True,
            run_dir=str(out/'wrapped'),clip_grad_norm=1.,summarize=False,share_code=False,
            trial_steps=2,snapshot_limit_gb=.25)
    else:
        opt=wrap(base,model,source=source,run_dir=out/'wrapped',options=options,
                 scheduler=scheduler,clip_grad_norm=1.)
    losses=[];began=time.monotonic()
    for index in range(steps):
        losses.append(step(model,opt,scheduler,True))
        if index+1==steps//2:
            saved=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
                rng=rng_state(),manifest=checkpoint_manifest(opt),completed_steps=index+1,
                planned_steps=steps,decision_updates=list(options.decision_updates))
            saved['expected_rng_probe']=probe()
            torch.save(saved,out/'midpoint.pt')
    result=dict(model=copy.deepcopy(model.state_dict()),adam=copy.deepcopy(base.state_dict()),
                scheduler=copy.deepcopy(scheduler.state_dict()),manifest=checkpoint_manifest(opt),losses=losses)
    elapsed=time.monotonic()-began;opt.close()
    subprocess.run([sys.executable,'-m','training.venus.opal_smoke','--source',str(source),
        '--resume',str(out/'midpoint.pt'),'--output',str(out)],check=True,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},timeout=90)
    child_result=torch.load(out/'child-final.pt',map_location='cpu',weights_only=False)
    assert exact_state(result['model'],child_result['model'])
    assert exact_state(result['adam'],child_result['adam'])
    assert exact_state(result['scheduler'],child_result['scheduler'])
    assert result['manifest']==child_result['manifest']
    assert result['losses'][steps//2:]==child_result['losses']
    if rewrite:
        assert result['manifest']['program_version']>0
        assert result['adam']['param_groups'][0]['weight_decay']==.02
    else:
        plain,plain_opt,plain_scheduler=make(steps)
        plain_losses=[step(plain,plain_opt,plain_scheduler,False) for _ in range(steps)]
        assert plain_losses==losses
        assert exact_state(result['model'],plain.state_dict())
        assert exact_state(result['adam'],plain_opt.state_dict())
    return dict(steps=steps,actual_upstream_wrapper=True,code_rewritten=rewrite,
        code_writer='deterministic offline fixture' if rewrite else None,
        unwrapped_keep_loss_and_parameter_and_adam_exact=not rewrite,
        fresh_process_program_adam_rng_continuation_exact=True,
        final_program=result['manifest'],first_loss=losses[0],last_loss=losses[-1],
        wrapped_cpu_seconds=elapsed,api_calls=0,native_candidate_updates=0)


def main(args):
    if args.resume:return child(args)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    results={}
    for name,rewrite in [('keep100',False),('code_swap24',True)]:
        folder=out/name;folder.mkdir();results[name]=run_case(args.source,folder,rewrite=rewrite)
    report=dict(schema='dl-upstream-opal-offline-proof-v1',upstream_commit=UPSTREAM_COMMIT,
        torch=torch.__version__,device='cpu',training_runtime_modified=False,
        package_minimum_supported_torch='2.5; CPU smoke on2.4 is narrower evidence, not GPU support',
        tests=results,api_calls=0,gpu_seconds=0,native_model_training=False,
        passed=True)
    (out/'proof.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--resume');main(parser.parse_args())
