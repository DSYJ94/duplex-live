"""Run the actual native trainer step on tiny CPU Qwen, including a fresh process.

This is an optimizer integration proof, not native9B/GPU or behavior approval.
"""
import argparse,copy,json,os,subprocess,sys
from pathlib import Path
import pytest
import torch
from evals.harness.test_venus_native_opal import setup,make,SOURCE
from training.venus import train
from training.venus.native_optimizer import NativeOpal,assert_optimizer_equal
from training.venus.lora import adapter_state
from training.venus.state import save,load,rng_state,restore_rng,exact_state


def run(a):
    out=Path(a.output)
    if not a.child:out.mkdir(exist_ok=False,parents=True)
    with pytest.MonkeyPatch.context() as patches:
        recipe=setup(patches);m,base,rows=make()
        # Establish actual inherited Adam moments before OPAL wrapping.
        train.train_step(m,base,None,rows,0)
        runtime=NativeOpal(base,m,source=SOURCE,run_dir=out/('child-opal' if a.child else 'parent-opal'),recipe=recipe)
        meta=dict(source='actual-native-train-step-tiny-CPU-proof',recipe=recipe)
        if a.child:
            assert load(out/'step1.pt',m,runtime.optimizer,meta)==1
            probe=dict(torch=torch.rand(4).tolist())
            result=train.train_step(m,runtime.optimizer,None,rows,1,opal=runtime)
            torch.save(dict(weights=adapter_state(m),optimizer=runtime.optimizer.state_dict(),result=result,probe=probe,pid=os.getpid()),out/'child.pt')
            runtime.close();return
        runtime.development({'ordinary':1.2,'quiet':.8})
        first=train.train_step(m,runtime.optimizer,None,rows,0,opal=runtime)
        save(out/'step1.pt',m,runtime.optimizer,1,meta)
        probe=dict(torch=torch.rand(4).tolist())
        second=train.train_step(m,runtime.optimizer,None,rows,1,opal=runtime)
        expected=copy.deepcopy(runtime.optimizer.state_dict());weights=adapter_state(m)
        runtime.close()
    subprocess.run([sys.executable,'-m','evals.venus_native_opal_proof','--output',str(out),'--child'],check=True,timeout=90,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    child=torch.load(out/'child.pt',map_location='cpu',weights_only=False)
    assert child['pid']!=os.getpid() and probe==child['probe']
    assert exact_state(weights,child['weights'])
    assert_optimizer_equal(expected,child['optimizer'])
    assert all(second[k]==child['result'][k] for k in ('ce','kl','hidden','gradient_norm','lr','objective'))
    report=dict(passed=True,parent_pid=os.getpid(),child_pid=child['pid'],actual_native_train_step=True,device='cpu',model='tiny Qwen3 fixture',actual_fresh_process=True,adapter_adam_scheduler_program_observations_rng_equal=True,native9b_gpu_qualified=False,api_calls=0,gpu_seconds=0,candidate_updates=0)
    (out/'proof.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--child',action='store_true');run(p.parse_args())
