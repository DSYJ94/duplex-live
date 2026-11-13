"""Fresh-process two-rank CPU restore of real OPAL and checkpoint envelopes."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from evals.harness.test_distributed_native_build import ToyNative
from training.personaplex.distributed_native import DistributedNative
from training.personaplex.distributed_opal import DistributedOpal
from training.personaplex import distributed_state
from training.venus.state import rng_state, restore_rng, exact_state
from training.venus.native_optimizer import assert_optimizer_equal


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['save','restore'],required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    dist.init_process_group('gloo');active=None
    try:
        rank=dist.get_rank();torch.manual_seed(1717)
        runtime=DistributedNative(ToyNative(),'broad-lora64','cpu')
        base=torch.optim.AdamW(runtime.parameters(),lr=1e-3)
        recipe=dict(updates=32,warmup_updates=16,learning_rate=1e-3,gradient_clip=1.)
        folder=a.root/(a.mode+'-rank-'+str(rank));folder.mkdir(parents=True,exist_ok=False)
        active=DistributedOpal(runtime,base,source=Path('/projects/u6ada/duplex-live/opal-upstream-v1'),run_dir=folder,recipe=recipe)
        identity=dict(proof='cpu-fixture-only',recipe=recipe,source=os.environ['DUPLEX_PROOF_SOURCE'])
        path=a.root/'checkpoint.pt';x=torch.full((1,2,4),float(rank+1))
        if a.mode=='save':
            loss=runtime.view.forward_codes(x)[1].square().mean();loss.backward();active.step(float(loss.detach()),0)
            with torch.no_grad(): reference=runtime.view.forward_codes(x)[1].clone()
            state=active.rank_state(rng_state());state.update(reference=reference,pid=os.getpid())
            distributed_state.save(path,runtime,base,identity=identity,step=1,rank_state=state)
        else:
            state=torch.load(path,map_location='cpu',weights_only=False);proof={}
            def consume(saved):
                if saved['pid']==os.getpid():raise ValueError('Distinct process required')
                active.active.optimizer.load_state_dict(saved['opal'])
                active.active.restore_diagnostics(saved['diagnostics']);restore_rng(saved['rng'])
                assert_optimizer_equal(saved['opal'],active.active.optimizer.state_dict())
                assert exact_state(saved['rng'],rng_state())
                with torch.no_grad(): torch.testing.assert_close(runtime.view.forward_codes(x)[1],saved['reference'],rtol=0,atol=0)
                proof.update(passed=True,rank=rank,save_pid=saved['pid'],restore_pid=os.getpid(),complete_OPAL_Adam_RNG_exact=True,logits_exact=True,native_weights_loaded=False)
            step=distributed_state.restore(state,runtime,base,identity=identity,restore_rank_state=consume,checkpoint_directory=a.root)
            assert step==1
            (a.root/('restore-rank-'+str(rank)+'.json')).write_text(json.dumps(proof,indent=2)+'\n')
            print(json.dumps(proof),flush=True)
    finally:
        if active is not None:active.close()
        dist.destroy_process_group()


if __name__=='__main__':main()
