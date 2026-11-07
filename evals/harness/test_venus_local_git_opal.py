"""Real pinned OPAL zero-cost Git rewrite/restore on tiny CPU tensors only."""
import copy,json,os,subprocess,sys,time
from pathlib import Path
import pytest
import torch
from training.venus.native_optimizer import NativeOpal
from training.venus.opal_adapter import checkpoint_manifest
from evals.harness.test_venus_opal_supervisor import commit,wait_for,atomic

SOURCE=Path(os.environ.get('DUPLEX_TEST_OPAL_SOURCE','/private/tmp/duplex-opal-upstream-20261028'))


def one(runtime,model,step):
    runtime.base.zero_grad(set_to_none=True)
    loss=model(torch.tensor([[.2,-.4],[.8,.1]])).square().mean()
    loss.backward();runtime.step(loss,step)


@pytest.mark.skipif(not (SOURCE/'opal/__init__.py').is_file(),reason='Actual pinned OPAL required')
def test_real_zero_api_git_code_rewrite_nonblocking_training_and_fresh_process_restore(tmp_path):
    recipe=dict(updates=24,warmup_updates=4,learning_rate=.01,gradient_clip=1.)
    repo=tmp_path/'program-repo';commit(repo,{'training/venus/opal_programs/seed/updater.py':'seed=True\n'})
    torch.manual_seed(41);model=torch.nn.Linear(2,1)
    base=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01)
    runtime=NativeOpal(base,model,source=SOURCE,run_dir=tmp_path/'first',recipe=recipe,
        backend='local-git:'+str(tmp_path/'decisions'),local_git_repo=str(repo),agent_budget=0.)
    try:
        runtime.development({'ordinary':.5})
        for step in range(16):one(runtime,model,step)
        path=wait_for(lambda:next((tmp_path/'decisions').glob('*/request.json'),None))
        request=json.loads(path.read_text());identity=json.loads(path.with_name('identity.json').read_text())
        assert request['broker_api_calls']==0 and 'dev_ordinary_ce' in request['message']
        for step in (16,17):one(runtime,model,step)
        assert checkpoint_manifest(runtime.optimizer)['program_version']==0
        original=request['programs']['updater.py'];marker='        for o in self.opts:\n            o.step()'
        assert original.count(marker)==1
        changed=original.replace(marker,"        for group in self.param_groups:\n            group['weight_decay'] = .02\n"+marker)
        program='training/venus/opal_programs/fixture/updater.py';source=commit(repo,{program:changed})
        atomic(path.with_name('decision.json'),dict(**identity,action='adapt',source=source,programs={'updater.py':program},reason='Deterministic tiny integration fixture, not optimizer efficacy.'))
        wait_for(lambda:path.with_name('committed-decision.json').exists())
        wait_for(lambda:runtime.optimizer.worker.ready is not None)
        one(runtime,model,18)
        manifest=checkpoint_manifest(runtime.optimizer)
        assert manifest['program_version']==1 and manifest['update']==19 and base.param_groups[0]['weight_decay']==.02
        saved=dict(recipe=recipe,model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(runtime.optimizer.state_dict()),rng=torch.get_rng_state())
        torch.save(saved,tmp_path/'before.pt')
        one(runtime,model,19)
        torch.save(dict(model=model.state_dict(),optimizer=runtime.optimizer.state_dict()),tmp_path/'expected.pt')
        code='''import torch,sys
from pathlib import Path
from training.venus.native_optimizer import NativeOpal,assert_optimizer_equal
from training.venus.state import exact_state
root=Path(sys.argv[1]);source=Path(sys.argv[2]);repo=Path(sys.argv[3])
s=torch.load(root/'before.pt',weights_only=False);m=torch.nn.Linear(2,1);m.load_state_dict(s['model'])
b=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.01)
r=NativeOpal(b,m,source=source,run_dir=root/'fresh',recipe=s['recipe'],backend='local-git:'+str(root/'fresh-decisions'),local_git_repo=str(repo),agent_budget=0.)
try:
 r.optimizer.load_state_dict(s['optimizer']);torch.set_rng_state(s['rng'])
 assert_optimizer_equal(s['optimizer'],r.optimizer.state_dict())
 assert exact_state(m.state_dict(),s['model']) and torch.equal(torch.get_rng_state(),s['rng'])
 b.zero_grad(set_to_none=True);loss=m(torch.tensor([[.2,-.4],[.8,.1]])).square().mean();loss.backward();r.step(loss,19)
 expected=torch.load(root/'expected.pt',weights_only=False)
 assert exact_state(m.state_dict(),expected['model'])
 assert_optimizer_equal(expected['optimizer'],r.optimizer.state_dict())
 print('FRESH_PROCESS_ADAM_RNG_PROGRAM_AND_NEXT_UPDATE_EXACT')
finally:r.close()
'''
        completed=subprocess.run([sys.executable,'-B','-c',code,str(tmp_path),str(SOURCE),str(repo)],capture_output=True,text=True,timeout=90)
        assert completed.returncode==0,completed.stdout+completed.stderr
        assert 'FRESH_PROCESS_ADAM_RNG_PROGRAM_AND_NEXT_UPDATE_EXACT' in completed.stdout
        audit=json.loads(path.with_name('committed-decision.json').read_text())
        assert audit['source']==source and audit['cost_usd']==0 and audit['external_api_calls']==0
    finally:runtime.close()
