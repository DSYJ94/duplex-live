"""Migration defaults are explicit; every old optimizer leaf stays exact."""
import copy
import torch
from training.venus.isambard_native import adam_format
from training.venus.state import exact_state
from training.venus.native_optimizer import state_difference

def test_legacy_adam_and_scheduler_restore_without_changing_old_fields():
    p=torch.nn.Parameter(torch.ones(1));adam=torch.optim.AdamW([p]);scheduler=torch.optim.lr_scheduler.LambdaLR(adam,lambda _:1.)
    p.grad=torch.ones_like(p);adam.step();scheduler.step()
    a=copy.deepcopy(adam.state_dict());s=copy.deepcopy(scheduler.state_dict())
    a['param_groups'][0].pop('decoupled_weight_decay',None);s.pop('_is_initial',None)
    legacy=dict(optimizer=dict(user_optimizers=[a],user_schedulers=[s],updater=dict(opts=[a],scheds=[s])))
    original=copy.deepcopy(legacy);derived,changes=adam_format(legacy,torch)
    assert exact_state(original,legacy)
    fresh=torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))]);sched=torch.optim.lr_scheduler.LambdaLR(fresh,lambda _:1.)
    fresh.load_state_dict(derived['optimizer']['updater']['opts'][0]);sched.load_state_dict(derived['optimizer']['updater']['scheds'][0])
    assert exact_state(fresh.state_dict(),derived['optimizer']['user_optimizers'][0])
    assert exact_state(sched.state_dict(),derived['optimizer']['user_schedulers'][0])
    assert exact_state(a['state'],fresh.state_dict()['state'])
    assert all(c.endswith(('decoupled_weight_decay=true','_is_initial=false')) for c in changes)

def test_mismatch_reports_the_real_nested_field():
    assert state_difference({'scheds':[{}]},{'scheds':[{'_is_initial':False}]},'updater')=="updater/scheds/0 missing=[] extra=['_is_initial']"
