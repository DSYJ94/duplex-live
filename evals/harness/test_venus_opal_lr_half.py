import copy
import torch
from training.venus.opal_programs.lr_half_v1.updater import Updater


def make():
    p=torch.nn.Parameter(torch.tensor([1.,2.]))
    opt=torch.optim.AdamW([p],lr=.01,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:1/(step+1))
    return p,opt,scheduler,Updater([('p',p)],spec=dict(user_optimizers=[opt],user_schedulers=[scheduler]))


def test_changes_only_effective_rate_without_compounding_or_losing_adam_state():
    p,opt,scheduler,updater=make()
    for step in range(3):
        p.grad=torch.tensor([.2,.1]);before=p.detach().clone()
        expected=torch.nn.Parameter(before.clone())
        reference=torch.optim.AdamW([expected],lr=.01)
        reference.load_state_dict(copy.deepcopy(opt.state_dict()))
        nominal=opt.param_groups[0]['lr']
        reference.param_groups[0]['lr']=nominal*.5
        expected.grad=p.grad.clone();reference.step()
        updater.step({'p':p.grad},step)
        assert torch.equal(p,expected)
        assert opt.param_groups[0]['lr']==nominal
        assert updater.last_effective_rates==[nominal*.5]
        assert opt.state[p]['step'].item()==step+1
        scheduler.step()


def test_full_program_state_restores_next_adam_update_exactly():
    p,opt,scheduler,u=make();p.grad=torch.tensor([.2,.1]);u.step({'p':p.grad},0);scheduler.step()
    state=copy.deepcopy(u.state_dict());q,o2,s2,u2=make()
    with torch.no_grad():q.copy_(p)
    u2.load_state_dict(state)
    assert u2.last_effective_rates==u.last_effective_rates and u2.last_update==u.last_update
    p.grad=torch.tensor([.4,.1]);q.grad=p.grad.clone()
    u.step({'p':p.grad},1);u2.step({'p':q.grad},1)
    assert torch.equal(p,q) and u.last_effective_rates==u2.last_effective_rates
    assert torch.equal(opt.state[p]['exp_avg'],o2.state[q]['exp_avg'])
    assert scheduler.state_dict()==s2.state_dict()
