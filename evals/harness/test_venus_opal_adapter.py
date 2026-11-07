import pytest
from training.venus import opal_adapter as module


def test_defaults_cannot_start_a_paid_agent_and_wrapper_is_nonblocking(monkeypatch,tmp_path):
    calls=[]
    class Upstream:
        @staticmethod
        def wrap(optimizer,model,**kwargs):
            calls.append(kwargs);return 'wrapped'
    monkeypatch.setattr(module,'load_upstream',lambda source:Upstream)
    result=module.wrap(object(),object(),source=tmp_path,run_dir=tmp_path/'run',
        options=module.Options(total_steps=100,goal='Existing native objective'),clip_grad_norm=1.)
    assert result=='wrapped'
    args=calls[0]
    assert args['backend']=='keep' and args['blocking'] is False
    assert args['summarize'] is False and args['share_code'] is False
    assert args['clip_grad_norm']==1. and args['schedule']==[16,64,100]


@pytest.mark.parametrize('changes',[
    {'backend':'claude'}, {'backend':'codex'},
    {'backend':'mailbox:/explicit/path'},
    {'maximum_agent_cost_usd':float('nan')},
    {'unavailable_timeout_seconds':None},
])
def test_invalid_or_unfunded_agent_setup_is_rejected_before_import(changes):
    with pytest.raises((ValueError,TypeError)):
        module.Options(total_steps=100,goal='Native objective',**changes).validate()


def test_only_explicit_finite_aggregate_dev_metrics_reach_agent():
    class Optimizer:
        calls=[]
        def log(self,*args,**kwargs):self.calls.append((args,kwargs))
    opt=Optimizer()
    module.log_development(opt,{'dev_quiet_ce':.8},fold='DEV')
    assert opt.calls==[(({'dev_quiet_ce':.8},),{'show_agent':True})]
    for values,fold in [({'dev_quiet_ce':.8},'HOLDOUT'),({'test_accuracy':1.},'DEV'),
                        ({'answer_text':'hidden'},'DEV'),({'dev_ce':float('inf')},'DEV')]:
        with pytest.raises(ValueError):module.log_development(opt,values,fold=fold)
    assert len(opt.calls)==1


def test_checkpoint_manifest_rejects_an_adam_only_checkpoint():
    class Optimizer:
        def state_dict(self):return {'state':{},'param_groups':[]}
    with pytest.raises(ValueError,match='Complete OPAL'):module.checkpoint_manifest(Optimizer())


def test_explicit_zero_api_local_git_backend_preserves_finite_decision_bounds(monkeypatch,tmp_path):
    from training.venus import opal_supervisor
    sentinel=object();calls=[]
    monkeypatch.setattr(opal_supervisor,'local_backend',lambda root,repo,timeout:sentinel)
    class Upstream:
        @staticmethod
        def wrap(optimizer,model,**kwargs):calls.append(kwargs);return 'wrapped'
    monkeypatch.setattr(module,'load_upstream',lambda source:Upstream)
    options=module.Options(total_steps=84,goal='Full native goal',backend='local-git:'+str(tmp_path/'decisions'),local_git_repo=str(tmp_path/'repo'))
    module.wrap(object(),object(),source=tmp_path,run_dir=tmp_path/'run',options=options)
    assert calls[0]['backend'] is sentinel and calls[0]['max_cost_usd'] is None
    assert calls[0]['max_decisions']==3 and calls[0]['schedule']==[16,64,84]
    assert calls[0]['blocking'] is False and options.maximum_agent_cost_usd==0


@pytest.mark.parametrize('changes',[
    {'backend':'local-git:relative','local_git_repo':'/explicit'},
    {'backend':'local-git:/explicit'},
    {'backend':'local-git:/explicit','local_git_repo':'relative'},
    {'backend':'local-git:/explicit','local_git_repo':'/repo','maximum_agent_cost_usd':.01},
])
def test_local_backend_does_not_create_a_generic_paid_agent_bypass(changes):
    with pytest.raises(ValueError):module.Options(total_steps=84,goal='Native goal',**changes).validate()
