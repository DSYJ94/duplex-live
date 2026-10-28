import io
import json
from types import SimpleNamespace
import pytest
from training.venus import live_trial


def setup(monkeypatch,tmp_path,busy):
    clock=SimpleNamespace(now=0.)
    monkeypatch.setattr(live_trial.time,'time',lambda:clock.now)
    monkeypatch.setattr(live_trial.time,'sleep',lambda value:setattr(clock,'now',clock.now+value))
    calls=[];events=[]
    tunnel=SimpleNamespace(pid=123,poll=lambda:None,terminate=lambda:calls.append('terminated'),wait=lambda timeout:0,kill=lambda:None)
    monkeypatch.setattr(live_trial.subprocess,'Popen',lambda argv,**kwargs:(calls.append(argv) or tunnel))
    def health(*args,**kwargs):
        return io.StringIO(json.dumps(dict(ready=True,busy=busy(clock.now),optimizer_updates=244,
            candidate={'checkpoint_sha256':'c634e6cf38831210be0165ce7f6dd09c5454226f62ff622477fd4d051abde270'})))
    monkeypatch.setattr(live_trial,'urlopen',health)
    monkeypatch.setattr(live_trial,'websocket_probe',lambda url:dict(complete=True,fixture_only=True))
    c=SimpleNamespace(R=tmp_path,OUT='/approved-native',MODEL='/public-model',ssh_argv=['ssh','-o','StrictHostKeyChecking=yes','-o','ControlMaster=auto','-o','ControlPath=/old','owner@host'],
        guard=SimpleNamespace(poll=lambda:None),start_unit=lambda *args:calls.append(args),record=lambda event,**kwargs:events.append(event))
    return c,clock,calls,events


def test_live_trial_uses_loopback_and_stops_unused_after_five_minutes(monkeypatch,tmp_path):
    c,clock,calls,events=setup(monkeypatch,tmp_path,lambda now:False)
    live_trial.serve(c,'python',1000,'trial')
    assert 'native_microphone_ready' in events and events[-1]=='live_trial_unused_five_minutes'
    assert 300<clock.now<310 and calls[-1]=='terminated'
    tunnel=next(call for call in calls if isinstance(call,list))
    assert tunnel[tunnel.index('-L')+1]=='127.0.0.1:8037:127.0.0.1:8037'
    assert 'StrictHostKeyChecking=yes' in tunnel
    assert 'ControlMaster=no' in tunnel and 'ControlPath=none' in tunnel
    assert json.loads((tmp_path/'live-ready.json').read_text())['microphone_recorded'] is False


def test_live_trial_stops_after_conversation_disconnect(monkeypatch,tmp_path):
    c,clock,calls,events=setup(monkeypatch,tmp_path,lambda now:6<=now<=12)
    live_trial.serve(c,'python',1000,'trial')
    assert events[-1]=='live_trial_idle_after_conversation'
    assert 100<clock.now<110 and calls[-1]=='terminated'


def test_dead_guard_closes_tunnel_instead_of_leaving_paid_server_unwatched(monkeypatch,tmp_path):
    c,clock,calls,events=setup(monkeypatch,tmp_path,lambda now:False)
    c.guard.poll=lambda:1
    with pytest.raises(RuntimeError,match='guard ended'):live_trial.serve(c,'python',1000,'trial')
    assert calls[-1]=='terminated' and not events


def test_original_variant_matches_frozen_foundation_and_restores_candidate_even_on_error():
    import torch
    from training.venus.lora import install,selected_forward,LoRALinear
    from portable.venus.live_variant import variant_context,variant_profile
    from evals.harness.test_venus_training import model,example
    m=model();e=example(m)
    original=selected_forward(m,e,'cpu')[0].detach().clone()
    install(m,2,2,last_layers=1)
    with torch.no_grad():
        for module in m.modules():
            if isinstance(module,LoRALinear):module.b.add_(.5)
    m.requires_grad_(False)
    trained=selected_forward(m,e,'cpu')[0].detach().clone()
    assert not torch.equal(original,trained)
    with pytest.raises(RuntimeError):
        with variant_context(m,'released'):
            assert torch.equal(selected_forward(m,e,'cpu')[0].detach(),original)
            raise RuntimeError('Simulated disconnect')
    assert torch.equal(selected_forward(m,e,'cpu')[0].detach(),trained)
    assert not any(p.requires_grad for p in m.parameters())
    profile=dict(optimizer_updates=244,candidate={'experimental_candidate':True})
    assert variant_profile(profile,'released')['optimizer_updates']==0
    assert variant_profile(profile,'candidate')['optimizer_updates']==244
    assert profile['optimizer_updates']==244 and 'candidate' in profile
    with pytest.raises(ValueError):variant_context(m,'arbitrary-checkpoint')
