"""Explicit optimizer-moment transplant for one new8 stage; no inherited RNG reset."""
import copy
import json
from pathlib import Path


def restore_moments(state,learned,optimizer,rate):
    import torch
    from training.control.train_context import assert_equal,cpu_tree
    if state['completed_steps']!=520 or state['parameter_names']!=list(learned) or len(state['optimizer']['param_groups'])!=1:
        raise ValueError('Exact named native520 Adam state required')
    if optimizer.state or rate!=2e-7:raise ValueError('Moment port requires a fresh optimizer shell at explicit small-step rate')
    original=state['optimizer'];group=original['param_groups'][0]
    if (len(group['params'])!=len(learned) or set(original['state'])!=set(group['params'])
            or group['lr']!=1e-5 or group['weight_decay']!=0. or group['betas']!=(.9,.999) or group['eps']!=1e-8):
        raise ValueError('Changed native520 optimizer recipe or incomplete state')
    for key,(name,p) in zip(group['params'],learned.items()):
        assert_equal(state['adapters'][name],p.detach().cpu())
        entry=original['state'][key]
        if set(entry)!={'step','exp_avg','exp_avg_sq'} or float(entry['step'])!=520:
            raise ValueError('Changed parent Adam fields or clock')
        for field in ('exp_avg','exp_avg_sq'):
            value=entry[field]
            if value.shape!=p.shape or value.dtype!=p.dtype or not torch.isfinite(value).all():raise ValueError('Invalid named Adam moment')
        if (entry['exp_avg_sq']<0).any():raise ValueError('Negative Adam second moment')
    expected=copy.deepcopy(original);expected['param_groups'][0]['lr']=rate
    optimizer.load_state_dict(expected)
    assert_equal(expected,cpu_tree(optimizer.state_dict()))
    return dict(source_completed_updates=520,source_adam_parameter_steps=[520],target_initial_adam_parameter_steps=[520],
        original_lr=1e-5,effective_lr=rate,all_named_moments_and_steps_exact=True,only_optimizer_group_change='lr',
        rng_imported=False,stage_update_counter_starts_at_zero=True,stage_update_counter_scope='Calls in this new corrective stage; Adam clocks remain520 until the first update')


def port(parent,learned,optimizer,rate,out):
    import torch
    import random,numpy as np
    from evals.moshi_model_download import sha256
    from evals.moshi_training_smoke import tensor_hashes
    parent=Path(parent);r=json.loads((parent/'report.json').read_text());p=parent/'training-state.pt'
    if r['optimizer_steps']!=520 or sha256(p)!=r['training_state_sha256']:raise ValueError('Unverified native520 source state')
    before=tensor_hashes(learned.items());cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all();python_rng=random.getstate();numpy_rng=np.random.get_state()
    state=torch.load(p,map_location='cpu',weights_only=True)
    result=restore_moments(state,learned,optimizer,rate)
    if tensor_hashes(learned.items())!=before or not torch.equal(cpu,torch.get_rng_state()) or any(not torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())):
        raise RuntimeError('Moment import changed adapter weights or current RNG')
    after_numpy=np.random.get_state()
    if python_rng!=random.getstate() or numpy_rng[0]!=after_numpy[0] or not np.array_equal(numpy_rng[1],after_numpy[1]) or numpy_rng[2:]!=after_numpy[2:]:raise RuntimeError('Moment import changed Python or NumPy RNG')
    result.update(parent_report_sha256=sha256(parent/'report.json'),parent_training_state_sha256=sha256(p),
        parent_checkpoint_sha256=r['checkpoint_sha256'],adapters_unchanged=True,current_rng_unchanged=True,optimizer_updates_during_import=0)
    (out/'parent-adam-import.json').write_text(json.dumps(result,indent=2)+'\n');return result


def validate_finished(proof,report):
    from evals.moshi_model_download import sha256
    p=Path(proof);audit=p/'parent-adam-import.json';a=json.loads(audit.read_text())
    if (sha256(audit)!=report['parent_adam_import_sha256'] or not a['all_named_moments_and_steps_exact']
            or not a['adapters_unchanged'] or not a['current_rng_unchanged'] or a['optimizer_updates_during_import']!=0
            or a['source_adam_parameter_steps']!=[520] or a['target_initial_adam_parameter_steps']!=[520]
            or a['effective_lr']!=2e-7 or a['only_optimizer_group_change']!='lr' or a['rng_imported']
            or a['parent_training_state_sha256']!=report['assistant_audio_auxiliary']['parent_adam_training_state_sha256']
            or report['adam_parameter_steps_after']!=[528] or report['warm_start_optimizer_reset']
            or report['optimizer_steps_scope']!='Updates in this corrective stage; preserved Adam clocks are recorded separately'):
        raise ValueError('Changed explicit parent Adam lineage')
    restored=json.loads((p/'reload-report.json').read_text())
    if restored.get('parent_adam_parameter_steps_restored')!=[528] or not restored['optimizer_and_rng_independently_compared_exact']:
        raise ValueError('Independent restore did not verify preserved Adam clocks')
    return a
