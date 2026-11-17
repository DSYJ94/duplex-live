"""Opt-in learned RET residual before native LMGen warmup/graph capture.

Inference validates the complete fit package; it does NOT replace the required
independent native Adam/RNG/OPAL restore or certify free-running behavior.
"""
import hashlib,json,re
import torch
from .native_action_readout import RetReadout,install


def identity(selection):
    if selection.get('step')!=196 or re.fullmatch('[0-9a-f]{64}',selection.get('checkpoint_sha256',''))is None or re.fullmatch('[0-9a-f]{40}',selection.get('trainer_source',''))is None:raise ValueError('Exact complete predecessor checkpoint identity required')
    return dict(base_revision='fdaf4090a61cb315c138a1faee287ffd6c716309',inference_family='context-spanning79df-personaplex16',checkpoint_sha256=selection['checkpoint_sha256'],trainer_source=selection['trainer_source'],predecessor_step=196)


def install_fit(lm,state,expected_identity):
    required={'metadata','step','delta','optimizer','rng','diagnostics','program','writer_pid'}
    if state.get('schema')!='dl-native-ret-fit-state-v1'or not required<=state.keys():raise ValueError('Complete learned readout/Adam/RNG/OPAL package required')
    meta=state['metadata'];mode=meta.get('mode');width=lm.text_linear.in_features
    if (meta.get('identity')!=expected_identity or mode not in ('readout','bias')or meta.get('input_features')!=width
        or meta.get('backbone_updates')!=0 or meta.get('previous_saved_updates_replayed')!=0
        or meta.get('trainable_parameters')!=(width+1 if mode=='readout'else 1)
        or state['step']!=128 or meta.get('recipe',{}).get('updates')!=128):raise ValueError('Exact final readout identity, ownership and progress required')
    optimizer=state['optimizer'];program=state['program'];keys={'opal','update','version','files','updater','user_optimizers','user_schedulers','observations','memory'}
    if (not keys<=optimizer.keys()or optimizer['opal']!=1 or optimizer['update']!=128 or program.get('update')!=128
        or program.get('upstream_commit')!='8b16759a3e9930916ed582e28068267859f935f8'
        or program.get('programs')!={k:hashlib.sha256(v.encode()).hexdigest()for k,v in optimizer['files'].items()}):raise ValueError('Complete matching saved OPAL programs and update clock required')
    learned=state['delta'];expected={'weight':(1,width),'bias':(1,)}
    if set(learned)!=set(expected)or any(not torch.is_tensor(v)or tuple(v.shape)!=expected[k]or v.dtype!=torch.float32 or not torch.isfinite(v).all()for k,v in learned.items()):raise ValueError('Complete finite FP32 learned residual required')
    if mode=='bias'and torch.count_nonzero(learned['weight']):raise ValueError('Scalar bias control cannot acquire hidden weights')
    if isinstance(lm.text_linear,RetReadout):raise ValueError('No stacking action heads')
    digest=hashlib.sha256(json.dumps(expected_identity,sort_keys=True).encode()).hexdigest()
    readout=install(lm,base_identity=digest);readout.restore(dict(schema='dl-native-ret-readout-v1',token=4,input_features=width,base_identity=digest,state=learned));lm.requires_grad_(False)
    return dict(mode=mode,new_readout_updates=128,new_trainable_coefficients=meta['trainable_parameters'],base_identity=digest,all_inference_weights_frozen=True,native_free_running_qualified=False)


def adapter_parameters(lm):
    """Audit unchanged old adapter values despite the explicit projection wrapper."""
    from evals.personaplex_adapters import adapter_parameters as named
    values=named(lm)
    if not isinstance(lm.text_linear,RetReadout):return values
    result={k.replace('text_linear.original.','text_linear.',1)if k.startswith('text_linear.original.')else k:v for k,v in values.items()}
    if len(result)!=len(values):raise ValueError('Old adapter names must remain bijective')
    return result


def install_zero(lm,expected_identity):
    """New zero-equivalent head, distinctly labelled as zero-update verification."""
    digest=hashlib.sha256(json.dumps(expected_identity,sort_keys=True).encode()).hexdigest()
    head=install(lm,base_identity=digest);lm.requires_grad_(False)
    return dict(mode='zero-insertion-proof',new_readout_updates=0,base_identity=digest,native_free_running_qualified=False)
