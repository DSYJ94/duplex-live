"""Strict frozen MaAI detector-head compatibility proof, without audio features.

Import only pinned model modules, not the desktop package entry point. Register
all checkpoint tensors, including the shared rate-conversion parameters unused
at native12.5Hz. This is not encoder parity, audio behavior or adoption approval.
"""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types
from evals.moshi_model_download import sha256
from evals.maai_backchannel_assets import SOURCE_COMMIT,MODEL_REVISION,SHA


def load_head(inventory_path,device='cpu'):
    import torch
    from torch import nn
    proof=json.loads(inventory_path.read_text());source=Path(proof['source']);checkpoint=Path(proof['checkpoint'])
    if (proof['schema']!='dl-maai-bcdet-inventory-v1' or not proof['passed']
            or proof['source_commit']!=SOURCE_COMMIT or proof['model_revision']!=MODEL_REVISION
            or proof['checkpoint_sha256']!=SHA or sha256(checkpoint)!=SHA
            or proof['code_sha256']!=sha256(Path(__file__).with_name('maai_backchannel_assets.py'))
            or proof['unsafe_pickle_globals']):raise RuntimeError('Missing matching tensor-only asset inventory')
    def git(*args):return subprocess.check_output(['git','-C',str(source),*args],text=True).strip()
    if git('rev-parse','HEAD')!=SOURCE_COMMIT or git('status','--porcelain'):raise RuntimeError('Changed detector source')
    package='_dl_pinned_maai'
    if package not in sys.modules:
        namespace=types.ModuleType(package);namespace.__path__=[str(source/'src/maai')];sys.modules[package]=namespace
    elif Path(sys.modules[package].__path__[0])!=source/'src/maai':raise RuntimeError('Conflicting detector namespace')
    conf_class=importlib.import_module(package+'.models.config').VapConfig
    model_class=importlib.import_module(package+'.models.bc_det').BcDetGPT
    conf=conf_class(frame_hz=12.5,encoder_type='mimi',context_limit=250)
    model=model_class(conf)
    # The distributed checkpoint owns a shared encoder-conversion layer; no
    # frozen Mimi weights are in this checkpoint. No missing tensors skipped.
    model.encoder=nn.Module();model.encoder.frame_rate_conv=nn.Conv1d(512,512,3,bias=True)
    model.decrease_dimension=nn.Linear(512,256)
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    loaded=model.load_state_dict(weights,strict=True);state=model.state_dict()
    if loaded.missing_keys or loaded.unexpected_keys or set(state)!=set(weights):raise RuntimeError('Incomplete detector head')
    if any(not torch.equal(state[k],v) for k,v in weights.items()):raise RuntimeError('Head tensors do not match original weights')
    model.eval().requires_grad_(False).to(device)
    return model,dict(strict=True,all_tensors_exact=True,tensor_count=len(state),checkpoint_sha256=SHA,
        source_commit=SOURCE_COMMIT,continuous_feature_size=512,head_size=256,context_frames=250,
        frame_rate_conversion_registered_but_unused_at_native_rate=True,gui_entrypoint_imported=False,
        frozen_mimi_encoder_not_in_head_checkpoint=True,publisher_feature_parity_proven=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--inventory',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    model,load=load_head(args.inventory);torch.manual_seed(481516)
    with torch.inference_mode():
        raw=torch.randn(2,17,512);features=torch.relu(model.decrease_dimension(raw))
        full,_=model(features[:1],features[1:],return_all_frames=True)
        stream=[];cache=None
        for t in range(17):
            result,cache=model(features[:1,t:t+1],features[1:,t:t+1],cache=cache);stream.append(result)
        full_values=np.array([r['p_bc_det'] for r in full]);stream_values=np.array([r['p_bc_det'] for r in stream])
        error=float(np.max(np.abs(full_values-stream_values)))
        changed=features.clone();changed[:,9:]+=50.
        future,_=model(changed[:1],changed[1:],return_all_frames=True)
        future_error=float(np.max(np.abs(full_values[:9]-np.array([r['p_bc_det'] for r in future[:9]]))))
        # Current-input negative: a model that ignores either channel must fail.
        effects=[]
        for channel in (0,1):
            perturbed=features.clone();perturbed[channel,0]+=20.
            result,_=model(perturbed[:1],perturbed[1:],return_all_frames=True)
            effects.append(float(np.max(np.abs(full_values-np.array([r['p_bc_det'] for r in result])))))
    if error>1e-5 or future_error>1e-6 or min(effects)<1e-6:
        raise RuntimeError('Detector head failed declared cached/causal/input-consumption checks: '+str((error,future_error,effects)))
    report=dict(schema='dl-maai-bcdet-head-cpu-v1',passed=True,load=load,
        cached_vs_full_max_probability_error=error,future_perturbation_prefix_error=future_error,
        channel_perturbation_effects=effects,inventory_sha256=sha256(args.inventory),code_sha256=sha256(Path(__file__)),
        inputs_are_synthetic_random_features=True,audio_inference_executed=False,gpu_used=False,
        publisher_encoder_parity_proven=False,backchannel_behavior_qualified=False,production_approved=False,training_approved=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
