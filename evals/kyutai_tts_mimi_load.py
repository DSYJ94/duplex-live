"""Verify strict Mimi loading through Moshi's official legacy attention hook.

Safetensors0.6 validates raw names before PyTorch hooks. Use load_file plus
strict PyTorch loading, then compare every resulting tensor with its source.
No weight file, library package or shared environment is edited.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sys
from evals.moshi_model_download import sha256
from evals.kyutai_tts_compat import verified_overlay

MIMI_SHA256='09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50'


def verify_exact(model,source):
    import torch
    used=set();mapping={};count=0
    for name,value in model.state_dict().items():
        original=name
        if name in source:expected=source[name]
        else:
            match=re.fullmatch(r'(.*\.)(in_projs|out_projs)\.(\d+)\.weight',name)
            if match is None:raise RuntimeError('Unknown loaded Mimi tensor: '+name)
            prefix,kind,index=match.groups()
            names=('in_proj_weight','in_proj.weight') if kind=='in_projs' else ('out_proj.weight',)
            originals=[prefix+n for n in names if prefix+n in source]
            if len(originals)!=1:raise RuntimeError('Missing or ambiguous legacy Mimi projection')
            original=originals[0];module=model.get_submodule(prefix[:-1])
            expected=source[original].view(module.mult,-1,*source[original].shape[1:])[int(index)]
            mapping[name]=original
        if value.dtype!=expected.dtype or value.shape!=expected.shape or not torch.equal(value,expected.to(value.device)):
            raise RuntimeError('Loaded Mimi tensor differs from source: '+name)
        used.add(original);count+=1
    if used!=set(source):raise RuntimeError('Some source Mimi tensors were not loaded')
    return dict(all_tensors_exact=True,tensors=count,legacy_projection_mapping=mapping,
        source_tensors=len(source),source_tensors_unused=0,strict=True)


@contextmanager
def strict_mimi_loader(path,records):
    from moshi.models import loaders
    from safetensors.torch import load_file
    original=loaders.load_model
    def load(model,filename,strict=True,device='cpu'):
        if (not strict or type(model).__name__!='MimiModel' or Path(filename).resolve()!=path.resolve()
                or sha256(path)!=MIMI_SHA256):
            raise RuntimeError('Strict Mimi compatibility loader acquired an unexpected model or source')
        state=load_file(str(path),device=str(device))
        result=model.load_state_dict(state,strict=True)
        if result.missing_keys or result.unexpected_keys:raise RuntimeError('Incomplete strict Mimi load')
        record=verify_exact(model,state)
        if len(record['legacy_projection_mapping'])!=32:raise RuntimeError('Unexpected pinned Mimi legacy mapping')
        records.append(dict(record,checkpoint_sha256=MIMI_SHA256,official_load_hook_used=True,
            missing=[],unexpected=[],weight_file_modified=False))
        return [],[]
    loaders.load_model=load
    try:yield
    finally:loaders.load_model=original


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();assets=json.loads(args.assets.read_text())
    if not assets['passed']:raise RuntimeError('Unverified TTS assets')
    compat,overlay=verified_overlay(Path(assets['compat_folder']));sys.path.insert(0,str(overlay))
    import torch
    from safetensors.torch import load_file
    from moshi.models import loaders
    path=Path(next(r['path'] for r in assets['assets'] if r['role']=='mimi'))
    if sha256(path)!=MIMI_SHA256:raise RuntimeError('Changed pinned Mimi source')
    args.out.mkdir(parents=True,exist_ok=False)
    try:loaders.get_mimi(path,device='cpu',num_codebooks=32)
    except RuntimeError as error:
        default_error=str(error)
        if not all(x in default_error for x in ('Missing key(s)','in_projs.0.weight','Unexpected key(s)','in_proj_weight')):raise
    else:raise RuntimeError('Original raw-name failure did not reproduce; do not apply this compatibility path')
    records=[]
    with strict_mimi_loader(path,records):model=loaders.get_mimi(path,device='cpu',num_codebooks=32)
    state=load_file(str(path),device='cpu');negative=[]
    key=next(k for k,v in state.items() if v.dtype.is_floating_point and v.numel())
    changed=dict(state);changed[key]=state[key].clone();flat=changed[key].view(-1);flat[0]=0. if float(flat[0])!=0. else 1.
    missing=dict(state);missing.pop(key)
    extra=dict(state);extra['unrecognized_tensor']=torch.zeros(1)
    for name,bad in [('changed-source-value',changed),('missing-source-tensor',missing),('unused-source-tensor',extra)]:
        try:verify_exact(model,bad)
        except RuntimeError:negative.append(name)
        else:raise RuntimeError('Mimi integrity negative control failed: '+name)
    report=dict(schema='dl-kyutai-tts-mimi-load-cpu-v1',passed=True,
        original_default_loader_failed=True,original_error=default_error,
        records=records,negative_controls_rejected=negative,
        assets_manifest_sha256=sha256(args.assets),code_sha256=sha256(Path(__file__)),
        torch=torch.__version__,package_versions=compat['package_versions'],gpu_used=False,
        shared_environment_modified=False,weight_file_modified=False,gpu_execution_proven=False,
        native_duplex_qualified=False,hybrid_system_qualified=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
