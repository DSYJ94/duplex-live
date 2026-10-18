"""Inspect pinned public backchannel-detector assets without adopting the model.

This CPU inventory does not classify audio, train, import GUI packages or modify
shared dependencies. The publisher separates MIT code from model licenses; this
checkpoint has no model card/license metadata, so deployment approval is absent.
"""
import argparse
import json
from pathlib import Path
import subprocess
from evals.moshi_model_download import sha256

SOURCE_COMMIT='ec88ff4c4d3a527397ea3a225279d7f8db0e52e2'
MODEL_REVISION='982c561da312b26577593002fc1e32a046517535'
FILE='bc_det_mimi_state_dict_en_12.5hz_20000msec.pt'
SHA='0059087953a90494f1791503b3a4643b3e91be77f468387831b4e9d93599c2e0'


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--dl',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    source=args.dl/'vendor'/('MaAI-'+SOURCE_COMMIT)
    def git(*cmd):return subprocess.check_output(['git',*cmd],text=True).strip()
    if not source.exists():
        source.parent.mkdir(parents=True,exist_ok=True)
        git('clone','--no-checkout','https://github.com/MaAI-Kyoto/MaAI.git',str(source))
        git('-C',str(source),'checkout','--detach',SOURCE_COMMIT)
    if git('-C',str(source),'rev-parse','HEAD')!=SOURCE_COMMIT or git('-C',str(source),'status','--porcelain'):
        raise RuntimeError('Changed pinned acoustic-detector source')
    from huggingface_hub import hf_hub_download
    import torch
    path=Path(hf_hub_download('maai-kyoto/bc_det_en',FILE,revision=MODEL_REVISION,cache_dir=str(args.dl/'models/hf-cache/hub')))
    if path.stat().st_size!=17352706 or sha256(path)!=SHA:raise RuntimeError('Changed public detector weights')
    unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(path)
    if unsafe:raise RuntimeError('Detector inventory accepts tensor-only checkpoint; unsafe globals: '+str(unsafe))
    state=torch.load(path,map_location='cpu',weights_only=True)
    def inventory(value):
        if torch.is_tensor(value):return dict(shape=list(value.shape),dtype=str(value.dtype),finite=bool(torch.isfinite(value).all()))
        if isinstance(value,dict):return {str(k):inventory(v) for k,v in value.items()}
        if isinstance(value,(str,int,float,bool,type(None))):return value
        if isinstance(value,(list,tuple)):return [inventory(v) for v in value]
        raise RuntimeError('Unexpected non-tensor model metadata: '+str(type(value)))
    report=dict(schema='dl-maai-bcdet-inventory-v1',passed=True,source=str(source),source_commit=SOURCE_COMMIT,
        source_clean=True,model_revision=MODEL_REVISION,checkpoint=str(path),checkpoint_sha256=SHA,
        checkpoint_bytes=path.stat().st_size,unsafe_pickle_globals=unsafe,inventory=inventory(state),
        code_sha256=sha256(Path(__file__)),gpu_used=False,inference_executed=False,
        model_compatibility_proven=False,training_approved=False,production_approved=False,
        weight_license='not specified in publisher HF repository; source MIT is explicitly separate',
        source_license_sha256=sha256(source/'LICENSE'),publisher_readme_sha256=sha256(source/'README.md'))
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
