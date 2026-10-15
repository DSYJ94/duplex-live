"""Pin public Context Spanning assets for an isolated compatibility audition."""
import argparse
import json
import subprocess
from pathlib import Path
from evals.moshi_model_download import sha256

SOURCE_COMMIT='79df302fa2820cae10c61bb2b7f302c52edb6ed2'
MODEL_REVISION='e62a4494a295ff8cee345451cf40aae85719a9f5'
WEIGHT_SHA256='32a4dec8ea8feb28d20ce485f0e49a09b118413aaaebdd182379db41cd441fcc'
VOICE_SHA256='83aa916e3ad1b4a8edb38626d2697f6c6c12b25c7ca104bddb36e74e176d442c'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--vendor',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    from huggingface_hub import snapshot_download
    args.out.mkdir(exist_ok=False)
    if not args.vendor.exists():
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',
            'https://github.com/mindlogic-ai/Context-Spanning.git',str(args.vendor)],check=True)
        subprocess.run(['git','-C',str(args.vendor),'sparse-checkout','set',
            'contextspan','docs'],check=True)
        subprocess.run(['git','-C',str(args.vendor),'checkout','--detach',SOURCE_COMMIT],check=True)
    commit=subprocess.check_output(['git','-C',str(args.vendor),'rev-parse','HEAD'],text=True).strip()
    if commit!=SOURCE_COMMIT or subprocess.check_output(['git','-C',str(args.vendor),'status','--porcelain'],text=True):
        raise RuntimeError('Context Spanning source must be clean and pinned')
    path=Path(snapshot_download('mindlogicinc/context-spanning-7b',revision=MODEL_REVISION,
        allow_patterns=['context_spanning_7b.pt','voices/main.pt','README.md'],max_workers=2))
    files={str(p.relative_to(path)):{'sha256':sha256(p),'bytes':p.stat().st_size}
        for p in path.rglob('*') if p.is_file()}
    if not {'context_spanning_7b.pt','voices/main.pt','README.md'}<=set(files):
        raise RuntimeError('Incomplete public Context Spanning assets')
    if (files['context_spanning_7b.pt']['sha256']!=WEIGHT_SHA256
            or files['voices/main.pt']['sha256']!=VOICE_SHA256):
        raise RuntimeError('Context Spanning download differs from pinned upstream LFS digests')
    record={'schema':'dl-context-spanning-assets-v1','source':str(args.vendor),'source_commit':commit,
        'repo':'mindlogicinc/context-spanning-7b','revision':MODEL_REVISION,'path':str(path),'files':files,
        'inference_tested':False,'training_performed':False,'quality_qualified':False}
    (args.out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
