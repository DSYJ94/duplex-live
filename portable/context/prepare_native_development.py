"""Prepare this paired-evaluation container without changing the shared runtime.

Only public model bytes are copied to ephemeral local disk. All native originals
remain on the previously verified network volume. Recognition gets its own venv;
the NGC Torch build is constrained and checked rather than replaced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-commit',required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    source=Path(__file__).resolve().parents[2]
    if (subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=args.source_commit
            or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()):
        raise RuntimeError('Exact clean Git source required')
    if args.out.exists():raise RuntimeError('Preparation receipt already exists')
    started=time.monotonic()
    from evals.personaplex_train import MODEL_HASHES
    from evals.context_spanning_download import WEIGHT_SHA256,VOICE_SHA256
    original=Path('/workspace/duplex-live')
    root=Path('/opt/native10-runtime-root')
    if root.exists():raise RuntimeError('Do not overwrite existing native runtime')
    if shutil.disk_usage('/opt').free<40_000_000_000:
        raise RuntimeError('Insufficient ephemeral disk for public models')
    root.mkdir()
    for name in ('runs','vendor','worktrees','tmp','logs'):
        (root/name).symlink_to(original/name,target_is_directory=True)
    print(json.dumps({'phase':'copy_public_models','source':str(original/'models')}),flush=True)
    shutil.copytree(original/'models',root/'models',symlinks=True)
    for name in ('/projects/u6si/duplex-live','/lus/lfs1aip2/projects/u6si/duplex-live'):
        target=Path(name)
        if target.exists() or target.is_symlink():raise RuntimeError('Refuse existing native alias')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.symlink_to(root,target_is_directory=True)
    logical=Path('/projects/u6si/duplex-live')
    assets=json.loads((logical/'runs/context-spanning-assets-7454205/manifest.json').read_text())
    base=logical/'models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309'
    expected={str(base/name):value for name,value in MODEL_HASHES.items()}
    expected.update({str(Path(assets['path'])/name):r['sha256'] for name,r in assets['files'].items()})
    assert assets['files']['context_spanning_7b.pt']['sha256']==WEIGHT_SHA256
    assert assets['files']['voices/main.pt']['sha256']==VOICE_SHA256
    for path,wanted in expected.items():
        if digest(path)!=wanted:raise RuntimeError('Public model copy hash mismatch: '+path)
        print(json.dumps({'verified_public_model':path,'sha256':wanted}),flush=True)
    # No installation into /workspace/runtime-x86-ngc2510: that runtime may be
    # serving the other live audit. System NGC Torch is shared read-only here.
    asr=Path('/opt/native10-asr-env')
    subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(asr)],check=True)
    python=asr/'bin/python'
    version=subprocess.check_output([str(python),'-c','import torch;print(torch.__version__)'],text=True).strip()
    if version!='2.9.0a0+145a3a7bda.nv25.10':raise RuntimeError('Unexpected NGC Torch build')
    constraint=asr/'torch-constraint.txt';constraint.write_text('torch==='+version+'\n')
    subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check','--constraint',str(constraint),
                    'openai-whisper==20250625','whisper-timestamped==1.15.9',
                    'numpy==2.2.6','scipy==1.16.2','soundfile==0.13.1'],check=True)
    if subprocess.check_output([str(python),'-c','import torch;print(torch.__version__)'],text=True).strip()!=version:
        raise RuntimeError('ASR installation replaced pinned Torch')
    public_download=("import whisper,hashlib; from pathlib import Path; "
        "url=whisper._MODELS['turbo']; "
        "assert '/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/' in url; "
        "path=whisper._download(url,str(Path.home()/'.cache/whisper'),False); "
        "assert hashlib.sha256(Path(path).read_bytes()).hexdigest()=='aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a'; "
        "print('Pinned public Whisper turbo hash verified')")
    subprocess.run([str(python),'-c',public_download],check=True)
    receipt=dict(schema='dl-native-paired-container-preparation-v1',source_commit=args.source_commit,
                 original_archive_unchanged=True,public_model_hashes=expected,ephemeral_runtime=str(root),
                 public_copy_is_not_backup=True,shared_native_runtime_modified=False,
                 asr_python=str(python),torch=version,provider_credential_transferred=False,
                 seconds=time.monotonic()-started,training_or_behavior_approved=False)
    with args.out.open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
