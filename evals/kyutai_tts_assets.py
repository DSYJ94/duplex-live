"""Retrieve pinned assets for a separate streaming-TTS compatibility trial.

No inference, training, cloud resource or shared environment change occurs.
An asset pass is not a GPU, speech, duplex or controller qualification.
"""
import argparse
import json
from pathlib import Path
import subprocess
from evals.kyutai_tts_compat import MODEL_REVISION,VOICE_REVISION,WHEEL_SHA256,verified_overlay
from evals.moshi_model_download import sha256

VOICE_FILE='expresso/ex03-ex01_happy_001_channel1_334s.wav.1e68beda@240.safetensors'
DSM_COMMIT='4c4f65e147df056adf3346290d64c7b9649b18c9'


def main():
    from huggingface_hub import HfApi,hf_hub_download
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compat',type=Path,required=True)
    ap.add_argument('--reuse-mimi',type=Path,required=True)
    ap.add_argument('--vendor-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();compat,overlay=verified_overlay(args.compat)
    args.out.mkdir(parents=True,exist_ok=False)
    plans=[('kyutai/tts-1.6b-en_fr',MODEL_REVISION,[('config','config.json'),
        ('model','dsm_tts_1e68beda@240.safetensors'),
        ('mimi','tokenizer-e351c8d8-checkpoint125.safetensors'),
        ('tokenizer','tokenizer_spm_8k_en_fr_audio.model'),('model_card','README.md')]),
        ('kyutai/tts-voices',VOICE_REVISION,[('voice',VOICE_FILE),('voice_card','README.md')])]
    assets=[];api=HfApi(token=False)
    for repo,revision,files in plans:
        info=api.model_info(repo,revision=revision,files_metadata=True,token=False)
        if info.sha!=revision:raise RuntimeError('Upstream revision differs from the reviewed immutable pin')
        for role,name in files:
            meta=next(s for s in info.siblings if s.rfilename==name)
            lfs=meta.lfs
            expected=(lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)) if lfs else None
            if name.endswith('.safetensors') and (not expected or len(expected)!=64):
                raise RuntimeError('Weight/voice asset lacks its published LFS SHA256: '+name)
            reused=False
            if role=='mimi' and expected and args.reuse_mimi.is_file() and sha256(args.reuse_mimi)==expected:
                path=args.reuse_mimi;reused=True
            else:path=Path(hf_hub_download(repo,name,revision=revision,token=False))
            digest=sha256(path)
            if meta.size is not None and path.stat().st_size!=meta.size:
                raise RuntimeError('Published asset size mismatch: '+name)
            if expected and digest!=expected:raise RuntimeError('Published LFS content hash mismatch: '+name)
            if role=='config' and digest!=compat['config_sha256']:
                raise RuntimeError('Model configuration differs from the CPU compatibility probe')
            row=dict(role=role,repo=repo,revision=revision,filename=name,path=str(path),
                sha256=digest,bytes=path.stat().st_size,published_lfs_sha256=expected,reused_existing_mimi=reused)
            assets.append(row);print(json.dumps(row),flush=True)
    args.vendor_root.mkdir(parents=True,exist_ok=True)
    source=args.vendor_root/('delayed-streams-modeling-'+DSM_COMMIT)
    if not source.exists():
        subprocess.run(['git','init',str(source)],check=True)
        subprocess.run(['git','-C',str(source),'remote','add','origin','https://github.com/kyutai-labs/delayed-streams-modeling.git'],check=True)
        subprocess.run(['git','-C',str(source),'fetch','--depth=1','origin',DSM_COMMIT],check=True)
        subprocess.run(['git','-C',str(source),'checkout','--detach',DSM_COMMIT],check=True)
    if (subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=DSM_COMMIT
            or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)):
        raise RuntimeError('Changed upstream streaming source')
    report=dict(schema='dl-kyutai-tts-assets-v1',passed=True,assets=assets,
        model_revision=MODEL_REVISION,voice_revision=VOICE_REVISION,wheel_sha256=WHEEL_SHA256,
        compat_folder=str(args.compat),compat_report_sha256=sha256(args.compat/'report.json'),
        package_folder=str(overlay),code_sha256=sha256(Path(__file__)),
        dependency_versions=compat['package_versions'],torch=compat['torch'],
        streaming_source=str(source),streaming_source_commit=DSM_COMMIT,
        streaming_example_sha256=sha256(source/'scripts/tts_pytorch_streaming.py'),
        torch_within_upstream_requirement=compat['torch_within_upstream_requirement'],
        shared_environment_modified=False,gpu_compatibility_proven=False,
        voice_identity_is_accepted_reference=False,native_duplex_qualified=False,hybrid_system_qualified=False)
    (args.out/'assets.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
