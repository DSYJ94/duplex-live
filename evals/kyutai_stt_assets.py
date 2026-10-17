"""Verify pinned streaming-ASR assets for a separate listening component.

No training, inference, private data export or shared environment modification.
"""
import argparse
import json
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.kyutai_tts_compat import verified_overlay
from evals.kyutai_tts_mimi_load import MIMI_SHA256

STT_REVISION='095e38f6242006a93c2541149b181988397f5c7c'


def main():
    from huggingface_hub import HfApi,hf_hub_download
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compat',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();compat,overlay=verified_overlay(args.compat)
    args.out.mkdir(parents=True,exist_ok=False)
    repo='kyutai/stt-1b-en_fr-candle';info=HfApi(token=False).model_info(repo,revision=STT_REVISION,files_metadata=True,token=False)
    if info.sha!=STT_REVISION:raise RuntimeError('Changed streaming-ASR revision')
    assets=[]
    for role,name in [('config','config.json'),('model','model.safetensors'),
            ('mimi','mimi-pytorch-e351c8d8@125.safetensors'),
            ('tokenizer','tokenizer_en_fr_audio_8000.model'),('model_card','README.md')]:
        meta=next(s for s in info.siblings if s.rfilename==name);lfs=meta.lfs
        expected=(lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)) if lfs else None
        if name.endswith('.safetensors') and (not expected or len(expected)!=64):
            raise RuntimeError('Streaming-ASR weight lacks its published SHA256')
        path=Path(hf_hub_download(repo,name,revision=STT_REVISION,token=False));digest=sha256(path)
        if (expected and expected!=digest) or (meta.size is not None and path.stat().st_size!=meta.size):
            raise RuntimeError('Streaming-ASR asset hash/size mismatch')
        if role=='mimi' and digest!=MIMI_SHA256:raise RuntimeError('ASR Mimi differs from its exact CPU load proof')
        row=dict(role=role,path=str(path),filename=name,revision=STT_REVISION,
            sha256=digest,bytes=path.stat().st_size,published_lfs_sha256=expected)
        assets.append(row);print(json.dumps(row),flush=True)
    config=json.loads(Path(next(r['path'] for r in assets if r['role']=='config')).read_text())
    if (config['model_type']!='stt' or config['n_q']!=32 or config['dep_q']!=0
            or config['delays']!=[0]*33 or config['extra_heads_num_heads']!=4 or config['extra_heads_dim']!=6
            or config['stt_config']!={'audio_delay_seconds':.5,'audio_silence_prefix_seconds':0.}):
        raise RuntimeError('Unexpected pinned streaming-ASR layout or delay')
    report=dict(schema='dl-kyutai-stt-assets-v1',passed=True,revision=STT_REVISION,assets=assets,
        compat_folder=str(args.compat),compat_report_sha256=sha256(args.compat/'report.json'),
        code_sha256=sha256(Path(__file__)),audio_delay_seconds=.5,audio_silence_prefix_seconds=0.,
        package_folder=str(overlay),package_versions=compat['package_versions'],
        shared_environment_modified=False,gpu_compatibility_proven=False,duplex_system_qualified=False)
    (args.out/'assets.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
