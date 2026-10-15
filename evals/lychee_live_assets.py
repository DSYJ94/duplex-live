"""Pinned native streaming-vocoder assets for a concurrent component capture."""
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from evals.lychee_model_download import SOURCE_COMMIT
from evals.moshi_model_download import sha256


def load(assets_path, voice_path, provenance_path):
    import torch
    import torchaudio
    from evals.lychee_environment_check import configure_bf16
    configure_bf16()
    assets=json.loads(assets_path.read_text());source=Path(assets['source'])
    if (assets['source_commit']!=SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)):
        raise RuntimeError('Pinned native source changed')
    voice=json.loads(provenance_path.read_text())
    if (voice['schema']!='dl-voice-codec-probe-v1' or voice['source_model']!='gpt-live-1'
            or voice['reference_sha256']!=sha256(voice_path)):
        raise RuntimeError('Accepted synthetic reference changed')
    for key in ('vocoder','duplex'):
        model=assets['models'][key]
        # The tokenizer/config and all vocoder weights are independently pinned.
        for name,record in model['files'].items():
            if key=='duplex' and name.endswith('.safetensors'):continue
            if sha256(Path(model['path'])/name)!=record['sha256']:
                raise RuntimeError('Pinned streaming asset changed: '+name)
    sys.path[:0]=[str(source),str(source/'third_party/Step-Audio2')]
    from evals.minicpm_recovery_probe import read_reference_wav
    def wav_load(*a,**kw):
        pcm,rate=read_reference_wav(*a,**kw)
        return torch.from_numpy(pcm),rate
    torchaudio.load=wav_load
    from token2wav import Token2wav
    from transformers import AutoTokenizer
    from lychee_fd.runtime.vllm_generation import StreamingDecoder
    root=Path(assets['models']['vocoder']['path'])
    vocoder=Token2wav(str(root/'token2wav'))
    duplex=Path(assets['models']['duplex']['path'])
    config=json.loads((duplex/'config.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(duplex,local_files_only=True,trust_remote_code=True)
    fw=SimpleNamespace(text_pad_token_id=config['text_pad_token_id'],
        tts_pad_id=tokenizer(['<tts_pad>']).input_ids[0][0],
        stoken_delay_token_id=config['stoken_delay_token_id'],stoken_pad_token_id=config['stoken_pad_token_id'])
    return vocoder,StreamingDecoder(tokenizer,fw,tts_chunk_size=1),{
        'source_commit':SOURCE_COMMIT,'assets_manifest_sha256':sha256(assets_path),
        'voice_sha256':sha256(voice_path),'vocoder_weights_strict':True,
        'flow_implementation_sha256':sha256(Path(inspect.getfile(type(vocoder.flow))))}
