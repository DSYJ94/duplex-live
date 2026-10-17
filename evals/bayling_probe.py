"""Unchanged BayLing offline speech audition; no live or delegation acceptance."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from evals.moshi_model_download import sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','runtime','data','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--scenario',required=True)
    args=ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from evals.recovery_data import validate_conversion
    assets=json.loads(args.assets.read_text());runtime=json.loads(args.runtime.read_text())
    if (assets['schema']!='dl-bayling-assets-v1' or runtime['schema']!='dl-bayling-runtime-v1'
            or not runtime['imports_passed'] or assets['training_performed'] is not False
            or assets['pins_sha256']!=sha256(Path(__file__).with_name('assets')/'bayling-audition-pins.json')
            or runtime['requirements_sha256']!=sha256(Path(__file__).with_name('assets')/'bayling-runtime-requirements.txt')):
        raise RuntimeError('Unverified BayLing assets or runtime')
    vendor=Path(assets['source'])
    if (subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()!=assets['source_commit']
            or subprocess.check_output(['git','-C',str(vendor),'status','--porcelain'],text=True)):
        raise RuntimeError('Changed baseline source')
    models={m['repo']:Path(m['path']) for m in assets['models']}
    for model in assets['models']:
        for name,expected in model['files'].items():
            if sha256(Path(model['path'])/name)!=expected['sha256']:
                raise RuntimeError('Changed BayLing asset: '+name)
    # PyTorch's default weights-only loader remains enabled for both decoder
    # state dicts; no checkpoint globals are allowlisted by this audition.
    decoder=models['zai-org/glm-4-voice-decoder']
    for name in ('flow.pt','hift.pt'):
        if torch.serialization.get_unsafe_globals_in_checkpoint(decoder/name):
            raise RuntimeError('Decoder checkpoint needs unsupported pickle globals: '+name)
    source=next(r for r in validate_conversion(args.data)['accepted'] if r['scenario_id']==args.scenario)
    if source['split']!='validation' or source['kind'] not in ('simple','clarify'):
        raise RuntimeError('Unchanged audition is limited to approved ordinary/clarification validation stimuli')
    pcm,rate=sf.read(source['path'],dtype='float32',always_2d=True)
    if pcm.shape[1]!=2 or rate!=24000 or not np.isfinite(pcm).all():
        raise RuntimeError('Invalid approved user recording')
    args.out.mkdir(exist_ok=False)
    user_file=args.out/'user.wav';sf.write(user_file,pcm[:,1],rate,subtype='FLOAT')
    sys.path.insert(0,str(vendor))
    from transformers import AutoModel
    from bayling_duplex_runtime.speech_tokenizer.modeling_whisper import WhisperVQEncoder
    from bayling_duplex import BayLingDuplex
    loading={}
    def checked_loader(original,name):
        def load(*a,**kw):
            kw['output_loading_info']=True
            model,info=original(*a,**kw)
            loading[name]=info
            if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
                (args.out/'loading-failure.json').write_text(json.dumps(loading,indent=2,default=str)+'\n')
                raise RuntimeError('Released '+name+' weights do not load exactly')
            return model
        return load
    AutoModel.from_pretrained=staticmethod(checked_loader(AutoModel.from_pretrained,'language_model'))
    WhisperVQEncoder.from_pretrained=staticmethod(checked_loader(WhisperVQEncoder.from_pretrained,'speech_tokenizer'))
    random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
    started=time.monotonic()
    model=BayLingDuplex(model_path=models['BayLing-Models/BayLing-Duplex'],
        speech_tokenizer_path=models['zai-org/glm-4-voice-tokenizer'],decoder_path=decoder,
        interleave_ratio='10:5:10',device='cuda',torch_dtype='bfloat16')
    loaded=time.monotonic();torch.cuda.synchronize()
    result=model.generate(str(user_file),max_duration=40.,temperature=.6,top_p=.8,
        silence_penalty=0.,max_epad_count=2 if source['kind']=='clarify' else 1)
    torch.cuda.synchronize();generated=time.monotonic()
    result.save_json(args.out/'generation.json',include_tokens=True)
    if result.response_audio_tokens:
        model.save_audio(result.audio_tokens,args.out/'model.wav')
    torch.cuda.synchronize()
    record={'schema':'dl-bayling-offline-audition-v1','scenario':args.scenario,'kind':source['kind'],
        'source_split':source['split'],'source_audio_sha256':source['wav_sha256'],
        'source_sidecar_sha256':source['sidecar_sha256'],'user_sha256':sha256(user_file),
        'assets_manifest_sha256':sha256(args.assets),'runtime_manifest_sha256':sha256(args.runtime),
        'source_commit':assets['source_commit'],'loading':loading,'seed':17,'text':result.text,
        'input_audio_preencoded':True,'input_wall_paced':False,'audio_decoded_after_generation':True,
        'load_seconds':loaded-started,'generation_seconds':generated-loaded,
        'decode_seconds':time.monotonic()-generated,'audio_exists':(args.out/'model.wav').exists(),
        'model_parameters_updated':False,'backend_connected':False,'voice_reference_changed':False,
        'voice_mode':'released default BayLing voice','live_streaming_qualified':False,
        'delegation_qualified':False,'quality_qualified':False,'training_approved':False}
    if record['audio_exists']:record['audio_sha256']=sha256(args.out/'model.wav')
    (args.out/'report.json').write_text(json.dumps(record,indent=2,default=str)+'\n')
    print(json.dumps(record,default=str),flush=True)


if __name__=='__main__':main()
