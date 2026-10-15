"""Isolated actual-audio ASR service; no reference transcript or private values."""
import argparse
import json
from pathlib import Path
import sys


def context(result, input_available_s, assistant_text, assistant_available_s):
    import math
    words=[word for segment in result['segments'] for word in segment.get('words',[])]
    groups=[];previous=0.
    for word in words:
        start,end=word['start'],word['end'];text=word['text'].strip()
        if (not text or not all(math.isfinite(v) for v in (start,end))
                or not 0<=start<end<=input_available_s+.04 or start<previous-.04):
            raise RuntimeError('ASR returned invalid or future word bounds')
        if not groups or start-groups[-1]['end_s']>.6:
            groups.append({'role':'user','text':text,'start_s':start,'end_s':end,
                'source':'actual arrived user PCM; Whisper timed recognition, not reference words'})
        else:
            groups[-1]['text']+=' '+text;groups[-1]['end_s']=end
        previous=end
    if not groups:raise RuntimeError('Native retrieval marker has no recognized user words')
    if assistant_text.strip():
        groups.append({'role':'assistant','text':assistant_text,'packet_available_s':assistant_available_s,
            'source':'sampled native text through actual retrieval trigger; not verified tool receipt'})
    # Each user span keeps ASR bounds. The assistant entry is the cumulative
    # sampled snapshot available at the trigger; do not fabricate word times.
    return groups


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args()
    import numpy as np
    import soundfile as sf
    import whisper_timestamped as whisper
    import torch
    import importlib.metadata
    import os
    from scipy.signal import resample_poly
    model=whisper.load_model('turbo',device='cuda')
    print(json.dumps({'ready':True,'model':'Whisper turbo','reference_transcript_supplied':False,
        'whisper':importlib.metadata.version('openai-whisper'),
        'whisper_timestamped':importlib.metadata.version('whisper-timestamped'),
        'torch':torch.__version__,'torch_file':torch.__file__,
        'assigned_visible_gpu':os.environ['CUDA_VISIBLE_DEVICES']}),flush=True)
    for line in sys.stdin:
        request=json.loads(line);path=Path(request['path']).resolve()
        if not path.is_relative_to(args.root.resolve()):raise RuntimeError('ASR input outside capture directory')
        pcm,rate=sf.read(path,dtype='float32')
        if rate!=24000 or pcm.ndim!=1 or not np.isfinite(pcm).all() or not 0<len(pcm)<=60*rate:
            raise RuntimeError('Invalid bounded actual user audio')
        result=whisper.transcribe(model,resample_poly(pcm,2,3).astype(np.float32),language='en',
            vad=False,beam_size=5,best_of=5,temperature=0.,verbose=None)
        print(json.dumps({'id':request['id'],'text':result['text'],'segments':result['segments'],
            'input_samples':len(pcm),'input_seconds':len(pcm)/rate}),flush=True)


if __name__=='__main__':main()
