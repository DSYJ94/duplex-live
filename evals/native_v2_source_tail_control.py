"""Read-only independent ASR of two rejected conversion question boundaries.

No expected transcript is supplied to ASR. Original source waveforms remain
unchanged; windows are diagnostics, never replacement training/evaluation data.
"""
import argparse
import json
from pathlib import Path
from evals.moshi_model_download import sha256

CASES = (
    ('native2-train-004', '3e395657148a0f98571a02b3c5d5c679273e89da991c6b8ea70d86ec48a7ed05',
     [('full',0.,None),('prefix',0.,4.2),('tail',4.5,None)]),
    ('native2-train-001', '6db9933bf2f6676738088981f097e195c83d6f5d6889bcd0fd667d062712e6be',
     [('tail',3.9,None)]),
)


def main():
    import numpy as np
    import soundfile as sf
    import whisper_timestamped as whisper
    from scipy.signal import resample_poly
    from evals.moshi_speech_audit import restore_clock
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    model=whisper.load_model('turbo',device='cpu');rows=[]
    for sid,digest,windows in CASES:
        wav=args.source/sid/'utterances/00_user.wav'
        if sha256(wav)!=digest:raise RuntimeError('Diagnosed source waveform changed')
        pcm,rate=sf.read(wav,dtype='float32')
        if rate!=24000 or pcm.ndim!=1 or not np.isfinite(pcm).all():raise RuntimeError('Invalid source')
        for name,start,end in windows:
            first=round(start*rate);last=len(pcm) if end is None else round(end*rate)
            audio=pcm[first:last]
            frame=480
            rms=[float(np.sqrt(np.mean(audio[i:i+frame]**2))) for i in range(0,len(audio),frame)]
            result=dict(text='',segments=[]) if max(rms,default=0.)<=.001 else whisper.transcribe(
                model,resample_poly(audio,2,3).astype(np.float32),language='en',vad=False,
                beam_size=5,best_of=5,temperature=0.,verbose=None)
            result=restore_clock(result,.8+first/rate)
            row=dict(id=sid,window=name,source_sha256=digest,start_s=.8+first/rate,
                end_s=.8+last/rate,maximum_20ms_rms=max(rms,default=0.),
                silent_at_existing_floor=max(rms,default=0.)<=.001,transcription=result,
                expected_words_supplied_to_asr=False,original_waveform_changed=False)
            rows.append(row);print(json.dumps(row),flush=True)
        if sha256(wav)!=digest:raise RuntimeError('Original changed during read-only control')
    (args.out/'report.json').write_text(json.dumps(dict(schema='dl-native-source-tail-control-v1',
        device='cpu',model='whisper turbo',results=rows,training_approved=False,optimizer_steps=0),indent=2)+'\n')


if __name__=='__main__':main()
