"""Independent waveform-region recognition without concatenating quiet gaps.

Diagnostic only: unchanged region detector, original samples, decoder and clock.
Each region is decoded independently, preventing a word from being aligned into
another utterance across a removed multi-second gap. No reference text is read.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path

from evals import native_reasoning_teacher_speech_v2 as previous

SCHEMA = 'dl-native-reasoning32-independent-burst-recognition-v1'


def restore(raw, start_sample, end_sample, rate, first_id):
    """Restore decoder clocks by the exact original sample offset, without edits."""
    result = deepcopy(raw)
    duration = (end_sample-start_sample)/rate
    offset = start_sample/rate
    if not 0 <= start_sample < end_sample or rate != 24000:
        raise ValueError('Explicit original sample region required')
    for i, segment in enumerate(result.get('segments', [])):
        segment['id'] = first_id+i
        for item in [segment, *segment.get('words', [])]:
            if not 0 <= item['start'] <= item['end'] <= duration+.02:
                raise ValueError('Independent clock escaped its original region')
            item['start'] += offset
            item['end'] += offset
    return result


def run(a):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    import whisper_timestamped as whisper
    plan = json.loads(a.plan.read_text())
    rows = plan['recordings']
    if (plan['schema'] != SCHEMA+'-word-free-plan' or len(rows) not in (2,32)
            or len({r['id'] for r in rows}) != len(rows)
            or any(set(r) != {'id','wav','wav_sha256'} for r in rows)
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_available()
            or previous.sha(a.weights) != previous.WEIGHT
            or {p:importlib.metadata.version(p) for p in previous.PACKAGES} != previous.PACKAGES):
        raise ValueError('Pinned CPU-only recognizer and exact word-free plan required')
    for row in rows:
        p = Path(row['wav'])
        if p.is_symlink() or previous.sha(p) != row['wav_sha256']:
            raise ValueError('Original waveform changed before recognition')
    a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    model = whisper.load_model(str(a.weights),device='cpu')
    state = dict(schema=SCHEMA,started_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256=previous.sha(Path(__file__)),plan_sha256=previous.sha(a.plan),
        region_policy=previous.POLICY,recognition='Each waveform-only region decoded separately; original clock restored by exact sample offset',
        asr_weight_sha256=previous.WEIGHT,asr_packages=previous.PACKAGES,
        expected=len(rows),results=[],phase='recognizing',provider_requests=0,
        original_audio_changed=False,reference_words_used=False,whole32_approved=False,
        semantic_approved=False,training_approved=False)
    previous.write(a.out/'recognition.json',state)
    for row in rows:
        pcm,rate=sf.read(row['wav'],dtype='float32')
        if rate != 24000:
            raise ValueError('Original24kHz waveform required')
        spans=previous.regions(pcm,rate)
        segments=[]; texts=[]; evidence=[]
        for i,(start,end) in enumerate(spans):
            first,last=round(start*rate),round(end*rate)
            raw=whisper.transcribe(model,resample_poly(pcm[first:last],2,3).astype(np.float32),
                language='en',vad=False,best_of=5,beam_size=5,temperature=0.,fp16=False,verbose=None)
            previous.write(a.out/(row['id']+'-region-%02d.json'%i),raw)
            restored=restore(raw,first,last,rate,len(segments))
            segments.extend(restored.get('segments',[]));texts.append(raw['text'])
            evidence.append(dict(first_sample=first,last_sample=last,
                raw_sha256=previous.sha(a.out/(row['id']+'-region-%02d.json'%i))))
        combined=dict(text=' '.join(texts),segments=segments)
        target=a.out/(row['id']+'-asr.json');previous.write(target,combined)
        if previous.sha(Path(row['wav'])) != row['wav_sha256']:
            raise ValueError('Original waveform changed during recognition')
        state['results'].append(dict(id=row['id'],wav_sha256=row['wav_sha256'],
            original_regions=spans,regions=evidence,asr_file=target.name,asr_sha256=previous.sha(target)))
        previous.write(a.out/'recognition.json',state)
        print(json.dumps(dict(id=row['id'],complete=len(state['results']),expected=len(rows))),flush=True)
    state.update(phase='complete_diagnostic_not_approval',finished_utc=datetime.now(timezone.utc).isoformat())
    previous.write(a.out/'recognition.json',state)
    return state


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('plan','weights','out'):
        p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
