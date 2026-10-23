"""Whole reasoning32 source words/waveforms, using independently pinned ASR.

Forty unique original WAVs bind all56 utterances/all32 scenes. This new source
schema permits a long question only within the existing88-second/1100-frame
episode bound; actual prefix, spans, result and lifecycle timing still must fit.
No legacy source cap, quality threshold, inference or training rule changes.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path

from data.synthetic.filter import norm_words,wer
from evals.native_reasoning_source import verify as source_verify,requests_for,sha
from evals.native_control_teacher import atomic_json

ASR_WEIGHT='aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a'
SCHEMA='dl-native-reasoning32-independent-user-source-quality-v1'
WER_BAR=.05
PEAK_BAR=.995


def assess(text,result,pcm,rate):
    import numpy as np
    reasons=[];duration=len(pcm)/rate
    words=[w for segment in result.get('segments',[]) for w in segment.get('words',[])]
    actual=result.get('text','');error=wer(norm_words(text),norm_words(actual))
    if not words or error>WER_BAR:reasons.append('Independent exact-word agreement failed')
    if [v for w in words for v in norm_words(w['text'])]!=norm_words(actual):
        reasons.append('Word intervals do not cover the independent transcript')
    previous=0.
    for w in words:
        start=w.get('start');end=w.get('end')
        if (type(start) not in (int,float) or type(end) not in (int,float)
                or not math.isfinite(start) or not math.isfinite(end)
                or not previous<=start<end<=duration):
            reasons.append('Invalid independent word interval');break
        previous=end
    peak=float(np.abs(pcm).max()) if len(pcm) else 0.
    if (rate!=24000 or pcm.ndim!=1 or not .1<=duration<=88
            or not np.isfinite(pcm).all() or not .001<peak<=PEAK_BAR):
        reasons.append('Empty, invalid, clipped or over-bound original source waveform')
    active=[i for i in range(0,len(pcm),480) if float(np.sqrt(np.mean(pcm[i:i+480]**2)))>.001]
    onset=active[0]/rate if active else None
    if onset is None or (words and isinstance(words[0].get('start'),(int,float)) and words[0]['start']<onset-.2):
        reasons.append('Word onset predates detectable original voice beyond200ms')
    return dict(passed=not reasons,reasons=reasons,raw_wer=error,authored_text=text,asr_text=actual,
                measured_words=words,duration_s=duration,peak=peak,first_active_s=onset,
                reference_text_prompt_used=False,waveform_changed=False)


def audit(source,weights,out,threads=4):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import whisper_timestamped as whisper
    import torch
    manifest,plan=source_verify(source);objects,_=requests_for(plan)
    if sha(weights)!=ASR_WEIGHT:raise ValueError('Exact previously pinned independent Whisper turbo required')
    if not 1<=threads<=8:raise ValueError('Bounded local independent recognition threads required')
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    code=sha(Path(__file__));model=whisper.load_model(str(weights),device='cpu');results=[]
    for row in manifest['objects']:
        identity=row['object_id'];wav=source/row['wav'];pcm,rate=sf.read(wav,dtype='float32')
        # Recognition receives only an independent16kHz copy; original provider
        # WAV bytes stay unchanged and are rehashed after all recognition calls.
        if rate!=24000 or pcm.ndim!=1:raise ValueError('Original provider must supply24kHz mono; never repair input silently')
        result=whisper.transcribe(model,resample_poly(pcm,2,3).astype(np.float32),language='en',
            vad=False,best_of=5,beam_size=5,temperature=0.,fp16=False,verbose=None)
        name=identity+'-asr.json';atomic_json(out/name,result)
        verdict=assess(objects[identity]['input'],result,pcm,rate)
        results.append(dict(object_id=identity,wav_sha256=sha(wav),asr_file=name,
                            asr_sha256=sha(out/name),**verdict))
        atomic_json(out/'progress.json',dict(complete_objects=len(results),expected=40,
                    latest_passed=verdict['passed'],training_approved=False))
        print(json.dumps(dict(source_object=len(results),expected=40,passed=verdict['passed'])),flush=True)
    source_verify(source)
    if code!=sha(Path(__file__)):raise ValueError('Source audit implementation changed')
    r=dict(schema=SCHEMA,at_utc=datetime.now(timezone.utc).isoformat(),source_manifest_sha256=sha(source/'source-audio.json'),
        plan_sha256=manifest['plan_sha256'],code_sha256=code,source_code_sha256=manifest['code_sha256'],
        asr_model='whisper-turbo',asr_weight_sha256=ASR_WEIGHT,asr_device='cpu',threads=threads,
        asr_packages={p:importlib.metadata.version(p) for p in ('openai-whisper','whisper-timestamped')},
        word_policy='raw lexical words; no synonyms, recognizer substitution or answer prompt',
        wer_bar=WER_BAR,peak_bar=PEAK_BAR,complete_objects=40,complete_scene_clips=56,complete_scenes=32,
        results=results,passed=all(x['passed'] for x in results),original_waveforms_changed=False,
        passing_subset_accepted=False,reference_text_prompt_used=False,training_approved=False,
        teacher_collection_completed=False,full_measured_native_episode_frame_limit=1100,
        complete_native_episode_bound_verified=False,required_next='Actual asynchronous native teacher protocol/recordings, whole quality/semantic/conversion/numerical/gradient/retention chain')
    atomic_json(out/'source-quality.json',r)
    return r


def verify(source,approval,*,require_pass=True):
    import soundfile as sf
    manifest,plan=source_verify(source);objects,_=requests_for(plan)
    r=json.loads(approval.read_text())
    if (r['schema']!=SCHEMA or r['source_manifest_sha256']!=sha(source/'source-audio.json')
            or r['plan_sha256']!=manifest['plan_sha256'] or r['code_sha256']!=sha(Path(__file__))
            or r['source_code_sha256']!=manifest['code_sha256'] or r['asr_weight_sha256']!=ASR_WEIGHT
            or r['asr_model']!='whisper-turbo' or r['asr_device']!='cpu'
            or r['wer_bar']!=.05 or r['peak_bar']!=.995
            or (r['complete_objects'],r['complete_scene_clips'],r['complete_scenes'])!=(40,56,32)
            or [x['object_id'] for x in r['results']]!=list(objects)
            or r['original_waveforms_changed'] is not False or r['passing_subset_accepted'] is not False
            or r['reference_text_prompt_used'] is not False or r['training_approved'] is not False
            or r['teacher_collection_completed'] is not False or r['full_measured_native_episode_frame_limit']!=1100
            or r['complete_native_episode_bound_verified'] is not False):
        raise ValueError('Complete unchanged independent reasoning32 source audit required')
    verdicts=[]
    for row,original in zip(r['results'],manifest['objects']):
        name=row['object_id']+'-asr.json';path=approval.parent/name;wav=source/original['wav']
        if row['asr_file']!=name or sha(path)!=row['asr_sha256'] or sha(wav)!=row['wav_sha256']:
            raise ValueError('Changed independent ASR or original WAV bytes')
        pcm,rate=sf.read(wav,dtype='float32');verdict=assess(objects[row['object_id']]['input'],json.loads(path.read_text()),pcm,rate)
        if any(row[k]!=v for k,v in verdict.items()):raise ValueError('Recomputed source verdict differs')
        verdicts.append(verdict['passed'])
    if r['passed'] is not all(verdicts) or (require_pass and r['passed'] is not True):
        raise ValueError('Every original source object must pass; failed corpus stays unapproved')
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','weights','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    audit(a.source,a.weights,a.out,a.threads);verify(a.source,a.out/'source-quality.json')
