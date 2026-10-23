"""Reassess all40 originals; recognize only the one changed provider waveform.

The other39 independent ASR artifacts are reused byte-for-byte only after their
original source qualification and unchanged WAV hashes are verified. Every
word, interval, peak and full-corpus verdict is recomputed at unchanged limits.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil

from evals.native_control_teacher import atomic_json
from evals.native_reasoning_source import requests_for, sha
from evals.native_reasoning_source_pacing import OBJECT, QUALITY, verify as source_verify
from evals import native_reasoning_source_quality as previous

SCHEMA = 'dl-native-reasoning32-independent-user-source-quality-v4'


def rows(source, manifest, plan, folder, parent):
    import soundfile as sf
    objects, scenes = requests_for(plan)
    if (len(manifest['objects']) != 40 or manifest['scenes'] != scenes
            or [r['object_id'] for r in manifest['objects']] != list(objects)
            or [r['object_id'] for r in parent['results']] != list(objects)):
        raise ValueError('All40 objects,56 clips and32 original matched TRAIN scenes required')
    old = {r['object_id']:r for r in parent['results']}; result = []
    for row in manifest['objects']:
        identity = row['object_id']; name = identity+'-asr.json'; path = folder/name
        wav = source/row['wav']; pcm, rate = sf.read(wav,dtype='float32')
        reused = identity != OBJECT
        if reused and (sha(wav) != old[identity]['wav_sha256'] or sha(path) != old[identity]['asr_sha256']):
            raise ValueError('Reused independent evidence requires identical WAV and ASR bytes')
        verdict = previous.assess(objects[identity]['input'],json.loads(path.read_text()),pcm,rate)
        if reused and any(old[identity][k] != v for k,v in verdict.items()):
            raise ValueError('Recomputed unchanged-source verdict differs from original qualification')
        result.append(dict(object_id=identity,wav_sha256=sha(wav),asr_file=name,asr_sha256=sha(path),
                           independent_asr_reused=reused,**verdict))
    return result


def metadata(source, manifest):
    return dict(schema=SCHEMA,source_manifest_sha256=sha(source/'source-audio.json'),
                plan_sha256=manifest['plan_sha256'],code_sha256=sha(Path(__file__)),
                source_code_sha256=manifest['code_sha256'],assessor_code_sha256=sha(Path(previous.__file__)),
                parent_quality_sha256=QUALITY,asr_model='whisper-turbo',asr_weight_sha256=previous.ASR_WEIGHT,
                asr_device='cpu',threads=8,
                asr_packages={p:importlib.metadata.version(p)for p in ('openai-whisper','whisper-timestamped')},
                word_policy=previous.WORD_POLICY,normalization_code_sha256=sha(previous.NORMALIZATION_CODE),
                wer_bar=.05,peak_bar=.995,complete_objects=40,complete_scene_clips=56,complete_scenes=32,
                unchanged_ASR_artifacts_reused=39,new_independent_ASR_recordings=1,
                original_waveforms_changed=False,passing_subset_accepted=False,reference_text_prompt_used=False,
                training_approved=False,teacher_collection_completed=False,
                full_measured_native_episode_frame_limit=1100,complete_native_episode_bound_verified=False)


def parent_approval(source):
    return previous.verify(source/'parent-v3',source/'parent-quality-v3/source-quality.json')


def audit(source, weights, out):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    import whisper_timestamped as whisper
    manifest, plan = source_verify(source); parent = parent_approval(source)
    if sha(weights) != previous.ASR_WEIGHT:
        raise ValueError('Exact previously pinned independent Whisper turbo required')
    out.mkdir(parents=True,exist_ok=False); code = sha(Path(__file__))
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    model = whisper.load_model(str(weights),device='cpu')
    old = {r['object_id']:r for r in parent['results']}
    for row in manifest['objects']:
        identity=row['object_id']; target=out/(identity+'-asr.json')
        if identity != OBJECT:
            shutil.copyfile(source/'parent-quality-v3'/old[identity]['asr_file'],target)
        else:
            pcm,rate=sf.read(source/row['wav'],dtype='float32')
            if rate != 24000 or pcm.ndim != 1:
                raise ValueError('Original provider waveform required; no input repair')
            independent=whisper.transcribe(model,resample_poly(pcm,2,3).astype(np.float32),language='en',
                            vad=False,best_of=5,beam_size=5,temperature=0.,fp16=False,verbose=None)
            previous.persisted_asr(target,independent)
    verdicts=rows(source,manifest,plan,out,parent)
    source_verify(source)
    if code != sha(Path(__file__)):
        raise ValueError('Quality implementation changed during recognition')
    r=dict(**metadata(source,manifest),at_utc=datetime.now(timezone.utc).isoformat(),
           results=verdicts,passed=all(v['passed']for v in verdicts))
    atomic_json(out/'source-quality.json',r)
    verify(source,out/'source-quality.json',require_pass=False)
    print(json.dumps(dict(passed=r['passed'],complete_objects=40,new_ASR=1,reused_ASR=39,
                          changed_object=next(v for v in verdicts if v['object_id']==OBJECT))),flush=True)
    return r


def verify(source, approval, *, require_pass=True):
    manifest,plan=source_verify(source); parent=parent_approval(source)
    r=json.loads(approval.read_text()); expected=metadata(source,manifest)
    if any(r.get(k) != v for k,v in expected.items()):
        raise ValueError('Exact unchanged limits, recognizer, implementation and complete40 source binding required')
    verdicts=rows(source,manifest,plan,approval.parent,parent)
    if (r.get('results') != verdicts or r.get('passed') is not all(v['passed']for v in verdicts)
            or (require_pass and r.get('passed') is not True)):
        raise ValueError('Every source object must independently pass; no subset approval')
    expected_files={'source-quality.json'}|{v['asr_file']for v in verdicts}
    if (any(p.is_symlink()for p in approval.parent.rglob('*'))
            or {p.relative_to(approval.parent).as_posix()for p in approval.parent.rglob('*')if p.is_file()} != expected_files):
        raise ValueError('Exact independent quality inventory required')
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('audit','verify'))
    for name in ('source','weights','out','approval'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.action=='audit':audit(a.source,a.weights,a.out)
    else:verify(a.source,a.approval)
