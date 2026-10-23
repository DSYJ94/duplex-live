"""Independent whole32 teacher speech audit, never a semantic/training approval.

Requires every actual raw episode and reconstructs its immutable transport first.
No oracle/reference prompt, recognizer substitution, crop or passing subset.
"""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path

from evals import native_reasoning_source_quality as source_quality
from evals import native_reasoning_teacher as teacher
from evals.native_control_teacher import atomic_json
from evals.native_reasoning_source import sha

SCHEMA='dl-native-reasoning32-independent-teacher-speech-v1'


def whole_collection(source,quality,batch,engine,tokenizer_path):
    header=json.loads((batch/'teacher-collection.json').read_text())
    if header['schema']=='dl-native-reasoning32-teacher-recovery-v4':
        from evals.native_reasoning_teacher_recovery_v4 import verify
        return verify(source,quality,batch,engine,tokenizer_path)
    if header['schema']=='dl-native-reasoning32-teacher-recovery-v3':
        from evals.native_reasoning_teacher_recovery_v3 import verify
        return verify(source,quality,batch,engine,tokenizer_path)
    if header['schema']=='dl-native-reasoning32-teacher-recovery-v2':
        from evals.native_reasoning_teacher_recovery import verify
        return verify(source,quality,batch,engine,tokenizer_path)
    source_quality.verify(source,quality);manifest,plan=teacher.source_manifest(source)
    tokenizer=teacher.geometry(engine,tokenizer_path)
    value=json.loads((batch/'teacher-collection.json').read_text());ids=[s['id'] for s in manifest['scenes']]
    if (value['schema']!=teacher.SCHEMA or value['status']!='raw_complete_unapproved'
            or value['source_manifest_sha256']!=sha(source/'source-audio.json')
            or value['source_quality_sha256']!=sha(quality) or value['code_hashes']!=teacher.code_hashes()
            or sorted(value['attempted_ids'])!=ids or [r['id'] for r in value['recordings']]!=ids
            or len(value['recordings'])!=32 or value['maximum_sessions']!=32
            or value['maximum_concurrent_sessions']!=2 or value['maximum_backend_requests']!=192
            or value['automatic_retry'] is not False or value['training_approved'] is not False
            or value.get('failures')):raise ValueError('Exact complete actual32 teacher collection required; no passing subset')
    for scene,record in zip(manifest['scenes'],value['recordings']):
        teacher.verify_record(source,scene,batch/scene['id'],record,tokenizer)
    return value,manifest,plan


def expected_output(folder):
    rows=json.loads((folder/'transcripts.json').read_text())
    # Concatenation preserves the actual provider fragments. Normalization is
    # the unchanged lexical/numeral policy, never an authored scoring answer.
    return ''.join(e['delta'] for e in rows if e['type']=='session.output_transcript.delta').strip()


def audit(source,quality,batch,engine,tokenizer_path,weights,out,threads=8):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    import whisper_timestamped as whisper
    collection,manifest,_=whole_collection(source,quality,batch,engine,tokenizer_path)
    if sha(weights)!=source_quality.ASR_WEIGHT or type(threads) is not int or not 1<=threads<=8:
        raise ValueError('Exact retained independent ASR and measured bounded CPU thread count required')
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    out.mkdir(parents=True,exist_ok=False);code=sha(Path(__file__));model=whisper.load_model(str(weights),device='cpu');results=[]
    for scene in manifest['scenes']:
        folder=batch/scene['id'];wav=folder/'model_out.wav';pcm,rate=sf.read(wav,dtype='float32')
        if rate!=24000 or pcm.ndim!=1:raise ValueError('No silent audio conversion or repair')
        text=expected_output(folder)
        raw=whisper.transcribe(model,resample_poly(pcm,2,3).astype(np.float32),language='en',
            vad=False,best_of=5,beam_size=5,temperature=0.,fp16=False,verbose=None)
        name=scene['id']+'-asr.json';raw=source_quality.persisted_asr(out/name,raw)
        verdict=source_quality.assess(text,raw,pcm,rate)
        report=json.loads((folder/'report.json').read_text())
        if not text:verdict['reasons'].append('No actual provider speech transcript')
        if report['native_frame_budget']['frame_count_diagnostic_pass'] is not True:
            verdict['reasons'].append('Complete prefix/live/actually sent context spans exceed unchanged1100-frame bound')
        verdict['passed']=not verdict['reasons']
        results.append(dict(id=scene['id'],original_rendered_wav_sha256=sha(wav),
                            actual_provider_transcript=text,asr_file=name,asr_sha256=sha(out/name),**verdict))
        atomic_json(out/'progress.json',dict(complete_cases=len(results),expected=32,latest_passed=verdict['passed'],training_approved=False))
        print(json.dumps(dict(id=scene['id'],complete=len(results),expected=32,passed=verdict['passed'])),flush=True)
    whole_collection(source,quality,batch,engine,tokenizer_path)
    if sha(Path(__file__))!=code:raise ValueError('Audit implementation changed')
    value=dict(schema=SCHEMA,at_utc=datetime.now(timezone.utc).isoformat(),complete_cases=32,
        teacher_collection_sha256=sha(batch/'teacher-collection.json'),source_quality_sha256=sha(quality),
        code_sha256=code,assessment_code_sha256=sha(Path(source_quality.__file__)),
        asr_model='whisper-turbo',asr_weight_sha256=source_quality.ASR_WEIGHT,asr_device='cpu',threads=threads,
        asr_packages={p:importlib.metadata.version(p) for p in ('openai-whisper','whisper-timestamped')},
        normalization_code_sha256=sha(source_quality.NORMALIZATION_CODE),word_policy=source_quality.WORD_POLICY,
        wer_bar=.05,peak_bar=.995,results=results,passed=all(r['passed'] for r in results),
        reference_text_prompt_used=False,operator_or_oracle_answers_used=False,original_audio_changed=False,
        passing_subset_accepted=False,semantic_approved=False,training_approved=False,
        native_conversion_numerical_gradient_and_behavior_approved=False)
    atomic_json(out/'teacher-speech-quality.json',value);return value


def verify(source,quality,batch,engine,tokenizer_path,approval,*,require_pass=True):
    import soundfile as sf
    _,manifest,_=whole_collection(source,quality,batch,engine,tokenizer_path);v=json.loads(approval.read_text())
    if (v['schema']!=SCHEMA or v['complete_cases']!=32 or [r['id'] for r in v['results']]!=[s['id'] for s in manifest['scenes']]
            or v['teacher_collection_sha256']!=sha(batch/'teacher-collection.json') or v['source_quality_sha256']!=sha(quality)
            or v['code_sha256']!=sha(Path(__file__)) or v['assessment_code_sha256']!=sha(Path(source_quality.__file__))
            or v['asr_model']!='whisper-turbo' or v['asr_weight_sha256']!=source_quality.ASR_WEIGHT
            or v['asr_device']!='cpu' or type(v['threads']) is not int or not 1<=v['threads']<=8
            or v['asr_packages']!={p:importlib.metadata.version(p) for p in ('openai-whisper','whisper-timestamped')}
            or v['normalization_code_sha256']!=sha(source_quality.NORMALIZATION_CODE) or v['word_policy']!=source_quality.WORD_POLICY
            or v['wer_bar']!=.05 or v['peak_bar']!=.995
            or any(v[k] is not False for k in ('reference_text_prompt_used','operator_or_oracle_answers_used',
                'original_audio_changed','passing_subset_accepted','semantic_approved','training_approved',
                'native_conversion_numerical_gradient_and_behavior_approved'))):raise ValueError('Changed whole32 independent teacher speech procedure')
    verdicts=[]
    for row in v['results']:
        folder=batch/row['id'];wav=folder/'model_out.wav';path=approval.parent/row['asr_file'];text=expected_output(folder)
        if (row['asr_file']!=row['id']+'-asr.json' or sha(path)!=row['asr_sha256']
                or sha(wav)!=row['original_rendered_wav_sha256'] or row['actual_provider_transcript']!=text):raise ValueError('Changed raw words/audio/independent recognition')
        pcm,rate=sf.read(wav,dtype='float32');verdict=source_quality.assess(text,json.loads(path.read_text()),pcm,rate)
        report=json.loads((folder/'report.json').read_text())
        if not text:verdict['reasons'].append('No actual provider speech transcript')
        if report['native_frame_budget']['frame_count_diagnostic_pass'] is not True:
            verdict['reasons'].append('Complete prefix/live/actually sent context spans exceed unchanged1100-frame bound')
        verdict['passed']=not verdict['reasons']
        if any(row[k]!=value for k,value in verdict.items()):raise ValueError('Recomputed teacher speech verdict differs')
        verdicts.append(verdict['passed'])
    if v['passed'] is not all(verdicts) or (require_pass and v['passed'] is not True):raise ValueError('All32 teachers must pass; no passing subset')
    return v


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','source-quality','batch','engine','tokenizer','weights','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--threads',type=int,default=8);a=p.parse_args()
    audit(a.source,a.source_quality,a.batch,a.engine,a.tokenizer,a.weights,a.out,a.threads)
    verify(a.source,a.source_quality,a.batch,a.engine,a.tokenizer,a.out/'teacher-speech-quality.json')
