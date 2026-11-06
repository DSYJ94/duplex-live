"""Complete frozen32-case native collection per capacity arm, unchanged speech gates.

The old paced_behavior phase/lineage checks are not weakened. This separate
entrypoint requires the new completed arm and independent final native restore.
"""
import argparse
import asyncio
import gc
import json
from pathlib import Path
import time

from .behavior import preflight, run_case
from .behavior_review import cue_frame
from .large_inputs import sha
from .large_native import guarded


async def collect(a,*,british_composition=False,execution_context=None):
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import set_seed
    from portable.venus.native_app import NativeEngine,ensure_guard
    from .voice_compare import recognizer
    from .spoken_audit import references,REVISION
    from data.venus.pacing import words
    from data.venus.repair import wer
    check=guarded;training_source=a.source;engine_options={}
    if execution_context is not None:
        execution_context.admit(a)
        check=execution_context.guarded
        training_source=execution_context.original_training_source
        engine_options=dict(nonspoken_actions=True,speech_completion=True)
    elif british_composition:
        from .british_behavior import admit,guarded as british_guarded
        admit(a);check=british_guarded
        training_source='b39ce3ebe167b059b67fa3d131978e5da7c41ac1'
        engine_options=dict(nonspoken_actions=True,speech_completion=True)
    check(a.lease,600)
    root=Path(a.root);arm=root/a.arm
    terminal=json.loads((arm/'training-terminal.json').read_text())
    restore=json.loads((arm/'final-native-restore.json').read_text())
    checkpoint=arm/('step-%04d.pt'%terminal['actual_new_updates'])
    if (not terminal['completed'] or not restore['passed'] or restore['source']!=training_source
            or not restore['actual_CUDA'] or restore['new_optimizer_updates']!=0
            or sha(checkpoint)!=restore['checkpoint_sha256']):
        raise ValueError('Actual complete training and independent full native restore required')
    fresh=preflight(a.plan,a.timelines,a.approval)
    retained=[x for x in preflight(a.old_plan,a.old_timelines,a.old_approval) if x['family']=='ordinary']
    if len(fresh)!=24 or len(retained)!=8:raise ValueError('All frozen24 plus old8 ordinary required')
    for x in fresh:x['evaluation_set']='fresh24'
    for x in retained:x['evaluation_set']='old8ordinary-retention'
    cases=fresh+retained
    quality=json.loads(Path(a.word_quality).read_text())
    if not quality['all_passed'] or quality['passed']!=quality['requested'] or quality['requested']!=218:
        raise ValueError('Complete original218 user-source word approval required')
    recognized={r['source_sha256']:r for r in quality['rows']}
    stop=json.loads(Path(a.lease).read_text())['stop_unix']
    if time.time()+sum(c['frames'] for c in cases)*1.5+600>=stop:
        raise TimeoutError('Whole32 plus independent speech review must fit; never a subset')
    out=root/'native'/a.arm;out.mkdir(parents=True,exist_ok=False)
    engine=NativeEngine(a.model,adapter_checkpoint=checkpoint,voice_reference=a.voice_reference,
                        voice_reference_sha256=a.voice_reference_sha256,tts_attention='released',**engine_options)
    (out/'profile.json').write_text(json.dumps(dict(engine.profile,source=a.source,
                  training_proof_source=restore['source'],checkpoint_sha256=restore['checkpoint_sha256'],arm=a.arm,adopted=False),indent=2)+'\n')
    results=[]
    for case in cases:
        ensure_guard();set_seed(42)
        result=dict(condition=a.arm,evaluation_set=case['evaluation_set'],**(await run_case(engine,case,out,stop)))
        if case['family']=='quiet':
            turn=case['user_turns'][-1];eligible,end=cue_frame(turn,recognized[turn['sha256']])
            traces=[json.loads(s) for s in (out/case['id']/'trace.jsonl').read_text().splitlines()]
            voiced=lambda t:t['audio_samples']>0 and t['peak']>.001
            result.update(cue_word_end_seconds=end,first_chunk_that_has_heard_cue=eligible,
                          voiced_frames_before_cue_word=sum(voiced(t) and t['sequence']<eligible for t in traces),
                          voiced_frames_after_cue_word=sum(voiced(t) and t['sequence']>=eligible for t in traces))
        results.append(result)
        (out/'results.json').write_text(json.dumps(dict(cases=results,required_cases=32,qualified=False),indent=2)+'\n')
        print(json.dumps(dict(event='capacity_native_case_complete',arm=a.arm,id=case['id'],completed=len(results),required=32)),flush=True)
    del engine;gc.collect();torch.cuda.empty_cache()
    pipe=recognizer();audits=[]
    for result in results:
        ensure_guard();check(a.lease,180)
        folder=out/result['id'];traces=[json.loads(s) for s in (folder/'trace.jsonl').read_text().splitlines()]
        data=[sf.read(folder/t['chunk_file'],dtype='float32')[0] for t in traces]
        audio=np.concatenate(data)
        if not np.isfinite(audio).all():raise ValueError('Nonfinite raw native audio')
        transcript=pipe(dict(array=resample_poly(audio,2,3).astype(np.float32),sampling_rate=16000),
                        generate_kwargs=dict(language='en',task='transcribe',do_sample=False))['text'] if np.any(audio) else ''
        ref=references(traces);wav=folder/'joined-emitted.wav';sf.write(wav,audio,24000,subtype='FLOAT')
        error=wer(' '.join(words(ref['intended_user_facing'])),' '.join(words(transcript)))
        peak=float(np.abs(audio).max()) if len(audio) else 0
        audits.append(dict(id=result['id'],condition=a.arm,evaluation_set=result['evaluation_set'],asr_text=transcript,
                references=ref,user_facing_text_wer=error,raw_peak=peak,raw_peak_pass=peak<=.995,word_pass=error<=.05,
                joined_wav_sha256=sha(wav),source_chunks=[dict(file=t['chunk_file'],sha256=sha(folder/t['chunk_file'])) for t in traces],qualified=False))
        (out/'speech-audit-progress.json').write_text(json.dumps(dict(completed=len(audits),required=32,cases=audits),indent=2)+'\n')
    report=dict(complete=len(results)==len(audits)==32,arm=a.arm,cases=audits,source=a.source,asr_revision=REVISION,
                new_optimizer_updates=0,seed=42,qualified=False,word_limit=.05,peak_limit=.995,
                naturalness_requires_listening=True,semantic_and_all_control_gates_required=True)
    (out/'native-report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['complete']:raise ValueError('Incomplete native arm')


def combine(a):
    root=Path(a.root)/'native';reports=[];results=[]
    for arm in ('main','control'):
        r=json.loads((root/arm/'native-report.json').read_text());x=json.loads((root/arm/'results.json').read_text())
        if not r['complete'] or len(r['cases'])!=32 or len(x['cases'])!=32 or r['source']!=a.source:
            raise ValueError('Both complete32 native/speech arms required')
        reports.extend(r['cases']);results.extend(x['cases'])
    identities=[{r['id']:r['input_sha256'] for r in results if r['condition']==arm} for arm in ('main','control')]
    if identities[0]!=identities[1] or len(identities[0])!=32:raise ValueError('All paired input hashes must match')
    (root/'results.json').write_text(json.dumps(dict(cases=results,required_cases=64,qualified=False),indent=2)+'\n')
    (root/'paired-native-report.json').write_text(json.dumps(dict(complete=True,conditions=['main','control'],cases=reports,
                source=a.source,new_optimizer_updates=0,qualified=False,adopted=False,word_limit=.05,peak_limit=.995,
                normalizer='unchanged predeclared orthography/cardinal0..100, no homophones',matched_input_hashes=True),indent=2)+'\n')


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--combine',action='store_true')
    p.add_argument('--root',required=True);p.add_argument('--source',required=True)
    p.add_argument('--arm',choices=('main','control'))
    for name in ('model','lease','plan','timelines','approval','word-quality','old-plan','old-timelines','old-approval','voice-reference','voice-reference-sha256'):
        p.add_argument('--'+name)
    return p


if __name__=='__main__':
    a=parser().parse_args();combine(a) if a.combine else asyncio.run(collect(a))
