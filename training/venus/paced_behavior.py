"""Complete paired fresh24 plus all8 original ordinary retention, unforced audio."""
import argparse,asyncio,base64,gc,hashlib,json,time
from pathlib import Path
from contextlib import nullcontext
from .behavior import preflight,run_case
from .spoken_audit import references,REVISION
from .voice_compare import recognizer
from .lora import load_adapter
from .behavior_review import cue_frame
from data.venus.pacing import words
from data.venus.repair import wer
from portable.venus.native_app import NativeEngine,ensure_guard

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

async def main(a):
    import numpy as np,soundfile as sf,torch
    from scipy.signal import resample_poly
    from transformers import set_seed
    fresh=preflight(a.plan,a.timelines,a.approval);assert len(fresh)==24
    retained=[x for x in preflight(a.old_plan,a.old_timelines,a.old_approval) if x['family']=='ordinary'];assert len(retained)==8
    for x in fresh:x['evaluation_set']='fresh24'
    for x in retained:x['evaluation_set']='old8ordinary-retention'
    cases=fresh+retained;stop=json.loads(Path(a.lease).read_text())['stop_unix']
    quality=json.loads(Path(a.word_quality).read_text());assert quality['all_passed'] and quality['passed']==quality['requested']==218
    recognized={r['source_sha256']:r for r in quality['rows']}
    assert stop-time.time()>2*sum(c['frames'] for c in cases)*1.15+450
    terminal=json.loads((Path(a.candidate)/'training-terminal.json').read_text());restore=json.loads((Path(a.candidate)/'fresh-process-restore.json').read_text());recipe=json.loads((Path(a.candidate)/'recipe.json').read_text())
    base_steps=terminal['parent_training_steps']
    assert terminal['completed'] and restore['passed'] and base_steps in (190,244) and 0<terminal['actual_updates']<=(54 if base_steps==244 else 512)
    checkpoint=Path(a.candidate)/('step-%04d.pt'%terminal['actual_updates']);assert sha(a.parent_checkpoint)==a.parent_sha256
    parent=torch.load(a.parent_checkpoint,map_location='cpu',weights_only=False)
    from .native_optimizer import verified_parent
    assert verified_parent(parent,a.parent_sha256,allow244=True)==base_steps
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['metadata']==recipe and state['step']==terminal['actual_updates'] and recipe['approval_sha256']==sha(a.approval)
    out=Path(a.output);out.mkdir(exist_ok=False);engine=NativeEngine(a.model,adapter_checkpoint=checkpoint,voice_reference=getattr(a,'voice_reference',None),voice_reference_sha256=getattr(a,'voice_reference_sha256',None),tts_attention=getattr(a,'tts_attention','released'));results=[]
    parent_label='parent'+str(base_steps)
    conditions=((parent_label,parent['adapter']),('paced-continuation',state['adapter']))
    for condition,adapter in conditions:
        load_adapter(engine.model.llm,adapter);root=out/condition;root.mkdir()
        (root/'profile.json').write_text(json.dumps(dict(engine.profile,condition=condition,optimizer_updates=base_steps if condition==parent_label else terminal['total_optimizer_updates'],checkpoint_sha256=sha(a.parent_checkpoint if condition==parent_label else checkpoint),speech_weights_frozen=True,qualified=False),indent=2)+'\n')
        for case in cases:
            ensure_guard();set_seed(42)
            result=await run_case(engine,case,root,stop);results.append(dict(condition=condition,evaluation_set=case['evaluation_set'],**result))
            if case['family']=='quiet':
                turn=case['user_turns'][-1];eligible,end=cue_frame(turn,recognized[turn['sha256']])
                trace=[json.loads(x) for x in (root/case['id']/'trace.jsonl').read_text().splitlines()]
                voiced=lambda t:t['audio_samples']>0 and t['peak']>.001
                results[-1].update(cue_word_end_seconds=end,first_chunk_that_has_heard_cue=eligible,voiced_frames_before_cue_word=sum(voiced(t) and t['sequence']<eligible for t in trace),voiced_frames_after_cue_word=sum(voiced(t) and t['sequence']>=eligible for t in trace))
            (out/'results.json').write_text(json.dumps(dict(cases=results,required_cases=64,qualified=False),indent=2)+'\n')
            print(json.dumps(dict(event='paired_native_case_complete',condition=condition,id=case['id'],completed=len(results),required=64)),flush=True)
    del engine;gc.collect();torch.cuda.empty_cache();pipe=recognizer();audits=[]
    for result in results:
        ensure_guard();assert stop-time.time()>180
        folder=out/result['condition']/result['id'];traces=[json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()]
        data=[sf.read(folder/t['chunk_file'],dtype='float32')[0] for t in traces];audio=np.concatenate(data);assert np.isfinite(audio).all()
        transcript=pipe(dict(array=resample_poly(audio,2,3).astype(np.float32),sampling_rate=16000),generate_kwargs=dict(language='en',task='transcribe',do_sample=False))['text'] if np.any(audio) else ''
        ref=references(traces);wav=folder/'joined-emitted.wav';sf.write(wav,audio,24000,subtype='FLOAT')
        error=wer(' '.join(words(ref['intended_user_facing'])),' '.join(words(transcript)));peak=float(np.abs(audio).max()) if len(audio) else 0
        audits.append(dict(id=result['id'],condition=result['condition'],evaluation_set=result['evaluation_set'],asr_text=transcript,references=ref,user_facing_text_wer=error,raw_peak=peak,raw_peak_pass=peak<=.995,word_pass=error<=.05,joined_wav_sha256=sha(wav),source_chunks=[dict(file=t['chunk_file'],sha256=sha(folder/t['chunk_file'])) for t in traces],qualified=False))
        (out/'speech-audit-progress.json').write_text(json.dumps(dict(completed=len(audits),required=64,cases=audits),indent=2)+'\n')
    report=dict(complete=len(results)==len(audits)==64,conditions=[parent_label,'paced-continuation'],fresh_cases_per_condition=24,old_ordinary_cases_per_condition=8,cases=audits,source=a.source,asr_revision=REVISION,optimizer_updates=0,seed=42,qualified=False,naturalness_requires_listening=True,semantic_and_all_control_gates_required=True,word_limit=.05,peak_limit=.995,normalizer='predeclared orthography plus cardinal0..100, no homophones')
    (out/'paired-native-report.json').write_text(json.dumps(report,indent=2)+'\n');assert report['complete']

if __name__=='__main__':
    p=argparse.ArgumentParser();[p.add_argument('--'+x,required=True) for x in ('model','candidate','parent-checkpoint','parent-sha256','plan','timelines','approval','word-quality','old-plan','old-timelines','old-approval','output','source','lease')];p.add_argument('--tts-attention',choices=['released','sdpa'],default='released');p.add_argument('--voice-reference');p.add_argument('--voice-reference-sha256');asyncio.run(main(p.parse_args()))
