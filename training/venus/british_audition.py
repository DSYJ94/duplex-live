"""Nine matched native voice auditions, distinct from full paired64 qualification.

Reference labels never count as an accent pass. No synthesis API or optimization.
"""
import asyncio
import gc
import json
from pathlib import Path

from .british_behavior import admit,guarded
from . import british_speech_guard as guard
from .behavior import preflight,run_case

IDS=('venus-control-v1-holdout-ordinary-0216','venus-repair-v2-holdout-quiet-0080','venus-repair-v2-holdout-pause-0097')
REFS={
 'rp-ben':'552adfe446367b04b29cdc0c715822fee95c45e485046ae50ff0a06b7afc61d0',
 'british-alice':'33a4b2813cd3736d9e54f2d840385f0ec7530681b70b1d793271c72ae0336049',
 'british-george-comparator':'40159a59a35053c358d5192b3919a86453c8c6bf1b2019b289119ceca444f674',
}


def selected_cases(args):
    fresh=preflight(args.plan,args.timelines,args.approval)
    old=preflight(args.old_plan,args.old_timelines,args.old_approval)
    by_id={c['id']:c for c in fresh+old}
    if not set(IDS)<=set(by_id):raise ValueError('Exact matched audition inputs required')
    return [by_id[name] for name in IDS]


async def collect(args):
    admit(args)
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import set_seed
    from portable.venus.native_app import NativeEngine,ensure_guard
    from portable.venus.voice import reference
    from .voice_compare import recognizer
    from .spoken_audit import references,REVISION
    from data.venus.repair import wer
    from data.venus.pacing import words
    plan=json.loads(guard.PLAN.read_text());cases=selected_cases(args)
    guarded(args.lease,3*sum(c['frames'] for c in cases)+240)
    paths=plan['audition_reference_paths']
    if set(paths)!=set(REFS):raise ValueError('All three reference profiles required')
    refs={profile:reference(args.model,paths[profile],digest) for profile,digest in REFS.items()}
    root=Path(args.root);checkpoint=root/'main/step-0060.pt'
    if guard.sha(checkpoint)!=guard.CHECKPOINTS['main']:raise ValueError('Exact trained main checkpoint required')
    out=root/'audition';out.mkdir(exist_ok=False);results=[]
    engine=NativeEngine(args.model,adapter_checkpoint=checkpoint,voice_reference=refs['rp-ben']['path'],voice_reference_sha256=REFS['rp-ben'],nonspoken_actions=True,speech_completion=True,tts_attention='released')
    stop=json.loads(Path(args.lease).read_text())['stop_unix']
    for profile in REFS:
        folder=out/profile;folder.mkdir()
        engine.voice_reference=refs[profile]
        (folder/'profile.json').write_text(json.dumps(dict(engine.profile,voice_reference=refs[profile],condition=profile,source=args.source,diagnostic_only=True,qualified=False),indent=2)+'\n')
        for case in cases:
            ensure_guard();guarded(args.lease,case['frames']+180);set_seed(42)
            result=await run_case(engine,case,folder,stop)
            results.append(dict(condition=profile,**result))
            (out/'results.json').write_text(json.dumps(dict(cases=results,required_cases=9,qualified=False),indent=2)+'\n')
            print(json.dumps(dict(event='native_voice_audition_case_complete',profile=profile,id=case['id'],completed=len(results),required=9)),flush=True)
    del engine;gc.collect();torch.cuda.empty_cache()
    pipe=recognizer();audits=[]
    for row in results:
        ensure_guard();guarded(args.lease,180)
        folder=out/row['condition']/row['id']
        trace=[json.loads(s) for s in (folder/'trace.jsonl').read_text().splitlines()]
        audio=np.concatenate([sf.read(folder/t['chunk_file'],dtype='float32')[0] for t in trace])
        if not np.isfinite(audio).all():raise ValueError('Nonfinite original native audio')
        transcript=pipe(dict(array=resample_poly(audio,2,3).astype(np.float32),sampling_rate=16000),generate_kwargs=dict(language='en',task='transcribe',do_sample=False))['text'] if np.any(audio) else ''
        ref=references(trace);wav=folder/'joined-emitted.wav';sf.write(wav,audio,24000,subtype='FLOAT')
        error=wer(' '.join(words(ref['intended_user_facing'])),' '.join(words(transcript)))
        peak=float(np.abs(audio).max()) if len(audio) else 0
        audits.append(dict(id=row['id'],condition=row['condition'],asr_text=transcript,references=ref,user_facing_text_wer=error,word_pass=error<=.05,raw_peak=peak,peak_pass=peak<=.995,joined_wav_sha256=guard.sha(wav),source_chunks=[dict(file=t['chunk_file'],sha256=guard.sha(folder/t['chunk_file'])) for t in trace],qualified=False))
    identities=[{r['id']:r['input_sha256'] for r in results if r['condition']==profile} for profile in REFS]
    if len(results)!=9 or len(audits)!=9 or not identities[0]==identities[1]==identities[2] or set(identities[0])!=set(IDS):
        raise ValueError('All matched9 native recordings required; never a passing subset')
    (out/'voice-audition.json').write_text(json.dumps(dict(complete=True,cases=audits,conditions=list(REFS),required_cases=9,source=args.source,checkpoint_sha256=guard.CHECKPOINTS['main'],asr_revision=REVISION,new_optimizer_updates=0,qualified=False,adopted=False,accent_and_naturalness_require_owner_listening=True,not_a_substitute_for_full_paired64=True,word_limit=.05,peak_limit=.995),indent=2)+'\n')


if __name__=='__main__':
    from .large_behavior import parser
    args=parser().parse_args()
    if args.combine:raise ValueError('Audition cannot be combined into a passing64 cohort')
    asyncio.run(collect(args))
