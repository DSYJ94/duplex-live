"""Replay cached real plans and recorded native requests against local records.

No new model inference occurs. Scaled timing is explicitly artificial; this is
not a fresh native/provider run, production latency test or model acceptance.
"""
import asyncio
import hashlib
import json
from pathlib import Path
import wave
from portable.venus.context_worker import ContextPlanWorker
from portable.venus.task_context import AudioEvidence


def read(p):return json.loads(Path(p).read_text())


async def main():
    proposal=read('docs/NATIVE-VENUS-CONTEXT-BACKEND21-PROPOSAL-2026-10-31.json')
    sources=read('docs/NATIVE-VENUS-CAUSAL-TASK-REPLAY-2026-10-31.json')['records']
    results=read('runs/venus-training-20261031/context-backend21-v2/report.json')
    assert results['all21_complete'] and results['actual_post_attempts']==21
    root=Path('runs/venus-training-20261031/context-backend21-v2')
    cases=read('runs/venus-training-20261029/action-v44/copyback/runtime-native-v44/results.json')['cases']
    scenes={s['id']:s for s in read('evals/fixtures/venus-control-repair-source-v2-20261027.json')['scenes']}
    scale=.002;rows=[]
    for case in cases:
        scheduled=[(i,s) for i,s in enumerate(sources) if s['scene_id']==case['id']]
        if not scheduled:
            rows.append(dict(id=case['id'],requests=0,exercised=False,qualified=False));continue
        scene=scenes[case['id']];cache={};evidences={};actual_reads=[]
        for i,s in scheduled:
            p=proposal['requests'][i]
            with wave.open(p['audio_file']) as w:pcm=w.readframes(w.getnframes())
            e=AudioEvidence(p['causal_audio']['first_sequence'],p['causal_audio']['last_sequence'],pcm)
            assert e.receipt()==s['evidence']
            key=(s['native_selected_description'],e.receipt()['pcm_sha256'])
            assert key not in cache, 'Ambiguous cached decision needs separate diagnosis'
            cache[key]=read(root/f'reasoning-{i:02d}.json');evidences[i]=e
        async def planner(q,e):return cache[q,e.receipt()['pcm_sha256']]
        async def lookup(location):
            await asyncio.sleep(scene['teacher_program'].get('worker_delay_seconds',8)*scale)
            value=(scene['synthetic_private_world'] or {}).get(location)
            actual_reads.append(dict(location=location,found=value is not None))
            return 'Lookup '+location+': '+value if value is not None else 'Entry '+location+' was not found.'
        w=ContextPlanWorker(planner,lookup);origin=asyncio.get_running_loop().time()
        for i,s in scheduled:
            await asyncio.sleep(max(0,origin+s['request_close_sequence']*scale-asyncio.get_running_loop().time()))
            w.submit(s['native_selected_description'],raw_query=s['native_raw_description'],audio_evidence=evidences[i])
        await asyncio.gather(*w.tasks.values(),return_exceptions=True)
        feedback=w.take_results()
        assert w.take_results() is None
        events=list(w.events)
        rows.append(dict(id=case['id'],requests=len(scheduled),accepted_requests=len(w.states),actual_reads=actual_reads,
            cancelled_tasks=sum(e['type']=='worker_cancelled' for e in events),
            cancellation_targets=[e['targets'] for e in events if e['type']=='cancellation_confirmed'],
            feedback=feedback,events=events,qualified=False))
        await w.close()
    assert len(rows)==32 and sum(r['requests'] for r in rows)==21
    report=dict(schema='dl-context-execution-replay-v1',source_cohort_cases=32,recorded_native_requests=21,
        all21_requests_accounted_for=True,scene_cases=rows,external_calls=0,new_native_inference=False,
        cached_real_provider_decisions=True,artificial_timing_scale=scale,
        cancelled_tasks=sum(r.get('cancelled_tasks',0) for r in rows),
        actual_record_reads=sum(len(r.get('actual_reads',[])) for r in rows),
        rejected_empty_native_requests=sum(e['type']=='native_delegate_rejected' for r in rows for e in r.get('events',[])),
        actual_provider_latency_replayed=False,qualified=False,adopted=False,
        causal_limit='Recorded native request sequence is fixed. Correct backend feedback could change subsequent native generation; this replay does not establish fresh end-to-end behavior.')
    path=Path('docs/NATIVE-VENUS-CONTEXT-EXECUTION-REPLAY-2026-10-31.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='scene_cases'}))


if __name__=='__main__':asyncio.run(main())
