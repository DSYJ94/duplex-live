"""Versioned whole32 recovery: exact originals, two diagnosed repairs, then new scenes.

No retry/resume, forced requests, oracle feedback, crop, changed source words or
quality bars. First repair002/003; then test the tight-frame case013 before the
remaining25 previously unattempted cases. A failed phase stops new sessions.
"""
import argparse
import asyncio
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import shutil
import subprocess

from evals import native_reasoning_teacher as original
from evals import native_reasoning_teacher_v2 as revised
from evals.native_control_teacher import atomic_json,collection_lock
from evals.native_reasoning_source_quality import verify as verify_source
from evals.native_reasoning_teacher_lifecycle_review import case_review
from evals.native_reasoning_source import sha
from evals.native_reasoning_lifecycle import canonical_hash
from evals.native_reasoning_stream_v2 import FOLLOWUP_GAP_S,RESPONSE_WINDOW_S

SCHEMA='dl-native-reasoning32-teacher-recovery-v2'
PARENT_SHA='34278d0e3014613a86d5066ded971055343143fc533809824210bece294f3c6b'
PARENT_INVENTORY_SHA='7e21d018b38cdc40cf2ac177aa3b99a1e41b0a64b7635fd52b71f068096cc8b2'
REUSE=[f'native-reasoning4-train-{i:03d}' for i in (0,1,4,5)]
REPAIR=[f'native-reasoning4-train-{i:03d}' for i in (2,3)]
SENTINEL='native-reasoning4-train-013'


def code_hashes():
    names=('native_reasoning_teacher_recovery.py','native_reasoning_teacher_lifecycle_review.py')
    return dict(revised.code_hashes(),**{n:sha(Path(__file__).with_name(n)) for n in names})


def reference_packing(source,manifest,plan,tokenizer):
    """Planning diagnostic only: exclude unknown native/provider/ack latency.

    Labels are used ONLY to score/plan capture order, never in model payloads.
    Positive margin does not prove an actual episode will fit.
    """
    import soundfile as sf
    objects={r['object_id']:r for r in manifest['objects']};rows=[]
    for scene,row in zip(manifest['scenes'],plan['cases']):
        lengths=[sf.info(source/objects[c['object_id']]['wav']).duration for c in scene['clips']]
        delay=max(row['annotations_not_model_inputs']['minimum_result_delivery_delay_s'],
                  FOLLOWUP_GAP_S+(lengths[1] if len(lengths)>1 else 0)+4.)
        end=FOLLOWUP_GAP_S+lengths[0]
        if len(lengths)>1:end+=FOLLOWUP_GAP_S+lengths[1]
        if row['family']=='delegate-reasoning':end=max(end,FOLLOWUP_GAP_S+lengths[0]+delay)
        content=row['annotations_not_model_inputs']['expected_final_spoken_words'] if row['family'] in ('delegate-reasoning','replace-reasoning') else ''
        frames=232+math.ceil((end+RESPONSE_WINDOW_S)*12.5)+(2+len(tokenizer.encode(content)) if content else 0)
        rows.append(dict(id=row['id'],source_durations_s=lengths,reference_frames_without_native_provider_ack_or_playout_latency=frames,
                         reference_frame_margin=1100-frames))
    return dict(schema='dl-native-reasoning32-recovery-reference-packing-v1',rows=rows,
                diagnostic_only=True,actual_native_full_frame_bound_proven=False,
                source_oracles_or_answers_supplied_to_model=False,API_or_GPU_calls=0)


def verify_parent(source,quality,parent,inventory,tokenizer):
    verify_source(source,quality);manifest,plan=original.source_manifest(source)
    if sha(parent/'teacher-collection.json')!=PARENT_SHA or sha(inventory)!=PARENT_INVENTORY_SHA:
        raise ValueError('Exact diagnosed six-attempt parent and independently preserved inventory required')
    seen=set()
    for row in json.loads(inventory.read_text())['files']:
        name=Path(row['path']);p=parent/name
        if (name.is_absolute() or '..' in name.parts or name.as_posix() in seen or p.is_symlink()
                or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']):raise ValueError('Parent raw evidence changed')
        seen.add(name.as_posix())
    if {p.relative_to(parent).as_posix() for p in parent.rglob('*') if p.is_file()}!=seen:
        raise ValueError('Unexpected or omitted parent artifact')
    value=json.loads((parent/'teacher-collection.json').read_text());expected=[r['id'] for r in plan['cases'][:6]]
    if (value['schema']!=original.SCHEMA or value['status']!='incomplete_failed_preserved_no_retry'
            or value['source_manifest_sha256']!=sha(source/'source-audio.json')
            or value['source_quality_sha256']!=sha(quality) or value['code_hashes']!=original.code_hashes()
            or value['attempted_ids']!=expected or [r['id'] for r in value['recordings']]!=expected):
        raise ValueError('Exact unchanged v1 collection lineage required')
    scenes={s['id']:s for s in manifest['scenes']};rows={r['id']:r for r in plan['cases']}
    for record in value['recordings']:
        identity=record['id'];folder=parent/identity
        if identity in REUSE:
            original.verify_record(source,scenes[identity],folder,record,tokenizer)
            if not case_review(rows[identity],json.loads((folder/'lifecycle.json').read_text()))['structural_pass']:
                raise ValueError('Reused record has an undiagnosed lifecycle failure')
        elif identity==REPAIR[0]:
            original.verify_record(source,scenes[identity],folder,record,tokenizer)
            diagnostic=case_review(rows[identity],json.loads((folder/'lifecycle.json').read_text()))
            if diagnostic['structural_pass'] or diagnostic['native_requests']!=1:raise ValueError('Original cancellation diagnosis changed')
        elif identity==REPAIR[1]:
            report=json.loads((folder/'report.json').read_text())
            if (record['transport_complete'] is not False or not any(e.get('error_code')=='context_injection_incomplete' for e in report['errors'])
                    or not any(e.get('type')=='missing_commentary_ack' for e in report['errors'])):
                raise ValueError('Original replacement-close diagnosis changed')
    return value,manifest,plan


def recording_implementation(record):
    if record['recorder_version']=='v1' and record['reused_parent_record'] is True:return original
    if record['recorder_version']=='v2' and record['reused_parent_record'] is False:return revised
    raise ValueError('Explicit original or revised recording lineage required')


def verify(source,quality,batch,engine,tokenizer_path):
    tokenizer=original.geometry(engine,tokenizer_path)
    parent,manifest,plan=verify_parent(source,quality,batch/'parent-v1',batch/'parent-inventory.json',tokenizer)
    value=json.loads((batch/'teacher-collection.json').read_text());ids=[s['id'] for s in manifest['scenes']]
    expected_new=REPAIR+[SENTINEL]+[i for i in ids if i not in REUSE+REPAIR+[SENTINEL]]
    if (value['schema']!=SCHEMA or value['status']!='raw_complete_unapproved' or value['code_hashes']!=code_hashes()
            or value['source_manifest_sha256']!=sha(source/'source-audio.json') or value['source_quality_sha256']!=sha(quality)
            or value['parent_collection_sha256']!=PARENT_SHA or value['parent_inventory_sha256']!=PARENT_INVENTORY_SHA
            or value['reused_ids']!=REUSE or value['new_attempt_order']!=expected_new
            or value['attempted_new_ids']!=expected_new or [r['id'] for r in value['recordings']]!=ids
            or value['maximum_new_sessions']!=28 or value['maximum_concurrent_sessions']!=2
            or value['maximum_aggregate_voice_attempts']!=34 or value['maximum_aggregate_backend_requests']!=192
            or value['backend_model']!='gpt-6-luna'
            or value['prior_voice_attempts']!=6 or value['prior_backend_requests']!=5
            or value['passing_subset_accepted'] is not False or value['automatic_retry'] is not False or value['training_approved'] is not False
            or value.get('failures')):raise ValueError('Exact whole32 recovered collection required; no passing subset')
    parents={r['id']:r for r in parent['recordings']}
    for scene,row,record in zip(manifest['scenes'],plan['cases'],value['recordings']):
        folder=batch/scene['id'];implementation=recording_implementation(record)
        if scene['id'] in REUSE:
            expected=dict(parents[scene['id']],recorder_version='v1',reused_parent_record=True,
                          original_record_sha256=canonical_hash(parents[scene['id']]))
            if record!=expected:raise ValueError('Reused original receipt altered')
        elif implementation is not revised:raise ValueError('New attempt must use revised actual recorder')
        implementation.verify_record(source,scene,folder,record,tokenizer)
        report=json.loads((folder/'report.json').read_text())
        if not report['native_frame_budget']['frame_count_diagnostic_pass']:raise ValueError('Complete measured native frame budget exceeded')
        if not case_review(row,json.loads((folder/'lifecycle.json').read_text()))['structural_pass']:
            raise ValueError('Actual lifecycle/oracle diagnostic failed')
    if (sha(batch/'reference-packing.json')!=value['reference_packing_sha256']
            or json.loads((batch/'reference-packing.json').read_text())!=reference_packing(source,manifest,plan,tokenizer)):
        raise ValueError('Predeclared source packing diagnostic changed')
    return value,manifest,plan


async def recover(source,quality,parent,inventory,out,env_file,engine,tokenizer_path):
    tokenizer=original.geometry(engine,tokenizer_path);old,manifest,plan=verify_parent(source,quality,parent,inventory,tokenizer)
    scenes={s['id']:s for s in manifest['scenes']};rows={r['id']:r for r in plan['cases']};ids=list(scenes)
    order=REPAIR+[SENTINEL]+[i for i in ids if i not in REUSE+REPAIR+[SENTINEL]]
    with collection_lock(out.parent/'.private/native-reasoning32-teacher.lock'):
        out.mkdir(parents=True,exist_ok=False);shutil.copytree(parent,out/'parent-v1');shutil.copyfile(inventory,out/'parent-inventory.json')
        code=code_hashes();value=dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),
            repo_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_hashes=code,
            source_manifest_sha256=sha(source/'source-audio.json'),source_quality_sha256=sha(quality),
            parent_collection_sha256=PARENT_SHA,parent_inventory_sha256=PARENT_INVENTORY_SHA,
            reused_ids=REUSE,new_attempt_order=order,attempted_new_ids=[],recordings=[],maximum_new_sessions=28,
            maximum_concurrent_sessions=2,prior_voice_attempts=6,maximum_aggregate_voice_attempts=34,
            backend_model='gpt-6-luna',prior_backend_requests=5,maximum_aggregate_backend_requests=192,automatic_retry=False,
            passing_subset_accepted=False,training_approved=False,status='preparing_versioned_recovery')
        packing=reference_packing(source,manifest,plan,tokenizer)
        if min(packing['rows'],key=lambda r:r['reference_frame_margin'])['id']!=SENTINEL:
            raise ValueError('Exact measured tight-frame sentinel changed')
        atomic_json(out/'reference-packing.json',packing);value['reference_packing_sha256']=sha(out/'reference-packing.json')
        target=out/'teacher-collection.json'
        def save():atomic_json(target,value)
        for record in old['recordings']:
            if record['id'] not in REUSE:continue
            shutil.copytree(parent/record['id'],out/record['id'])
            value['recordings'].append(dict(record,recorder_version='v1',reused_parent_record=True,original_record_sha256=canonical_hash(record)))
        save();stop=False;semaphore=asyncio.Semaphore(2)
        async def one(identity):
            nonlocal stop
            async with semaphore:
                if stop:return
                value['attempted_new_ids'].append(identity);save()
                try:
                    record=await revised.capture_one(source,scenes[identity],rows[identity],out/identity,sha(quality),
                                                      'gpt-6-luna',env_file,tokenizer)
                    record.update(recorder_version='v2',reused_parent_record=False)
                    value['recordings'].append(record);value['recordings'].sort(key=lambda r:r['id']);save()
                    if not record['transport_complete']:raise ValueError('Actual revised transport failed')
                    revised.verify_record(source,scenes[identity],out/identity,record,tokenizer)
                    diagnostic=case_review(rows[identity],json.loads((out/identity/'lifecycle.json').read_text()))
                    if not record['frame_count_diagnostic_pass']:raise ValueError('Complete native frame budget failed')
                    if not diagnostic['structural_pass']:
                        atomic_json(out/(identity+'-failed-lifecycle.json'),diagnostic)
                        raise ValueError('Actual lifecycle diagnostic failed')
                    print(json.dumps(dict(id=identity,transport_complete=True,structural_diagnostic_pass=True,
                                          complete_recordings=len(value['recordings']),expected=32,training_approved=False)),flush=True)
                except Exception as error:
                    stop=True;value.setdefault('failures',[]).append(dict(id=identity,error_type=type(error).__name__,reason=str(error)))
                    save();print(json.dumps(dict(id=identity,failed_preserved=True,reason=str(error),no_retry=True)),flush=True)
        try:
            value['status']='repairing_two_diagnosed_failures';save()
            await asyncio.gather(*(one(i) for i in REPAIR))
            if not stop:
                value['status']='checking_tight_native_frame_case';save();await one(SENTINEL)
            if not stop:
                value['status']='collecting_remaining_unattempted_scenes';save()
                await asyncio.gather(*(one(i) for i in order if i not in REPAIR+[SENTINEL]))
        finally:
            value['status']='raw_complete_unapproved' if not stop and len(value['recordings'])==32 else 'incomplete_failed_preserved_no_retry'
            if code_hashes()!=code:value['status']='capture_code_changed_failed'
            save()
        if value['status']=='raw_complete_unapproved':verify(source,quality,out,engine,tokenizer_path)
        return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','quality','parent','parent-inventory','out','env-file','engine','tokenizer'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=asyncio.run(recover(a.source,a.quality,a.parent,a.parent_inventory,a.out,a.env_file,a.engine,a.tokenizer))
    print(json.dumps(dict(status=r['status'],recordings=len(r['recordings']),training_approved=False)))
    if r['status']!='raw_complete_unapproved':raise SystemExit(1)
