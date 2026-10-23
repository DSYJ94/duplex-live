"""Whole32 recovery v3: preserve five original records; repair the actual cancel race.

No retries, source edits, subset approval, intent classifier or gate relaxation.
The original v1/v2 code and complete parent evidence remain immutable.
"""
import argparse
import asyncio
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
from evals import native_reasoning_teacher as original
from evals import native_reasoning_teacher_v2 as parent_recorder
from evals import native_reasoning_teacher_v3 as revised
from evals import native_reasoning_teacher_recovery as previous
from evals.native_control_teacher import atomic_json,collection_lock
from evals.native_reasoning_source_quality import verify as verify_source
from evals.native_reasoning_teacher_lifecycle_review import case_review
from evals.native_reasoning_source import sha
from evals.native_reasoning_lifecycle import canonical_hash

SCHEMA='dl-native-reasoning32-teacher-recovery-v3'
PARENT_SHA='54870fb4d813f4b779c052897cd5c6dc08303679876aa2f2416a1cc74be7ef20'
PARENT_INVENTORY_SHA='e56c751702c45e9238ff06efff12d8c12129cd836bc3936bb966f322e1ec8132'
REUSE=[f'native-reasoning4-train-{i:03d}' for i in (0,1,3,4,5)]
REPAIR=['native-reasoning4-train-002']
SENTINEL='native-reasoning4-train-013'
reference_packing=previous.reference_packing

def code_hashes():
    result=dict(previous.code_hashes(),**revised.code_hashes())
    result[Path(__file__).name]=sha(Path(__file__))
    return result


def recording_implementation(record):
    if record['reused_in_v3'] is True:
        if record['recorder_version']=='v1':return original
        if record['recorder_version']=='v2':return parent_recorder
    elif record['reused_in_v3'] is False and record['recorder_version']=='v3':return revised
    raise ValueError('Explicit preserved v1/v2 or revised v3 recording lineage required')


def verify_parent(source,quality,parent,inventory,tokenizer):
    _,manifest,plan=previous.verify_parent(source,quality,parent/'parent-v1',parent/'parent-inventory.json',tokenizer)
    if sha(parent/'teacher-collection.json')!=PARENT_SHA or sha(inventory)!=PARENT_INVENTORY_SHA:
        raise ValueError('Exact diagnosed v2 parent and independently preserved inventory required')
    seen=set()
    for row in json.loads(inventory.read_text())['files']:
        name=Path(row['path']);p=parent/name
        if (name.is_absolute() or '..' in name.parts or str(name) in seen or p.is_symlink()
                or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']):raise ValueError('Parent v2 raw evidence changed')
        seen.add(name.as_posix())
    if {p.relative_to(parent).as_posix() for p in parent.rglob('*') if p.is_file()}!=seen:
        raise ValueError('Unexpected or omitted v2 parent evidence')
    value=json.loads((parent/'teacher-collection.json').read_text());expected=[r['id'] for r in plan['cases'][:6]]
    if (value['schema']!=previous.SCHEMA or value['status']!='incomplete_failed_preserved_no_retry'
            or value['code_hashes']!=previous.code_hashes() or value['attempted_new_ids']!=previous.REPAIR
            or [r['id'] for r in value['recordings']]!=expected
            or value['source_manifest_sha256']!=sha(source/'source-audio.json')
            or value['source_quality_sha256']!=sha(quality)):
        raise ValueError('Exact immutable v2 recovery lineage required')
    scenes={s['id']:s for s in manifest['scenes']};rows={r['id']:r for r in plan['cases']}
    for record in value['recordings']:
        identity=record['id'];folder=parent/identity
        if identity in REUSE:
            implementation=previous.recording_implementation(record)
            implementation.verify_record(source,scenes[identity],folder,record,tokenizer)
            if not case_review(rows[identity],json.loads((folder/'lifecycle.json').read_text()))['structural_pass']:
                raise ValueError('Preserved original has an undiagnosed lifecycle failure')
        elif identity in REPAIR:
            report=json.loads((folder/'report.json').read_text());ledger=json.loads((folder/'lifecycle.json').read_text())
            tasks=list(ledger['tasks'].values())
            if (record['transport_complete'] is not False or len(tasks)!=2
                    or not tasks[1]['backend_started_at_s']<tasks[0]['wire_write_completed_at_s']<tasks[1]['provider_finished_at_s']
                    or tasks[1]['phase']!='failed_preserved' or tasks[0]['cancelled']
                    or not any(x.get('type')=='function_call' and x.get('name')=='cancel_reasoning' for x in tasks[1]['response']['output'])
                    or not any(x.get('type')=='backend_failure' and x.get('error_type')=='ValueError' for x in report['errors'])):
                raise ValueError('Actual diagnosed pending-decision cancellation race changed')
    return value,manifest,plan


def verify(source,quality,batch,engine,tokenizer_path):
    tokenizer=original.geometry(engine,tokenizer_path)
    parent,manifest,plan=verify_parent(source,quality,batch/'parent-v2',batch/'parent-inventory.json',tokenizer)
    value=json.loads((batch/'teacher-collection.json').read_text());ids=[s['id'] for s in manifest['scenes']]
    expected_new=REPAIR+[SENTINEL]+[i for i in ids if i not in REUSE+REPAIR+[SENTINEL]]
    if (value['schema']!=SCHEMA or value['status']!='raw_complete_unapproved' or value['code_hashes']!=code_hashes()
            or value['source_manifest_sha256']!=sha(source/'source-audio.json') or value['source_quality_sha256']!=sha(quality)
            or value['parent_collection_sha256']!=PARENT_SHA or value['parent_inventory_sha256']!=PARENT_INVENTORY_SHA
            or value['reused_ids']!=REUSE or value['new_attempt_order']!=expected_new
            or value['attempted_new_ids']!=expected_new or [r['id'] for r in value['recordings']]!=ids
            or value['maximum_new_sessions']!=27 or value['maximum_concurrent_sessions']!=2
            or value['maximum_aggregate_voice_attempts']!=35 or value['maximum_aggregate_backend_requests']!=192
            or value['backend_model']!='gpt-6-luna'
            or value['prior_voice_attempts']!=8 or value['prior_backend_requests']!=9
            or value['passing_subset_accepted'] is not False or value['automatic_retry'] is not False or value['training_approved'] is not False
            or value.get('failures')):raise ValueError('Exact whole32 recovered collection required; no passing subset')
    parents={r['id']:r for r in parent['recordings']}
    for scene,row,record in zip(manifest['scenes'],plan['cases'],value['recordings']):
        folder=batch/scene['id'];implementation=recording_implementation(record)
        if scene['id'] in REUSE:
            expected=dict(parents[scene['id']],reused_in_v3=True,
                          parent_v2_record_sha256=canonical_hash(parents[scene['id']]))
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
        out.mkdir(parents=True,exist_ok=False);shutil.copytree(parent,out/'parent-v2');shutil.copyfile(inventory,out/'parent-inventory.json')
        code=code_hashes();value=dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),
            repo_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_hashes=code,
            source_manifest_sha256=sha(source/'source-audio.json'),source_quality_sha256=sha(quality),
            parent_collection_sha256=PARENT_SHA,parent_inventory_sha256=PARENT_INVENTORY_SHA,
            reused_ids=REUSE,new_attempt_order=order,attempted_new_ids=[],recordings=[],maximum_new_sessions=27,
            maximum_concurrent_sessions=2,prior_voice_attempts=8,maximum_aggregate_voice_attempts=35,
            backend_model='gpt-6-luna',prior_backend_requests=9,maximum_aggregate_backend_requests=192,automatic_retry=False,
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
            value['recordings'].append(dict(record,reused_in_v3=True,parent_v2_record_sha256=canonical_hash(record)))
        save();stop=False;semaphore=asyncio.Semaphore(2)
        async def one(identity):
            nonlocal stop
            async with semaphore:
                if stop:return
                value['attempted_new_ids'].append(identity);save()
                try:
                    record=await revised.capture_one(source,scenes[identity],rows[identity],out/identity,sha(quality),
                                                      'gpt-6-luna',env_file,tokenizer)
                    record.update(recorder_version='v3',reused_parent_record=False,reused_in_v3=False)
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
            value['status']='repairing_diagnosed_pending_decision_race';save()
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
