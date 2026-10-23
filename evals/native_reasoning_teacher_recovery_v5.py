"""Preserve21 original episodes; repair one action-contract failure, then10 new.

No corpus subset is approved. Only the backend action contract changes; the
qualified V4 source, voice policy, pacing, audio limits and retention stay fixed.
"""
import argparse
import asyncio
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess

from evals import native_reasoning_teacher as original
from evals import native_reasoning_teacher_recovery_v4 as previous
from evals import native_reasoning_teacher_v4 as old_recorder
from evals import native_reasoning_teacher_v5 as revised
from evals.native_control_teacher import atomic_json,collection_lock
from evals.native_reasoning_lifecycle import canonical_hash
from evals.native_reasoning_source import sha
from evals.native_reasoning_source_repair import verify as old_source_manifest
from evals.native_reasoning_teacher_lifecycle_review import case_review

SCHEMA='dl-native-reasoning32-teacher-recovery-v5'
PARENT_SHA='96f9be42ddb39133c08cb4e7d21ea14fec909807def44ba68f07f12d5d9a3f6a'
INVENTORY_SHA='ebc4d86793b2112151b2a0bd52691024160d6b0612ed9b2777fcbd3b425c4eb3'
COMPONENT_SHA='e76cc9daafd39a6d66d9ac18347ff138daec7d3681a1774c16a32aecd88ef09c'
COMPONENT_INVENTORY='b8435243cf62e25a8c07dde00cf487ea2c5e5ffca91506e66732bd7a077411b8'
SENTINEL='native-reasoning4-train-019'
REUSE=[f'native-reasoning4-train-{i:03d}'for i in range(22)if i!=19]


def code_hashes():
    result=dict(previous.code_hashes(),**revised.code_hashes())
    for name in (Path(__file__).name,'native_reasoning_action_replay.py','native_reasoning_action_replay_review.py'):
        result[name]=sha(Path(__file__).with_name(name))
    return result


def component_gate(root,approval):
    if root.is_symlink()or any(p.is_symlink()for p in root.rglob('*')):
        raise ValueError('Regular immutable component evidence required')
    files=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
           for p in sorted(root.rglob('*'))if p.is_file()]
    if canonical_hash(files)!=COMPONENT_INVENTORY:raise ValueError('Complete original six-file backend evidence required')
    proof=json.loads(approval.read_text());record=json.loads((root/'replay.json').read_text())
    if (sha(approval)!=COMPONENT_SHA or proof['actual_backend_component_action_and_local_oracle_pass']is not True
            or proof['complete_voice_structural_pass']is not False or proof['training_approved']is not False
            or proof['replay_record_sha256']!=sha(root/'replay.json')
            or proof['code_sha256']!=sha(Path(__file__).with_name('native_reasoning_action_replay_review.py'))
            or record['code_sha256']!=sha(Path(__file__).with_name('native_reasoning_action_replay.py'))
            or record['lifecycle_code_sha256']!=sha(Path(__file__).with_name('native_reasoning_lifecycle_v3.py'))
            or record['actual_provider_response_sha256']!=sha(root/'backend/response.json')
            or record['actual_payload_sha256']!=sha(root/'backend/request.json')
            or record['actual_payload_canonical_sha256']!=canonical_hash(json.loads((root/'backend/request.json').read_text()))
            or record['actual_model_selected_operation']!='replace_reasoning' or record['original_retired']is not True
            or (record['actual_new_API_requests'],record['native_voice_sessions'],record['wire_writes'])!=(1,0,0)):
        raise ValueError('Exact independent backend-only component readback required; no voice approval inferred')


def old_readback(source,scene,folder,record,tokenizer):
    if record['id']in previous.REUSE:
        old_manifest,_=old_source_manifest(source/'parent-v3')
        old_scene=previous.case_binding(old_manifest,previous.source_manifest(source)[0],scene['id'])
        return previous.previous.recording_implementation(record).verify_record(source/'parent-v3',old_scene,folder,record,tokenizer)
    if record['recorder_version']!='v4' or record['reused_in_v4']is not False:
        raise ValueError('Explicit original V4 recorder required')
    return old_recorder.verify_record(source,scene,folder,record,tokenizer)


def verify_parent(source,quality,parent,inventory,tokenizer):
    _,manifest,plan,_=previous.verify_parent(source,quality,parent/'parent-v3',parent/'parent-inventory.json',tokenizer)
    if sha(parent/'teacher-collection.json')!=PARENT_SHA or sha(inventory)!=INVENTORY_SHA:
        raise ValueError('Exact preserved V4 failure and complete inventory required')
    seen=set()
    for row in json.loads(inventory.read_text())['files']:
        name=Path(row['path']);p=parent/name
        if (name.is_absolute()or '..'in name.parts or name.as_posix()in seen or p.is_symlink()
                or p.stat().st_size!=row['bytes']or sha(p)!=row['sha256']):raise ValueError('Changed parent evidence')
        seen.add(name.as_posix())
    if {p.relative_to(parent).as_posix()for p in parent.rglob('*')if p.is_file()}!=seen:
        raise ValueError('Unexpected or omitted V4 files')
    value=json.loads((parent/'teacher-collection.json').read_text())
    if (value['schema']!=previous.SCHEMA or value['status']!='incomplete_failed_preserved_no_retry'
            or value['code_hashes']!=previous.code_hashes()
            or [r['id']for r in value['recordings']]!=sorted(REUSE+[SENTINEL])
            or [r['id']for r in value['failures']]!=[SENTINEL]
            or value['source_manifest_sha256']!=previous.SOURCE_SHA or value['source_quality_sha256']!=previous.QUALITY_SHA):
        raise ValueError('Exact original V4 recording/code/source lineage required')
    scenes={s['id']:s for s in manifest['scenes']};rows={r['id']:r for r in plan['cases']}
    for record in value['recordings']:
        identity=record['id'];folder=parent/identity
        report=old_readback(source,scenes[identity],folder,record,tokenizer)
        check=case_review(rows[identity],json.loads((folder/'lifecycle.json').read_text()))
        if not report['native_frame_budget']['frame_count_diagnostic_pass']:raise ValueError('Unexpected parent frame failure')
        if identity in REUSE:
            if not check['structural_pass']:raise ValueError('Original episode has an undiagnosed lifecycle failure')
        elif check['reasons']!=['Missing actual committed replace_reasoning for original request',
                               'Original result was delivered despite required pending cancellation/replacement']:
            raise ValueError('Only the actual diagnosed replacement-action failure may be recaptured')
    return value,manifest,plan


def verify(source,quality,batch,engine,tokenizer_path):
    tokenizer=original.geometry(engine,tokenizer_path)
    parent,manifest,plan=verify_parent(source,quality,batch/'parent-v4',batch/'parent-inventory.json',tokenizer)
    component_gate(batch/'backend-component',batch/'backend-component-review.json')
    ids=[s['id']for s in manifest['scenes']];order=[SENTINEL]+[i for i in ids if i not in REUSE+[SENTINEL]]
    value=json.loads((batch/'teacher-collection.json').read_text())
    if (value['schema']!=SCHEMA or value['status']!='raw_complete_unapproved' or value['code_hashes']!=code_hashes()
            or value['source_manifest_sha256']!=previous.SOURCE_SHA or value['source_quality_sha256']!=previous.QUALITY_SHA
            or value['parent_collection_sha256']!=PARENT_SHA or value['parent_inventory_sha256']!=INVENTORY_SHA
            or value['backend_component_review_sha256']!=COMPONENT_SHA or value['reused_ids']!=REUSE
            or value['new_attempt_order']!=order or value['attempted_new_ids']!=order
            or [r['id']for r in value['recordings']]!=ids or value['backend_model']!='gpt-6-luna'
            or tuple(value[k]for k in ('maximum_new_sessions','maximum_concurrent_sessions','prior_voice_attempts',
                'maximum_aggregate_voice_attempts','prior_backend_requests','maximum_aggregate_backend_requests'))!=(11,2,26,37,33,192)
            or any(value[k]is not False for k in ('automatic_retry','passing_subset_accepted','training_approved'))
            or value.get('failures')):raise ValueError('Complete unchanged actual32 required; no passing subset')
    parents={r['id']:r for r in parent['recordings']};rows={r['id']:r for r in plan['cases']}
    for scene,record in zip(manifest['scenes'],value['recordings']):
        identity=scene['id'];folder=batch/identity
        if identity in REUSE:
            if record!=dict(parents[identity],reused_in_v5=True,parent_v4_record_sha256=canonical_hash(parents[identity])):
                raise ValueError('Reused raw receipt/record/version altered')
            report=old_readback(source,scene,folder,record,tokenizer)
        else:
            if record['recorder_version']!='v5'or record['reused_in_v5']is not False:raise ValueError('New V5 recorder required')
            report=revised.verify_record(source,scene,folder,record,tokenizer)
            if json.loads((folder/'environment.json').read_text())['source_quality_sha256']!=previous.QUALITY_SHA:
                raise ValueError('Changed source approval')
        if (not report['native_frame_budget']['frame_count_diagnostic_pass']or
                not case_review(rows[identity],json.loads((folder/'lifecycle.json').read_text()))['structural_pass']):
            raise ValueError('Actual complete frame/lifecycle check failed')
    if (sha(batch/'reference-packing.json')!=value['reference_packing_sha256']or
        json.loads((batch/'reference-packing.json').read_text())!=previous.previous.reference_packing(source,manifest,plan,tokenizer)):
        raise ValueError('Original full-source packing changed')
    return value,manifest,plan


async def recover(source,quality,parent,inventory,component,component_review,out,env_file,engine,tokenizer_path):
    tokenizer=original.geometry(engine,tokenizer_path);old,manifest,plan=verify_parent(source,quality,parent,inventory,tokenizer)
    component_gate(component,component_review)
    scenes={s['id']:s for s in manifest['scenes']};rows={r['id']:r for r in plan['cases']}
    order=[SENTINEL]+[i for i in scenes if i not in REUSE+[SENTINEL]]
    with collection_lock(out.parent/'.private/native-reasoning32-teacher.lock'):
        out.mkdir(parents=True,exist_ok=False);shutil.copytree(parent,out/'parent-v4');shutil.copyfile(inventory,out/'parent-inventory.json')
        shutil.copytree(component,out/'backend-component');shutil.copyfile(component_review,out/'backend-component-review.json')
        code=code_hashes();value=dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),
            repo_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_hashes=code,
            source_manifest_sha256=previous.SOURCE_SHA,source_quality_sha256=previous.QUALITY_SHA,
            parent_collection_sha256=PARENT_SHA,parent_inventory_sha256=INVENTORY_SHA,backend_component_review_sha256=COMPONENT_SHA,
            reused_ids=REUSE,new_attempt_order=order,attempted_new_ids=[],recordings=[],maximum_new_sessions=11,
            maximum_concurrent_sessions=2,prior_voice_attempts=26,maximum_aggregate_voice_attempts=37,
            backend_model='gpt-6-luna',prior_backend_requests=33,maximum_aggregate_backend_requests=192,
            automatic_retry=False,passing_subset_accepted=False,training_approved=False,status='preparing_versioned_recovery')
        atomic_json(out/'reference-packing.json',previous.previous.reference_packing(source,manifest,plan,tokenizer))
        value['reference_packing_sha256']=sha(out/'reference-packing.json');target=out/'teacher-collection.json'
        def save():atomic_json(target,value)
        for record in old['recordings']:
            if record['id']in REUSE:
                shutil.copytree(parent/record['id'],out/record['id'])
                value['recordings'].append(dict(record,reused_in_v5=True,parent_v4_record_sha256=canonical_hash(record)))
        save();stop=False;semaphore=asyncio.Semaphore(2)
        async def one(identity):
            nonlocal stop
            async with semaphore:
                if stop:return
                value['attempted_new_ids'].append(identity);save()
                try:
                    record=await revised.capture_one(source,scenes[identity],rows[identity],out/identity,previous.QUALITY_SHA,'gpt-6-luna',env_file,tokenizer)
                    record.update(recorder_version='v5',reused_in_v5=False);value['recordings'].append(record)
                    value['recordings'].sort(key=lambda r:r['id']);save()
                    if not record['transport_complete']:raise ValueError('Actual V5 transport failed')
                    revised.verify_record(source,scenes[identity],out/identity,record,tokenizer)
                    check=case_review(rows[identity],json.loads((out/identity/'lifecycle.json').read_text()))
                    if not record['frame_count_diagnostic_pass']:raise ValueError('Complete native frame budget failed')
                    if not check['structural_pass']:
                        atomic_json(out/(identity+'-failed-lifecycle.json'),check);raise ValueError('Actual lifecycle diagnostic failed')
                    print(json.dumps(dict(id=identity,transport_complete=True,structural_diagnostic_pass=True,complete_recordings=len(value['recordings']),expected=32,training_approved=False)),flush=True)
                except Exception as error:
                    stop=True;value.setdefault('failures',[]).append(dict(id=identity,error_type=type(error).__name__,reason=str(error)))
                    save();print(json.dumps(dict(id=identity,failed_preserved=True,reason=str(error),no_retry=True)),flush=True)
        try:
            value['status']='checking_action_contract_repair';save();await one(SENTINEL)
            if not stop:
                value['status']='collecting_remaining_unattempted_scenes';save();await asyncio.gather(*(one(i)for i in order[1:]))
        finally:
            value['status']='raw_complete_unapproved'if not stop and len(value['recordings'])==32 else'incomplete_failed_preserved_no_retry'
            if code_hashes()!=code:value['status']='capture_code_changed_failed'
            save()
        if value['status']=='raw_complete_unapproved':verify(source,quality,out,engine,tokenizer_path)
        return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','quality','parent','parent-inventory','component','component-review','out','env-file','engine','tokenizer'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();r=asyncio.run(recover(a.source,a.quality,a.parent,a.parent_inventory,a.component,a.component_review,a.out,a.env_file,a.engine,a.tokenizer))
    print(json.dumps(dict(status=r['status'],recordings=len(r['recordings']),training_approved=False)))
    if r['status']!='raw_complete_unapproved':raise SystemExit(1)
