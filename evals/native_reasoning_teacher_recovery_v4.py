"""Whole32 capture after one measured source-pacing repair.

Six recordings retain original source/quality/recorder lineage. Only the failed
timing scene and25 unattempted scenes use V4 source. Every scene remains required.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from evals import native_reasoning_teacher as original
from evals import native_reasoning_teacher_recovery_v3 as previous
from evals import native_reasoning_teacher_v4 as revised
from evals.native_control_teacher import atomic_json, collection_lock
from evals.native_reasoning_lifecycle import canonical_hash
from evals.native_reasoning_source import sha
from evals.native_reasoning_source_pacing import verify as source_manifest
from evals.native_reasoning_source_quality_v4 import verify as source_quality
from evals.native_reasoning_teacher_lifecycle_review import case_review

SCHEMA = 'dl-native-reasoning32-teacher-recovery-v4'
PARENT_SHA = '09bb11d17cc0cacfeb3d2c86d3855519571c658c4f8ba0dd86cdd8515488ff60'
INVENTORY_SHA = 'e5b43d32a309a2d041b6f355ae646dd921471dd5b95bb90197b77ced5e710137'
SOURCE_SHA = '3bccef7ba53835efc6c5be8b5d490551a2ad31bd5d3875d1cf0692e2d9f0254c'
QUALITY_SHA = 'e9f0296a5fcddd892556bfa409ef692b5f51956c76335f1d24e7b4691781ca0d'
REUSE = [f'native-reasoning4-train-{i:03d}'for i in range(6)]
SENTINEL = 'native-reasoning4-train-013'


def code_hashes():
    result=dict(previous.code_hashes(),**revised.code_hashes())
    result[Path(__file__).name]=sha(Path(__file__))
    return result


def case_binding(old_manifest, new_manifest, identity):
    old=next(s for s in old_manifest['scenes']if s['id']==identity)
    new=next(s for s in new_manifest['scenes']if s['id']==identity)
    before={r['object_id']:r for r in old_manifest['objects']}
    after={r['object_id']:r for r in new_manifest['objects']}
    if old != new or any(before[c['object_id']]['wav_sha256'] != after[c['object_id']]['wav_sha256'] for c in old['clips']):
        raise ValueError('Reused teacher requires identical original case words and PCM sources')
    return old


def verify_parent(source, quality, parent, inventory, tokenizer):
    source_quality(source,quality); manifest,plan=source_manifest(source)
    old_source=source/'parent-v3'; old_quality=source/'parent-quality-v3/source-quality.json'
    _,old_manifest,old_plan=previous.verify_parent(old_source,old_quality,parent/'parent-v2',parent/'parent-inventory.json',tokenizer)
    if (sha(source/'source-audio.json') != SOURCE_SHA or sha(quality) != QUALITY_SHA
            or sha(parent/'teacher-collection.json') != PARENT_SHA or sha(inventory) != INVENTORY_SHA
            or old_plan != plan):
        raise ValueError('Exact independently approved V4 source and diagnosed V3 parent required')
    seen=set()
    for row in json.loads(inventory.read_text())['files']:
        name=Path(row['path']);p=parent/name
        if (name.is_absolute() or '..'in name.parts or name.as_posix()in seen or p.is_symlink()
                or p.stat().st_size != row['bytes'] or sha(p) != row['sha256']):
            raise ValueError('Changed original raw teacher evidence')
        seen.add(name.as_posix())
    if {p.relative_to(parent).as_posix()for p in parent.rglob('*')if p.is_file()} != seen:
        raise ValueError('Unexpected or omitted parent files')
    value=json.loads((parent/'teacher-collection.json').read_text())
    if (value['schema'] != previous.SCHEMA or value['status'] != 'incomplete_failed_preserved_no_retry'
            or value['code_hashes'] != previous.code_hashes()
            or value['attempted_new_ids'] != ['native-reasoning4-train-002',SENTINEL]
            or [r['id']for r in value['recordings']] != REUSE+[SENTINEL]
            or value['source_manifest_sha256'] != sha(old_source/'source-audio.json')
            or value['source_quality_sha256'] != sha(old_quality)):
        raise ValueError('Original V3 capture/code/source associations changed')
    rows={r['id']:r for r in plan['cases']};scenes={s['id']:s for s in old_manifest['scenes']}
    for record in value['recordings']:
        identity=record['id']; folder=parent/identity
        report=previous.recording_implementation(record).verify_record(old_source,scenes[identity],folder,record,tokenizer)
        if identity in REUSE:
            case_binding(old_manifest,manifest,identity)
            if (not record['frame_count_diagnostic_pass']
                    or not case_review(rows[identity],json.loads((folder/'lifecycle.json').read_text()))['structural_pass']):
                raise ValueError('Reused original has an undiagnosed lifecycle/frame failure')
        elif (report['native_frame_budget']['full_measured_prefix_live_and_span_frames'] != 1125
              or report['native_frame_budget']['unchanged_frame_limit'] != 1100
              or report['native_frame_budget']['frame_count_diagnostic_pass'] is not False):
            raise ValueError('Only the independently preserved full-frame failure may be recaptured')
    return value,manifest,plan,old_manifest


def verify(source,quality,batch,engine,tokenizer_path):
    tokenizer=original.geometry(engine,tokenizer_path)
    parent,manifest,plan,old_manifest=verify_parent(source,quality,batch/'parent-v3',batch/'parent-inventory.json',tokenizer)
    value=json.loads((batch/'teacher-collection.json').read_text());ids=[s['id']for s in manifest['scenes']]
    order=[SENTINEL]+[i for i in ids if i not in REUSE+[SENTINEL]]
    if (value['schema'] != SCHEMA or value['status'] != 'raw_complete_unapproved'
            or value['code_hashes'] != code_hashes() or value['source_manifest_sha256'] != SOURCE_SHA
            or value['source_quality_sha256'] != QUALITY_SHA or value['parent_collection_sha256'] != PARENT_SHA
            or value['parent_inventory_sha256'] != INVENTORY_SHA or value['reused_ids'] != REUSE
            or value['new_attempt_order'] != order or value['attempted_new_ids'] != order
            or [r['id']for r in value['recordings']] != ids or value['backend_model'] != 'gpt-6-luna'
            or tuple(value[k]for k in ('maximum_new_sessions','maximum_concurrent_sessions','prior_voice_attempts',
                                       'maximum_aggregate_voice_attempts','prior_backend_requests','maximum_aggregate_backend_requests')) != (26,2,10,36,12,192)
            or any(value[k]is not False for k in ('automatic_retry','passing_subset_accepted','training_approved'))
            or value.get('failures')):
        raise ValueError('Whole32 mixed-lineage actual collection required; no passing subset')
    parents={r['id']:r for r in parent['recordings']};rows={r['id']:r for r in plan['cases']}
    for scene,record in zip(manifest['scenes'],value['recordings']):
        identity=scene['id'];folder=batch/identity
        if identity in REUSE:
            expected=dict(parents[identity],reused_in_v4=True,original_v3_record_sha256=canonical_hash(parents[identity]))
            if record != expected:raise ValueError('Original recorder receipt/lineage changed')
            old_scene=case_binding(old_manifest,manifest,identity)
            previous.recording_implementation(record).verify_record(source/'parent-v3',old_scene,folder,record,tokenizer)
            expected_quality=sha(source/'parent-quality-v3/source-quality.json')
        else:
            if record['recorder_version'] != 'v4' or record['reused_in_v4']is not False:
                raise ValueError('New recording must use explicit V4 recorder/source')
            revised.verify_record(source,scene,folder,record,tokenizer);expected_quality=QUALITY_SHA
        env=json.loads((folder/'environment.json').read_text());report=json.loads((folder/'report.json').read_text())
        if (env['source_quality_sha256'] != expected_quality or not report['native_frame_budget']['frame_count_diagnostic_pass']
                or not case_review(rows[identity],json.loads((folder/'lifecycle.json').read_text()))['structural_pass']):
            raise ValueError('Actual source quality, lifecycle or complete native frame bound failed')
    if (sha(batch/'reference-packing.json') != value['reference_packing_sha256']
            or json.loads((batch/'reference-packing.json').read_text()) != previous.reference_packing(source,manifest,plan,tokenizer)):
        raise ValueError('Predeclared full-source packing diagnostic changed')
    return value,manifest,plan


async def recover(source,quality,parent,inventory,out,env_file,engine,tokenizer_path):
    tokenizer=original.geometry(engine,tokenizer_path)
    old,manifest,plan,_=verify_parent(source,quality,parent,inventory,tokenizer)
    scenes={s['id']:s for s in manifest['scenes']};rows={r['id']:r for r in plan['cases']}
    order=[SENTINEL]+[i for i in scenes if i not in REUSE+[SENTINEL]]
    with collection_lock(out.parent/'.private/native-reasoning32-teacher.lock'):
        out.mkdir(parents=True,exist_ok=False);shutil.copytree(parent,out/'parent-v3');shutil.copyfile(inventory,out/'parent-inventory.json')
        code=code_hashes();value=dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),
            repo_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_hashes=code,
            source_manifest_sha256=SOURCE_SHA,source_quality_sha256=QUALITY_SHA,parent_collection_sha256=PARENT_SHA,
            parent_inventory_sha256=INVENTORY_SHA,reused_ids=REUSE,new_attempt_order=order,attempted_new_ids=[],
            recordings=[],maximum_new_sessions=26,maximum_concurrent_sessions=2,prior_voice_attempts=10,
            maximum_aggregate_voice_attempts=36,backend_model='gpt-6-luna',prior_backend_requests=12,
            maximum_aggregate_backend_requests=192,automatic_retry=False,passing_subset_accepted=False,
            training_approved=False,status='preparing_versioned_recovery')
        packing=previous.reference_packing(source,manifest,plan,tokenizer)
        atomic_json(out/'reference-packing.json',packing);value['reference_packing_sha256']=sha(out/'reference-packing.json')
        target=out/'teacher-collection.json'
        def save():atomic_json(target,value)
        for record in old['recordings']:
            if record['id']in REUSE:
                shutil.copytree(parent/record['id'],out/record['id'])
                value['recordings'].append(dict(record,reused_in_v4=True,original_v3_record_sha256=canonical_hash(record)))
        save();stop=False;semaphore=asyncio.Semaphore(2)
        async def one(identity):
            nonlocal stop
            async with semaphore:
                if stop:return
                value['attempted_new_ids'].append(identity);save()
                try:
                    record=await revised.capture_one(source,scenes[identity],rows[identity],out/identity,QUALITY_SHA,
                                                      'gpt-6-luna',env_file,tokenizer)
                    record.update(recorder_version='v4',reused_in_v4=False)
                    value['recordings'].append(record);value['recordings'].sort(key=lambda r:r['id']);save()
                    if not record['transport_complete']:raise ValueError('Actual V4 transport failed')
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
            value['status']='checking_repaired_full_frame_case';save();await one(SENTINEL)
            if not stop:
                value['status']='collecting_remaining_unattempted_scenes';save()
                await asyncio.gather(*(one(i)for i in order[1:]))
        finally:
            value['status']='raw_complete_unapproved' if not stop and len(value['recordings'])==32 else 'incomplete_failed_preserved_no_retry'
            if code_hashes() != code:value['status']='capture_code_changed_failed'
            save()
        if value['status']=='raw_complete_unapproved':verify(source,quality,out,engine,tokenizer_path)
        return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','quality','parent','parent-inventory','out','env-file','engine','tokenizer'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=asyncio.run(recover(a.source,a.quality,a.parent,a.parent_inventory,a.out,a.env_file,a.engine,a.tokenizer))
    print(json.dumps(dict(status=r['status'],recordings=len(r['recordings']),training_approved=False)))
    if r['status'] != 'raw_complete_unapproved':raise SystemExit(1)
