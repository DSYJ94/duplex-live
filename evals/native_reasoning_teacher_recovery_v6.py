"""One documented neural-delegation prompt experiment; retain all27 prior episodes.

Only026 is recaptured before the four unattempted scenes. No runtime forces a
request, inspects intent, retries a voice session, or approves a corpus subset.
"""
import argparse
import asyncio
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess

from evals import native_reasoning_teacher as original
from evals import native_reasoning_teacher_recovery_v5 as previous
from evals import native_reasoning_teacher_v5 as old_recorder
from evals import native_reasoning_teacher_v6 as revised
from evals.native_control_teacher import atomic_json,collection_lock
from evals.native_reasoning_lifecycle import canonical_hash
from evals.native_reasoning_source import sha
from evals.native_reasoning_teacher_lifecycle_review import case_review
from evals.native_reasoning_teacher_recovery import reference_packing

SCHEMA='dl-native-reasoning32-teacher-recovery-v6'
PARENT_SHA='cfdff0924bbec7364fb084c37ce9ff0cb70994c7b98f9c442a44d63c7ccacc3d'
INVENTORY_SHA='e4810b9974186214313065d908902d1d985e5d71851e8d581d6779a24f9d527b'
SOURCE_SHA=previous.previous.SOURCE_SHA
QUALITY_SHA=previous.previous.QUALITY_SHA
SENTINEL='native-reasoning4-train-026'
REUSE=[f'native-reasoning4-train-{i:03d}'for i in range(28)if i!=26]


def code_hashes():
    return dict(previous.code_hashes(),**revised.code_hashes(),**{Path(__file__).name:sha(Path(__file__))})


def old_readback(source,scene,folder,record,tokenizer):
    if record['reused_in_v5']is True:
        return previous.old_readback(source,scene,folder,record,tokenizer)
    if record['recorder_version']!='v5':raise ValueError('Explicit original V5 recorder required')
    return old_recorder.verify_record(source,scene,folder,record,tokenizer)


def verify_parent(source,quality,parent,inventory,tokenizer):
    _,manifest,plan=previous.verify_parent(source,quality,parent/'parent-v4',parent/'parent-inventory.json',tokenizer)
    previous.component_gate(parent/'backend-component',parent/'backend-component-review.json')
    if sha(parent/'teacher-collection.json')!=PARENT_SHA or sha(inventory)!=INVENTORY_SHA:
        raise ValueError('Exact preserved native-request failure and complete V5 inventory required')
    seen=set()
    for row in json.loads(inventory.read_text())['files']:
        name=Path(row['path']);p=parent/name
        if (name.is_absolute()or '..'in name.parts or name.as_posix()in seen or p.is_symlink()
                or p.stat().st_size!=row['bytes']or sha(p)!=row['sha256']):raise ValueError('Changed parent evidence')
        seen.add(name.as_posix())
    if {p.relative_to(parent).as_posix()for p in parent.rglob('*')if p.is_file()}!=seen:
        raise ValueError('Unexpected or omitted V5 evidence')
    value=json.loads((parent/'teacher-collection.json').read_text())
    if (value['schema']!=previous.SCHEMA or value['status']!='incomplete_failed_preserved_no_retry'
            or value['code_hashes']!=previous.code_hashes()
            or [r['id']for r in value['recordings']]!=sorted(REUSE+[SENTINEL])
            or [r['id']for r in value['failures']]!=[SENTINEL]
            or value['source_manifest_sha256']!=SOURCE_SHA or value['source_quality_sha256']!=QUALITY_SHA):
        raise ValueError('Exact actual V5 recording/source/code lineage required')
    scenes={s['id']:s for s in manifest['scenes']};rows={r['id']:r for r in plan['cases']}
    for record in value['recordings']:
        identity=record['id'];folder=parent/identity
        report=old_readback(source,scenes[identity],folder,record,tokenizer)
        ledger=json.loads((folder/'lifecycle.json').read_text());check=case_review(rows[identity],ledger)
        if not report['native_frame_budget']['frame_count_diagnostic_pass']:raise ValueError('Unexpected parent frame failure')
        if identity in REUSE:
            if not check['structural_pass']:raise ValueError('Undiagnosed original lifecycle failure')
        elif (len(ledger['tasks'])!=1 or report['delegations']!=1 or check['reasons']!=[
                'Missing actual committed cancel_reasoning for original request',
                'Original result was delivered despite required pending cancellation/replacement']):
            raise ValueError('Only the measured missing-native-request failure may be recaptured')
    return value,manifest,plan


def verify(source,quality,batch,engine,tokenizer_path):
    tokenizer=original.geometry(engine,tokenizer_path)
    parent,manifest,plan=verify_parent(source,quality,batch/'parent-v5',batch/'parent-inventory.json',tokenizer)
    ids=[s['id']for s in manifest['scenes']];order=[SENTINEL]+[i for i in ids if i not in REUSE+[SENTINEL]]
    value=json.loads((batch/'teacher-collection.json').read_text())
    if (value['schema']!=SCHEMA or value['status']!='raw_complete_unapproved' or value['code_hashes']!=code_hashes()
            or value['source_manifest_sha256']!=SOURCE_SHA or value['source_quality_sha256']!=QUALITY_SHA
            or value['parent_collection_sha256']!=PARENT_SHA or value['parent_inventory_sha256']!=INVENTORY_SHA
            or value['reused_ids']!=REUSE or value['new_attempt_order']!=order or value['attempted_new_ids']!=order
            or [r['id']for r in value['recordings']]!=ids or value['backend_model']!='gpt-6-luna'
            or tuple(value[k]for k in ('maximum_new_sessions','maximum_concurrent_sessions','prior_voice_attempts',
                'maximum_aggregate_voice_attempts','prior_backend_requests','maximum_aggregate_backend_requests'))!=(5,2,33,38,43,192)
            or any(value[k]is not False for k in ('automatic_retry','passing_subset_accepted','training_approved'))
            or value.get('failures')):raise ValueError('Complete actual32 with original lineage required; no subset')
    parents={r['id']:r for r in parent['recordings']};rows={r['id']:r for r in plan['cases']}
    for scene,record in zip(manifest['scenes'],value['recordings']):
        identity=scene['id'];folder=batch/identity
        if identity in REUSE:
            if record!=dict(parents[identity],reused_in_v6=True,parent_v5_record_sha256=canonical_hash(parents[identity])):
                raise ValueError('Reused raw record/version changed')
            report=old_readback(source,scene,folder,record,tokenizer)
        else:
            if record['recorder_version']!='v6'or record['reused_in_v6']is not False:raise ValueError('New V6 recorder required')
            report=revised.verify_record(source,scene,folder,record,tokenizer)
            if json.loads((folder/'environment.json').read_text())['source_quality_sha256']!=QUALITY_SHA:
                raise ValueError('Source approval changed')
        if (not report['native_frame_budget']['frame_count_diagnostic_pass']or
                not case_review(rows[identity],json.loads((folder/'lifecycle.json').read_text()))['structural_pass']):
            raise ValueError('Actual complete frame/lifecycle check failed')
    if (sha(batch/'reference-packing.json')!=value['reference_packing_sha256']or
        json.loads((batch/'reference-packing.json').read_text())!=reference_packing(source,manifest,plan,tokenizer)):
        raise ValueError('Complete source packing diagnostic changed')
    return value,manifest,plan


async def recover(source,quality,parent,inventory,out,env_file,engine,tokenizer_path):
    tokenizer=original.geometry(engine,tokenizer_path);old,manifest,plan=verify_parent(source,quality,parent,inventory,tokenizer)
    scenes={s['id']:s for s in manifest['scenes']};rows={r['id']:r for r in plan['cases']}
    order=[SENTINEL]+[i for i in scenes if i not in REUSE+[SENTINEL]]
    with collection_lock(out.parent/'.private/native-reasoning32-teacher.lock'):
        out.mkdir(parents=True,exist_ok=False);shutil.copytree(parent,out/'parent-v5');shutil.copyfile(inventory,out/'parent-inventory.json')
        code=code_hashes();value=dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),
            repo_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_hashes=code,
            source_manifest_sha256=SOURCE_SHA,source_quality_sha256=QUALITY_SHA,
            parent_collection_sha256=PARENT_SHA,parent_inventory_sha256=INVENTORY_SHA,
            reused_ids=REUSE,new_attempt_order=order,attempted_new_ids=[],recordings=[],maximum_new_sessions=5,
            maximum_concurrent_sessions=2,prior_voice_attempts=33,maximum_aggregate_voice_attempts=38,
            backend_model='gpt-6-luna',prior_backend_requests=43,maximum_aggregate_backend_requests=192,
            automatic_retry=False,passing_subset_accepted=False,training_approved=False,status='preparing_versioned_recovery')
        atomic_json(out/'reference-packing.json',reference_packing(source,manifest,plan,tokenizer))
        value['reference_packing_sha256']=sha(out/'reference-packing.json');target=out/'teacher-collection.json'
        def save():atomic_json(target,value)
        for record in old['recordings']:
            if record['id']in REUSE:
                shutil.copytree(parent/record['id'],out/record['id'])
                value['recordings'].append(dict(record,reused_in_v6=True,parent_v5_record_sha256=canonical_hash(record)))
        save();stop=False;semaphore=asyncio.Semaphore(2)
        async def one(identity):
            nonlocal stop
            async with semaphore:
                if stop:return
                value['attempted_new_ids'].append(identity);save()
                try:
                    record=await revised.capture_one(source,scenes[identity],rows[identity],out/identity,QUALITY_SHA,'gpt-6-luna',env_file,tokenizer)
                    record.update(recorder_version='v6',reused_in_v6=False);value['recordings'].append(record)
                    value['recordings'].sort(key=lambda r:r['id']);save()
                    if not record['transport_complete']:raise ValueError('Actual V6 transport failed')
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
            value['status']='checking_documented_neural_delegation_prompt';save();await one(SENTINEL)
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
    for n in ('source','quality','parent','parent-inventory','out','env-file','engine','tokenizer'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();r=asyncio.run(recover(a.source,a.quality,a.parent,a.parent_inventory,a.out,a.env_file,a.engine,a.tokenizer))
    print(json.dumps(dict(status=r['status'],recordings=len(r['recordings']),training_approved=False)))
    if r['status']!='raw_complete_unapproved':raise SystemExit(1)
