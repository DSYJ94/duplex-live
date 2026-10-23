"""One actual backend-only replay of the preserved failed action decision.

Reconstructs the measured pre-delivery state explicitly. No new native voice,
audio modification, source oracle payload, wire publication or training approval.
"""
import argparse
import asyncio
import copy
import json
from pathlib import Path
import time

from evals.native_control_teacher import atomic_json, collection_lock
from evals.native_reasoning_backend import request
from evals.native_reasoning_lifecycle import canonical_hash
from evals.native_reasoning_lifecycle_v3 import NativeReasoningLifecycle
from evals.native_reasoning_source import sha
from evals.native_reasoning_source_quality_v4 import verify as source_quality
from evals.native_reasoning_teacher import geometry
from evals.native_reasoning_teacher_v4 import verify_record
from evals.native_reasoning_teacher_lifecycle_review import case_review
from evals.native_reasoning_source_pacing import verify as source_manifest

PARENT_SHA='96f9be42ddb39133c08cb4e7d21ea14fec909807def44ba68f07f12d5d9a3f6a'
INVENTORY_SHA='ebc4d86793b2112151b2a0bd52691024160d6b0612ed9b2777fcbd3b425c4eb3'
CASE='native-reasoning4-train-019'


def preflight(source,quality,parent,inventory,engine,tokenizer):
    source_quality(source,quality);manifest,plan=source_manifest(source)
    if sha(parent/'teacher-collection.json') != PARENT_SHA or sha(inventory) != INVENTORY_SHA:
        raise ValueError('Exact independently preserved failed V4 collection required')
    seen=set()
    for row in json.loads(inventory.read_text())['files']:
        p=Path(row['path']);file=parent/p
        if (p.is_absolute() or '..'in p.parts or p.as_posix()in seen or file.is_symlink()
                or file.stat().st_size != row['bytes'] or sha(file) != row['sha256']):
            raise ValueError('Changed actual capture evidence')
        seen.add(p.as_posix())
    if {p.relative_to(parent).as_posix()for p in parent.rglob('*')if p.is_file()} != seen:
        raise ValueError('Original capture inventory differs')
    collection=json.loads((parent/'teacher-collection.json').read_text())
    if (collection['status'] != 'incomplete_failed_preserved_no_retry' or len(collection['recordings']) != 22
            or [r['id']for r in collection['failures']] != [CASE]):
        raise ValueError('Exact diagnosed failure required')
    scene=next(s for s in manifest['scenes']if s['id']==CASE)
    row=next(r for r in plan['cases']if r['id']==CASE)
    record=next(r for r in collection['recordings']if r['id']==CASE)
    verify_record(source,scene,parent/CASE,record,geometry(engine,tokenizer))
    ledger=json.loads((parent/CASE/'lifecycle.json').read_text());ids=list(ledger['tasks'])
    if (len(ids) != 2 or case_review(row,ledger)['structural_pass']
            or ledger['tasks'][ids[1]]['receipt']['operation'] != 'respond_to_user'
            or not ledger['tasks'][ids[0]]['provider_finished_at_s'] < ledger['tasks'][ids[1]]['backend_started_at_s']
            < ledger['tasks'][ids[1]]['provider_finished_at_s'] < ledger['tasks'][ids[0]]['wire_write_completed_at_s']):
        raise ValueError('Actual completed-but-held original and misclassified correction required')
    return row,ledger


def restored_world(ledger):
    """Explicit frozen-state replay; never pretend this is a new native event."""
    old_id,new_id=list(ledger['tasks']);old=copy.deepcopy(ledger['tasks'][old_id]);new=ledger['tasks'][new_id]
    origin=time.monotonic();offset=new['native_request_observed_at_s']
    world=NativeReasoningLifecycle(lambda:offset+time.monotonic()-origin,0.,maximum_requests=2)
    old['phase']='result_ready';old.pop('wire_write_completed_at_s');world.tasks[old_id]=old
    return world,old_id,new_id,new['context']


async def replay(source,quality,parent,inventory,engine,tokenizer,out,env_file):
    row,ledger=preflight(source,quality,parent,inventory,engine,tokenizer)
    with collection_lock(out.parent/'.private/native-reasoning32-action-replay.lock'):
        out.mkdir(parents=True,exist_ok=False)
        value=dict(schema='dl-native-reasoning32-backend-action-replay-v1',phase='prepared_unapproved_component',
            code_sha256=sha(Path(__file__)),lifecycle_code_sha256=sha(Path(__file__).with_name('native_reasoning_lifecycle_v3.py')),
            parent_collection_sha256=PARENT_SHA,parent_inventory_sha256=INVENTORY_SHA,
            original_failed_case=CASE,original_ledger_sha256=sha(parent/CASE/'lifecycle.json'),
            state_reconstruction='Actual pre-delivery snapshot explicitly reconstructed from measured original clocks',
            maximum_new_API_requests=1,prior_backend_requests=32,maximum_aggregate_backend_requests=33,
            native_voice_sessions=0,wire_writes=0,new_native_request_selection_proven=False,
            source_oracles_or_case_annotations_sent=False,automatic_retry=False,training_approved=False,
            whole32_teacher_approved=False)
        atomic_json(out/'replay.json',value)
        world=None
        try:
            world,old_id,new_id,context=restored_world(ledger)
            await world.reserve_native_request(new_id,context,minimum_delivery_delay_s=0.)
            receipt=await request(world,new_id,'gpt-6-luna',out/'backend',env_file)
            atomic_json(out/'lifecycle.json',world.evidence())
            if receipt is None or receipt['actual_transport'] is not True:
                raise ValueError('Actual provider response required; fixture cannot qualify')
            diagnostic=case_review(row,world.evidence());atomic_json(out/'lifecycle-diagnostic.json',diagnostic)
            value.update(phase='complete_component_only',actual_new_API_requests=1,
                         actual_model_selected_operation=receipt['operation'],
                         original_retired=world.tasks[old_id]['cancelled'],
                         independent_local_oracle_and_lifecycle_pass=diagnostic['structural_pass'],
                         actual_provider_response_sha256=sha(out/'backend/response.json'),
                         actual_payload_sha256=sha(out/'backend/request.json'),
                         actual_payload_canonical_sha256=canonical_hash(json.loads((out/'backend/request.json').read_text())))
        except BaseException as error:
            if world is not None:atomic_json(out/'lifecycle.json',world.evidence())
            value.update(phase='failed_preserved_no_retry',error_type=type(error).__name__)
            atomic_json(out/'replay.json',value);raise
        atomic_json(out/'replay.json',value)
        print(json.dumps(value),flush=True)
        return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','quality','parent','inventory','engine','tokenizer','out','env-file'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();r=asyncio.run(replay(a.source,a.quality,a.parent,a.inventory,a.engine,a.tokenizer,a.out,a.env_file))
    if not r.get('independent_local_oracle_and_lifecycle_pass'):raise SystemExit(1)
