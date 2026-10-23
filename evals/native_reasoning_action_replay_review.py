"""Independent backend-only readback; never excuse a missing voice episode.

Recomputes the exact request, semantic state transition and receipt from actual
provider bytes at the recorded clocks. Whole-voice diagnostics stay failed.
"""
import argparse
import asyncio
import copy
import json
from pathlib import Path

from evals.native_control_teacher import atomic_json
from evals.native_reasoning_action_replay import preflight,CASE
from evals.native_reasoning_lifecycle import canonical_hash
from evals.native_reasoning_lifecycle_v3 import NativeReasoningLifecycle
from evals.native_reasoning_teacher_lifecycle_review import case_review
from evals.native_reasoning_source import sha


async def transition(parent,observed,payload,response):
    old_id,new_id=list(parent['tasks']);old=copy.deepcopy(parent['tasks'][old_id])
    current=observed['tasks'][new_id];clock=[current['native_request_observed_at_s']]
    world=NativeReasoningLifecycle(lambda:clock[0],0.,maximum_requests=2)
    old['phase']='result_ready';old.pop('wire_write_completed_at_s');world.tasks[old_id]=old
    await world.reserve_native_request(new_id,parent['tasks'][new_id]['context'],minimum_delivery_delay_s=0.)
    clock[0]=current['backend_started_at_s']
    expected=await world.start_backend(new_id,'gpt-6-luna')
    if payload != expected:raise ValueError('Replay request differs from exact frozen state and generic action contract')
    clock[0]=current['provider_finished_at_s']
    await world.complete_backend(new_id,response,actual_transport=True)
    if canonical_hash(world.tasks) != canonical_hash(observed['tasks']):
        raise ValueError('Recorded semantic state/receipt differs from independent recomputation')
    return world


async def review(source,quality,parent,inventory,engine,tokenizer,replay,out):
    row,original=preflight(source,quality,parent,inventory,engine,tokenizer)
    record=json.loads((replay/'replay.json').read_text());trace=json.loads((replay/'backend/transport.json').read_text())
    observed=json.loads((replay/'lifecycle.json').read_text());payload=json.loads((replay/'backend/request.json').read_text())
    response=json.loads((replay/'backend/response.json').read_text())
    if (record['phase'] != 'complete_component_only' or record['native_voice_sessions'] != 0 or record['wire_writes'] != 0
            or record['actual_new_API_requests'] != 1 or record['maximum_new_API_requests'] != 1
            or record['code_sha256'] != sha(Path(__file__).with_name('native_reasoning_action_replay.py'))
            or record['lifecycle_code_sha256'] != sha(Path(__file__).with_name('native_reasoning_lifecycle_v3.py'))
            or record['original_ledger_sha256'] != sha(parent/CASE/'lifecycle.json')
            or trace['phase'] != 'semantic_action_committed' or trace['actual_provider_transport'] is not True
            or trace['http_status'] != 200 or not trace['provider_request_id'] or trace['maximum_attempts'] != 1
            or trace['automatic_retry'] is not False or trace['request_payload_sha256'] != canonical_hash(payload)
            or trace['response_sha256'] != sha(replay/'backend/response.json')
            or record['actual_provider_response_sha256'] != sha(replay/'backend/response.json')
            or record['actual_payload_sha256'] != sha(replay/'backend/request.json')
            or record['actual_payload_canonical_sha256'] != canonical_hash(payload)):
        raise ValueError('Actual unchanged one-request component provenance required')
    world=await transition(original,observed,payload,response)
    diagnostic=case_review(row,world.evidence())
    expected_missing_voice=['Committed current status/result never returned to voice teacher']
    if (diagnostic['reasons'] != expected_missing_voice or diagnostic['structural_pass'] is not False
            or any('wire_write_completed_at_s'in t for t in world.tasks.values())
            or json.loads((replay/'lifecycle-diagnostic.json').read_text()) != diagnostic):
        raise ValueError('Component must pass local action/oracle checks while complete voice delivery remains explicitly absent')
    value=dict(schema='dl-native-reasoning32-backend-only-independent-readback-v1',
               replay_record_sha256=sha(replay/'replay.json'),code_sha256=sha(Path(__file__)),
               exact_request_state_and_receipt_recomputed=True,actual_backend_component_action_and_local_oracle_pass=True,
               selected_operation='replace_reasoning',original_delivery_retired=True,
               complete_voice_structural_pass=False,expected_missing_voice_requirement=expected_missing_voice[0],
               slow_barrier_or_native_voice_behavior_proven=False,new_API_calls=0,new_optimizer_updates=0,
               training_approved=False,whole32_teacher_approved=False)
    if out.exists():raise ValueError('Preserve prior component reviews')
    atomic_json(out,value);print(json.dumps(value),flush=True)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','quality','parent','inventory','engine','tokenizer','replay','out'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();asyncio.run(review(a.source,a.quality,a.parent,a.inventory,a.engine,a.tokenizer,a.replay,a.out))
