"""Pinned real postupdate failure states; no invented successful window."""
import json,math
from pathlib import Path
from . import matched_guard as g
from .matched_interrupted import DURABLE

AUTH=Path('docs/NATIVE-MOSHI-POSTUPDATE-RECOVERY-AUTHORIZATION-2026-11-10.json')
EVIDENCE=Path('docs/NATIVE-MOSHI-HEARTBEAT-FAILURE-RESULT-2026-11-10.json')
SAVED=DURABLE/'preserved/failure-8260542-v1'
PREDECESSOR='65ea7b83d05450ca52df8ef9710dd35da7437bc3'
MARKER='actual-failed-8260542-v1'
STATES={'personaplex':(89,'be5a6d21a18ed02da89e1a0ccf276efb24453157ab021e8276b6607f70f6d347','interrupted.pt'),
        'context-spanning':(80,'af6fab3b095ebd400290e182fc5ae0186872f1db8efcc2b7e26269eda356e76c','live.pt')}


def authority():
    value=json.loads(AUTH.read_text())
    required=dict(status='OWNER_APPROVED_MATCHED_POSTUPDATE_RECOVERY',
        owner_answer='Approve recovery; keep 112 GPU-hours',
        proposal_commit='ed0285de1011703ffa8b07946cf485f8ccab1a01',
        provider='Isambard',account='brics.u6ada',failed_job='8260542',failed_source=PREDECESSOR,
        actual_prior_GPU_seconds=33821,maximum_aggregate_GPU_hours=112,maximum_total_running_intervals=6,
        next_interval=4,maximum_single_job_seconds=86400,simultaneous_GPUs=2,
        new_GCP_compute=False,new_purchases=False,new_external_model_API_calls=0,
        OneDrive_writes=False,Mac_checkpoint_downloads=False,unchanged_data_recipe_losses_folds_and_full_evaluation=True)
    if any(value.get(k)!=v for k,v in required.items()):raise ValueError('Exact owner postupdate recovery grant required')
    return g.sha(AUTH)


def entries():
    authority();value=json.loads(EVIDENCE.read_text())
    if value!=json.loads((SAVED/'actual-failure-and-state.json').read_text()):
        raise ValueError('Actual preserved failure evidence changed')
    if (value['job_id']!='8260542' or value['actual_state']!='FAILED' or value['exit_code']!='1:0'
            or value['elapsed_seconds']!=15147 or value['GPUs']!=2 or value['source']!=PREDECESSOR
            or value['total_prior_GPU_seconds_including_failure']!=33821):
        raise ValueError('Actual failed allocation/time must remain recorded as failed')
    rows=value['lineages']
    if len(rows)!=2 or {r['base'] for r in rows}!=set(STATES):raise ValueError('Both actual trained states required')
    return {r['base']:r for r in rows}


def reference_allowed(a,producer):
    base=getattr(a,'base',None)
    if producer!=PREDECESSOR or base not in STATES or getattr(a,'capacity',None)!='narrow':return False
    step,digest,name=STATES[base]
    if getattr(a,'checkpoint_sha',None)!=digest or Path(a.checkpoint)!=SAVED/base/name:return False
    authority();return True


def validate_failed(row,reference,step):
    base=row['base']
    if base not in STATES:raise ValueError('Unknown failed-state lineage')
    expected,digest,name=STATES[base];entry=entries()[base]
    if (row.get('postupdate_failure_recovery')!=MARKER or row['capacity']!='narrow' or step!=expected
            or Path(row['checkpoint'])!=SAVED/base/name or row['checkpoint_sha256']!=digest
            or reference['checkpoint_sha256']!=digest or reference['step']!=expected
            or reference['source']!=PREDECESSOR or reference['execution_id']!=entry['reference_execution_id']
            or entry['step']!=expected or entry['checkpoint_sha256']!=digest
            or Path(entry['preserved_checkpoint'])!=SAVED/base/name):
        raise ValueError('Only exact pinned actual saved89/80 states may recover')
    folder=SAVED/base
    for filename,item in entry['inventory'].items():
        if filename.endswith('.pt'):continue  # Native restore independently hashes the full checkpoint.
        if g.sha(folder/filename)!=item['sha256']:raise ValueError('Preserved original receipt/log changed')
    logs=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()]
    first=4 if base=='personaplex' else 1
    last=89 if base=='personaplex' else 87
    if ([r['step'] for r in logs]!=list(range(first,last+1)) or entry['last_logged_update']!=last
            or entry['unsaved_updates_lost']!=last-expected
            or any(r['source']!=PREDECESSOR or not math.isfinite(r['objective']) or r['LR']<=0 for r in logs)):
        raise ValueError('Actual completed update history changed; unsaved updates cannot be restored')
    if base=='personaplex':
        record=json.loads((folder/'training-interrupted.json').read_text())
        if (record['source']!=PREDECESSOR or record['completed'] is not False
                or record['total_lineage_updates']!=89 or record['actual_new_updates']!=86
                or record['checkpoint_sha256']!=digest):raise ValueError('Actual raw89 interruption record required')
    return True
