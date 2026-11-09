"""One pinned, genuinely interrupted quota-failure state; never invent success."""
import json,math
from pathlib import Path
from . import matched_guard as g

PREDECESSOR='e9622d6b6615d9b6135573fcbce00e32e24cb3a9'
STATE_SHA='fa2bbdef9434e9bdebe242e7b25a05f1d7dfea32d16c219d32a8b0170b5b3c8f'
EXECUTION='15f33cb966b4446fa8f207f5207866a6'
AUTH=Path('docs/NATIVE-MOSHI-QUOTA-RECOVERY-AUTHORIZATION-2026-11-09.json')
DURABLE=Path('/home/u6ada/dsyj94.u6ada/duplex-native-recovery-20261109-v1')
PRESERVED=DURABLE/'preserved/personaplex-narrow-step3'


def authority():
    value=json.loads(AUTH.read_text())
    required=dict(status='OWNER_DIRECTED_QUOTA_FAILURE_RECOVERY',failed_job='8235976',
        actual_prior_GPU_seconds=3527,maximum_aggregate_GPU_hours=112,
        maximum_total_running_intervals=4,next_interval=3,simultaneous_GPUs=2,
        maximum_single_job_seconds=86400,OneDrive_writes=False,new_GCP_compute=False,
        new_purchases=False,new_external_model_API_calls=0,preserve_saved_personaplex_step3=True)
    if any(value.get(k)!=v for k,v in required.items()):raise ValueError('Exact owner-directed quota recovery required')
    return g.sha(AUTH)


def reference_allowed(a,producer):
    if producer!=PREDECESSOR:return False
    authority()
    return (getattr(a,'base',None)=='personaplex' and getattr(a,'capacity',None)=='narrow' and getattr(a,'checkpoint_sha',None)==STATE_SHA
            and Path(a.checkpoint)==PRESERVED/'interrupted.pt')


def validate_interrupted(row,reference,step):
    authority()
    if (row.get('quota_interrupted_state')!='dl-moshi-quota-interrupted-v1'
            or row['base']!='personaplex' or row['capacity']!='narrow' or step!=3
            or Path(row['checkpoint'])!=PRESERVED/'interrupted.pt'
            or row['checkpoint_sha256']!=STATE_SHA or reference['source']!=PREDECESSOR
            or reference['checkpoint_sha256']!=STATE_SHA or reference['step']!=3
            or reference['execution_id']!=EXECUTION):raise ValueError('Only the pinned actual interrupted step3 may recover')
    record=json.loads((PRESERVED/'training-interrupted.json').read_text())
    if (record.get('source')!=PREDECESSOR or record.get('completed') is not False
            or record.get('actual_new_updates')!=3 or record.get('total_lineage_updates')!=3
            or record.get('checkpoint_sha256')!=STATE_SHA):raise ValueError('Actual interrupted record differs')
    log=[json.loads(x) for x in (PRESERVED/'training.jsonl').read_text().splitlines()]
    if ([r['step'] for r in log]!=[1,2,3] or any(r['source']!=PREDECESSOR
            or not math.isfinite(r['objective']) or r['LR']<=0 for r in log)):
        raise ValueError('Actual three-update training log required')
    receipt=json.loads((DURABLE/'preserved/failure-8235976/actual-failure.json').read_text())
    if (receipt.get('job_id')!='8235976' or receipt.get('state')!='FAILED'
            or receipt.get('exit_code')!='1:0' or receipt.get('elapsed_seconds')!=644
            or receipt.get('GPUs')!=2 or receipt.get('guard_error')!='EDQUOT'):
        raise ValueError('Actual failed allocation must remain recorded as failed')
    return True
