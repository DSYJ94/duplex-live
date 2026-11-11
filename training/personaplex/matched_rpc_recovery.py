"""Pinned failed-window6 states. Preparation alone never authorizes execution."""
import json,math
from pathlib import Path
from . import matched_guard as g
from .matched_interrupted import DURABLE

AUTH=Path('docs/NATIVE-MOSHI-RPC-RECOVERY-AUTHORIZATION-2026-11-11.json')
PROPOSAL=Path('docs/NATIVE-MOSHI-RPC-RECOVERY-PROPOSAL-2026-11-11.json')
PREDECESSOR='d9453cfaa09076b1c181723384da4c9cf49a9524'
FAILED_ROOT=DURABLE/'runs/storage-recovery-8285192'
MARKER='owner-approved-rpc-failure6-exact-states-v1'
STATES={
 ('personaplex','narrow'):(196,'a5f32f3ed8558ec3e47624003987fdfeacf6d81595ded80d491450a0ccde5e2a','epoch-04.pt'),
 ('context-spanning','narrow'):(196,'504a5f40d9232092a2028a8a92abe79432ca638e4b60712f12a630aeeb2a8670','epoch-04.pt'),
 ('personaplex','broad'):(62,'e5c8f62f11c65effc8bb91b7ee6fee226bb46091e8fafc90090e471a28ee7902','interrupted.pt'),
 ('context-spanning','broad'):(63,'93d0cc0b3a377cfa7f9d24f88b145e3d6fbfae99fd6c40b15b7d37a5bff023a4','interrupted.pt')}


def authority():
    proposal=json.loads(PROPOSAL.read_text());value=json.loads(AUTH.read_text())
    expected=dict(status='OWNER_APPROVED_RPC_FAILURE6_RECOVERY',provider='Isambard',account=g.ACCOUNT,
        failed_job='8285192',failed_source=PREDECESSOR,actual_prior_GPU_seconds=134991,
        maximum_aggregate_GPU_hours=112,maximum_total_running_intervals=8,next_interval=7,
        maximum_single_job_seconds=86400,simultaneous_GPUs=2,saved_update_replay=False,
        native_guard_seconds=45,storage_guard_seconds=120,new_GCP_compute=False,
        new_purchases=False,new_external_model_API_calls=0,OneDrive_writes=False,
        Mac_checkpoint_downloads=False,first_recovery_allowed_after_failed_window6=True)
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Exact owner failed6 recovery grant required')
    if value.get('proposal_sha256')!=g.sha(PROPOSAL) or not value.get('owner_answer'):
        raise ValueError('Actual owner answer and exact proposal binding required')
    if any(proposal.get(k)!=v for k,v in expected.items() if k!='status'):
        raise ValueError('Owner grant cannot broaden the concrete proposal')
    return g.sha(AUTH)


def selected(base,capacity):
    step,digest,name=STATES[(base,capacity)]
    return FAILED_ROOT/base/capacity/'train'/name,step,digest


def reference_allowed(a,producer):
    key=(getattr(a,'base',None),getattr(a,'capacity',None))
    if producer!=PREDECESSOR or key not in STATES:return False
    path,step,digest=selected(*key)
    if Path(a.checkpoint)!=path or getattr(a,'checkpoint_sha',None)!=digest:return False
    authority();return True


def validate(row,reference,step):
    authority();key=(row['base'],row['capacity']);path,wanted,digest=selected(*key)
    if (row.get('rpc_failure_recovery')!=MARKER or Path(row['checkpoint'])!=path
            or row['checkpoint_sha256']!=digest or step!=wanted
            or reference.get('step')!=wanted or reference.get('checkpoint_sha256')!=digest
            or reference.get('source')!=PREDECESSOR):
        raise ValueError('Exact four preserved states required; no rollback or replay')
    folder=path.parent
    if key[1]=='broad':
        value=json.loads((folder/'training-interrupted.json').read_text())
        if (value.get('source')!=PREDECESSOR or value.get('completed') is not False
                or value.get('total_lineage_updates')!=wanted or value.get('actual_new_updates')!=wanted
                or value.get('checkpoint_sha256')!=digest):raise ValueError('Actual interruption receipt required')
        logs=[json.loads(s) for s in (folder/'training.jsonl').read_text().splitlines()]
        if ([v['step'] for v in logs]!=list(range(1,wanted+1))
                or any(v['source']!=PREDECESSOR or not math.isfinite(v['objective']) or v['LR']<=0 for v in logs)):
            raise ValueError('Original completed update history must remain exact')
    else:
        value=json.loads((folder/'training-terminal.json').read_text())
        if (value.get('source')!=PREDECESSOR or value.get('completed') is not True
                or value.get('total_lineage_updates')!=196 or value.get('complete_epochs')!=4
                or value.get('checkpoint_sha256')!=digest or value.get('final_checkpoint')!=str(path)):
            raise ValueError('Actual completed narrow196 state required')
    return True
