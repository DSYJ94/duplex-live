"""Exact owner-approved failed-start7 recovery8 and positive continuation9."""
import json
from pathlib import Path
from . import matched_guard as g
from . import matched_rpc_recovery as recovery

AUTH=Path('docs/NATIVE-MOSHI-RPC-STARTUP-CONTINUATION-AUTHORIZATION-2026-11-11.json')
PROPOSAL=Path('docs/NATIVE-MOSHI-RPC-STARTUP-CONTINUATION-PROPOSAL-2026-11-11.json')
KEY='rpc_startup_authorization_sha256'


def authority():
    recovery.authority()
    proposal=json.loads(PROPOSAL.read_text());value=json.loads(AUTH.read_text())
    expected=dict(status='OWNER_APPROVED_RPC_STARTUP_RECOVERY8',provider='Isambard',account=g.ACCOUNT,
        failed_job='8314042',actual_prior_GPU_seconds=135051,maximum_aggregate_GPU_hours=112,
        maximum_total_running_intervals=9,next_interval=8,simultaneous_GPUs=2,
        maximum_single_job_seconds=86400,native_guard_seconds=45,storage_guard_seconds=120,
        all_four_fresh_native_restores_before_learning=True,saved_update_replay=False,
        new_GCP_compute=False,new_purchases=False,new_external_model_API_calls=0,
        OneDrive_writes=False,Mac_checkpoint_downloads=False)
    expected['selected_states']={b+'/'+c:dict(step=n,sha256=d,name=f) for (b,c),(n,d,f) in recovery.STATES.items()}
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Exact owner startup8/9 grant required')
    if not value.get('owner_answer') or value.get('proposal_sha256')!=g.sha(PROPOSAL):
        raise ValueError('Actual owner answer and immutable proposal required')
    if any(value.get(k)!=v for k,v in proposal.items() if k!='status'):
        raise ValueError('Startup grant cannot broaden owner-reviewed proposal')
    return g.sha(AUTH)


def first_interval(lease):
    if lease.get(KEY) is None:return 7
    if lease[KEY]!=authority():raise ValueError('Exact startup continuation identity required')
    return 8


def history(original):
    return dict(original,**{'8314042':dict(state='FAILED',elapsed=30,GPUs=2)})
