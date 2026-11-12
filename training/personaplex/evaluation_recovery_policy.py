"""Narrow authority for the owner's explicitly requested evaluation recovery."""
import json
from pathlib import Path
from . import matched_guard as g
KEY='evaluation_recovery_authorization_sha256'
AUTH=Path('docs/NATIVE-MOSHI-EVALUATION-RECOVERY-AUTHORIZATION-2026-11-12.json')


def authority():
    value=json.loads(AUTH.read_text())
    expected=dict(status='OWNER_DIRECTED_EVALUATION_RECOVERY',failed_job='8352856',
        training_job='8317068',maximum_aggregate_GPU_hours=112,next_interval=12,
        maximum_total_running_intervals=12,simultaneous_GPUs=2,maximum_single_job_seconds=13800,
        optimizer_updates=0,reuse_outputs=76,required_comparison_outputs=192,
        provider_attempt_spacing_seconds=60,provider_max_age_seconds=150,
        local_max_age_seconds=45,storage_max_age_seconds=120,
        new_purchases=False,new_GCP_compute=False,new_external_model_API_calls=0)
    if any(value.get(k)!=v for k,v in expected.items()) or not value.get('owner_request'):
        raise ValueError('Exact bounded evaluation-only recovery authority required')
    return g.sha(AUTH)


def validate(lease):
    if (lease.get(KEY)!=authority() or lease.get('continuation_interval')!=12
            or lease.get('optimizer_updates')!=0 or lease.get('maximum_seconds',13801)>13800):
        raise ValueError('Recovery cannot train, exceed its interval, or reset limits')
    return 150
