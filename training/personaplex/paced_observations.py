"""Actual read adapters for proposed paced monitoring; no live imports changed.

The existing provider validator and storage gate remain authoritative. These
adapters do not submit jobs, renew stale observations, or authorize a new TTL.
"""
import json
import time
from pathlib import Path
from . import matched_guard, matched_health, local_process_identity


def provider(lease, receipt_path):
    # Passing no previous success forbids the old fast-timeout retry/fallback.
    # Heartbeat owns attempt spacing and expiry of the ORIGINAL observation.
    checked = matched_guard.refresh(lease, receipt_path, last_success=None)
    record = json.loads(Path(receipt_path).read_text())
    if record.get('checked_unix') != checked:
        raise RuntimeError('Actual provider receipt timestamp differs')
    # Preserve the complete original identity and authorization validation,
    # checked at its own real observation time, before normalizing the record.
    matched_guard.ensure(lease, record, now=checked)
    return dict(lease, observed_unix=checked, state='RUNNING',
                only_project_job=True, original_provider_receipt=record)


def local(lease, parent_identity, storage_path, proc_root='/proc', clock=time.time):
    actual = local_process_identity.verify(parent_identity, proc_root)
    storage = matched_health.storage_valid(lease, storage_path, now=clock())
    return dict(lease, observed_unix=clock(), process_alive=True,
                same_pid_start=True, same_job_cgroup=True, same_UID=True,
                storage_admission_valid=True, parent_identity=actual,
                storage_observed_unix=storage['checked_unix'])
