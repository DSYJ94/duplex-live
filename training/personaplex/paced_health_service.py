"""Opt-in final10 service. No live launcher uses this module yet.

Requires an explicit owner amendment; missing authorization fails BEFORE
creating any process or native guard. Old training/control files stay intact.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from . import matched_guard as g, matched_health as old
from . import local_process_identity as process, paced_observations as observations
from .paced_heartbeat import Heartbeat
from .matched_job import write

PROPOSAL = Path('docs/NATIVE-MOSHI-PACED-FINAL10-PROPOSAL-2026-11-11.json')
AUTH = Path('docs/NATIVE-MOSHI-PACED-FINAL10-AUTHORIZATION-2026-11-11.json')


def authority():
    from .matched_rpc_startup import authority as original_authority
    original = original_authority()
    proposal = json.loads(PROPOSAL.read_text())
    grant = json.loads(AUTH.read_text())
    required = dict(original_authorization_sha256=original, interval=10,
        maximum_aggregate_GPU_hours=112, maximum_total_running_intervals=10,
        simultaneous_GPUs=2, maximum_single_job_seconds=86400,
        provider_attempt_spacing_seconds=60, provider_max_age_seconds=90,
        local_max_age_seconds=45, storage_max_age_seconds=120,
        successful_positive_parent_job='8317068', change_current_job=False,
        saved_update_replay=False, new_purchases=False, new_GCP_compute=False,
        new_external_model_API_calls=0, OneDrive_writes=False)
    if any(proposal.get(k) != v for k, v in required.items()):
        raise ValueError('Exact final10-only pacing proposal required')
    if (grant.get('status') != 'OWNER_APPROVED_PACED_FINAL10'
            or not grant.get('owner_answer') or grant.get('proposal_sha256') != g.sha(PROPOSAL)
            or any(grant.get(k) != v for k, v in proposal.items() if k != 'status')):
        raise ValueError('Explicit unchanged-budget provider-TTL amendment required')
    return g.sha(AUTH)


def check_context(lease):
    g.resource(lease)
    if (lease.get('continuation_interval') != 10
            or os.environ.get('SLURM_JOB_ID') != lease['job_id']
            or os.environ.get('SLURM_JOB_ACCOUNT') != lease['account']
            or socket.gethostname().split('.')[0] != lease['node']):
        raise RuntimeError('Actual final10 host allocation required')


def watch(lease, control):
    grant = authority()
    check_context(lease)
    parent = json.loads((control/'parent-identity.json').read_text())
    # Child service must share the originally admitted host parent's cgroup/UID.
    own = process.capture(os.getpid())
    if own['cgroup'] != parent['cgroup'] or own['uids'] != parent['uids']:
        raise RuntimeError('Service escaped admitted parent identity')
    engine = Heartbeat(lease,
        lambda: observations.provider(lease, control/'provider.json'),
        lambda: observations.local(lease, parent, control/'storage.json'))
    while True:
        try:
            proof = engine.tick()
        except Exception:
            receipt = dict(engine.failure_receipt or {}, job_id=lease['job_id'],
                           service_source=lease['controller_source'])
            write(control/'heartbeat-failure.json', receipt)
            print(json.dumps(dict(event='heartbeat_failed', **receipt)), flush=True)
            raise
        row = dict(engine.provider['original_provider_receipt'])
        row.update(checked_unix=proof['local_observed_unix'], expires_unix=proof['expires_unix'],
            provider_observed_unix=proof['provider_observed_unix'],
            local_observed_unix=proof['local_observed_unix'],
            paced_authorization_sha256=grant, service_source=lease['controller_source'],
            checked_unix_semantics='LOCAL_PROCESS_AND_STORAGE_NOT_NEW_PROVIDER_OBSERVATION')
        g.ensure(lease, row)
        write(control/'guard.json', row)
        time.sleep(2)


def start(lease, execution):
    grant = authority()  # fail before creating files/services if unapproved
    check_context(lease)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], universal_newlines=True).strip()
    if source != lease.get('controller_source'):
        raise ValueError('Exact paced service Git producer required')
    execution = Path(execution)
    control = Path('/tmp')/('duplex-paced-'+lease['job_id']+'-'+source[:12])
    control.mkdir(mode=0o700, exist_ok=False)
    if control.stat().st_dev == execution.stat().st_dev:
        raise ValueError('Paced health storage must be node-local')
    write(control/'lease.json', lease)
    write(control/'parent-identity.json', process.capture(os.getpid()))
    write(execution/'paced-policy.json', dict(authorization_sha256=grant, source=source,
        local_max_age_seconds=45, provider_max_age_seconds=90, provider_spacing_seconds=60))
    services = []
    from .matched_interrupted import DURABLE
    os.environ['APPTAINER_BINDPATH'] = str(DURABLE)+':'+str(DURABLE)+','+str(control)+':'+str(control)

    def launch(module, args, name):
        with (execution/('paced-'+name+'.log')).open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', '-u', '-m', module]+args,
                                     stdout=log, stderr=subprocess.STDOUT)
        services.append(child)
        return child

    def wait_for(path, child, seconds):
        deadline = time.monotonic()+seconds
        while not path.exists():
            if child.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('Paced health service did not start')
            time.sleep(.2)

    try:
        common = ['--lease', str(control/'lease.json')]
        storage = launch('training.personaplex.matched_health',
            ['--mode', 'storage']+common+['--output', str(control/'storage.json')], 'storage')
        wait_for(control/'storage.json', storage, 60)
        old.storage_valid(lease, control/'storage.json')
        watcher = launch('training.personaplex.paced_health_service',
            common+['--control', str(control)], 'watch')
        wait_for(control/'guard.json', watcher, 45)
        g.ensure(lease, json.loads((control/'guard.json').read_text()))
        launch('training.personaplex.matched_health', ['--mode', 'mirror']+common+
            ['--local', str(control/'guard.json'), '--output', str(execution/'guard.json')], 'mirror')
        write(execution/'health-services.json', dict(job_id=lease['job_id'], source=source,
            local_control_dir=str(control), service_PIDs=[c.pid for c in services],
            authorization_sha256=grant, NFS_mirror_is_not_native_authority=True))
        return watcher, services, control
    except BaseException:
        old.stop(services, control, execution)
        raise


stop = old.stop

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lease', required=True)
    parser.add_argument('--control', required=True)
    args = parser.parse_args()
    watch(json.loads(Path(args.lease).read_text()), Path(args.control))
