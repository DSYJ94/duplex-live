"""Mac-side guard for one already-created, approved saved8 recovery GPU.

No creation, training, restart or deletion operation. Credentials stay on the Mac.
Publish a fresh guard only after provider deadline, disk and non-overlap checks.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request

PROJECT = 'aurium-duplex-live-20261024'
ZONE = 'europe-west4-a'
DISK = 'duplex-native-stage-20261024-v1'
AUTH_SHA = '97c8d43fe7be11595e7d00775af4e1f2c7666a469041c720bf7586ee8a020322'
CATALOG_SHA = '94b42bae0152e8150ba7842146f66711b13df3b8ece9125bcb5db92a761196fb'
# Full approved CPU lifetime, seven days durable storage, and network reserve.
RESERVE_USD = 0.48 + 6.40 + 10.00
STOPPED = {'EXITED', 'STOPPED', 'TERMINATED', 'DELETED'}


def stamp(value):
    date = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if date.utcoffset() is None:
        raise ValueError('Timezone-qualified provider timestamp required')
    return date.timestamp()


def validate_lease(lease, authority, destination):
    if hashlib.sha256(authority).hexdigest() != AUTH_SHA:
        raise ValueError('Exact owner authorization required')
    grant = json.loads(authority)
    if (grant['status'] != 'APPROVED' or grant['project'] != PROJECT
            or grant['maximum_additional_gcp_usage_usd'] != 100
            or grant['optimizer_updates_authorized_by_this_proposal'] != 0):
        raise ValueError('Unchanged bounded zero-update scope required')
    if (lease['schema'] != 'dl-native-gcp-bounded-gpu-lease-v1'
            or lease['project'] != PROJECT or lease['zone'] != ZONE
            or not lease['name'].startswith('duplex-native-saved8-gpu-')
            or not str(lease['id']).isdigit() or lease['disk_name'] != DISK
            or not str(lease['disk_id']).isdigit()
            or lease['authorization_sha256'] != AUTH_SHA
            or type(lease['optimizer_updates']) is not int or lease['optimizer_updates'] != 0):
        raise ValueError('One exact recorded project/GPU/disk lease required')
    start, deadline = stamp(lease['created_utc']), stamp(lease['stop_utc'])
    rate = lease['hourly_upper_usd']
    if (type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0
            or not 0 < deadline-start <= 43200
            or rate*(deadline-start)/3600+RESERVE_USD > 100):
        raise ValueError('Whole GPU lifetime plus reserves must fit100USD/max12h')
    if (hashlib.sha256(destination).hexdigest() != lease['destination_report_sha256']):
        raise ValueError('Actual complete destination receipt changed')
    result = json.loads(destination)
    if (result['schema'] != 'dl-native-gcp-destination-hashes-v1'
            or result['inventory_sha256'] != CATALOG_SHA
            or result['passed'] is not True or result['all_catalog_entries_verified'] is not True
            or result['verified_regular_files'] != 40132
            or result['verified_bytes'] != 81154452891
            or result['verified_archival_links'] != 8 or result['failures']):
        raise ValueError('Full real native destination verification required')
    return start, deadline, rate


def owned(vm, lease):
    return (str(vm['id']) == str(lease['id']) and vm['name'] == lease['name']
            and stamp(vm['creationTimestamp']) == stamp(lease['created_utc'])
            and vm['zone'].endswith('/zones/'+ZONE))


def check_live(lease, vm, disk, instances, runpod_pods, now):
    if not owned(vm, lease):
        raise ValueError('Resource identity changed; never stop a replacement')
    deadline = stamp(lease['stop_utc'])
    if (vm['status'] != 'RUNNING' or not stamp(lease['created_utc']) <= now < deadline
            or not vm['machineType'].endswith('/machineTypes/a2-ultragpu-1g')
            or vm['scheduling'].get('instanceTerminationAction') != 'STOP'
            or stamp(vm['scheduling']['terminationTime']) != deadline
            or vm['scheduling'].get('automaticRestart') is not False):
        raise ValueError('Exact live singleA100 and native STOP deadline required')
    expected = '/projects/'+PROJECT+'/zones/'+ZONE+'/disks/'+DISK
    persistent = [d for d in vm['disks'] if d.get('type') == 'PERSISTENT']
    if (len(persistent) != 1 or not persistent[0]['source'].endswith(expected)
            or persistent[0].get('autoDelete') is not False
            or persistent[0].get('boot') is not True or persistent[0].get('mode') != 'READ_WRITE'
            or str(disk['id']) != str(lease['disk_id']) or disk['name'] != DISK
            or int(disk['sizeGb']) != 250):
        raise ValueError('Same250GB persistent non-auto-deleted disk required')
    accelerators = vm.get('guestAccelerators', [])
    if (len(accelerators) != 1 or accelerators[0]['acceleratorCount'] != 1
            or not accelerators[0]['acceleratorType'].endswith('/nvidia-a100-80gb')):
        raise ValueError('Exactly one A10080GB required')
    for other in instances:
        if other['name'] == DISK and other['status'] not in STOPPED:
            raise ValueError('The original staging CPU must be stopped')
        if (str(other['id']) != str(vm['id']) and other['status'] not in STOPPED
                and (other.get('guestAccelerators') or '/machineTypes/a2-' in other['machineType']
                     or '/machineTypes/a3-' in other['machineType'])):
            raise ValueError('Another GCP GPU is active')
    for pod in runpod_pods:
        if (pod.get('name', '').startswith('duplex-native-')
                and pod.get('status', pod.get('desiredStatus')) not in STOPPED
                and pod.get('computeType') != 'CPU'):
            raise ValueError('Another native Runpod GPU may be active')
    elapsed = max(0, now-stamp(lease['created_utc']))
    remaining = 100-RESERVE_USD-elapsed*lease['hourly_upper_usd']/3600
    planned = (deadline-now)*lease['hourly_upper_usd']/3600
    if not 0 < planned <= remaining:
        raise ValueError('Remaining full GPU window exceeds budget')
    return dict(execution_authorized=True, live_provider_deadline_verified=True,
                live_external_cost_guard_verified=True, single_gpu_across_providers_verified=True,
                checked_unix=now, deadline_unix=deadline, provider='gcp',
                remaining_budget_usd=remaining, planned_maximum_cost_usd=planned,
                resource_id=str(vm['id']), authorization_sha256=AUTH_SHA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('lease', 'destination', 'runpod-key-file', 'ssh-argv', 'status'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    lock = (args.status.parent/'gcp-native-budget-owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lease = json.loads(args.lease.read_text())
    authority = Path(__file__).resolve().parents[2]/'docs/GCP-NATIVE-CONTINUATION-AUTHORIZATION-2026-10-24.json'
    _, deadline, _ = validate_lease(lease, authority.read_bytes(), args.destination.read_bytes())
    if args.runpod_key_file.stat().st_mode & 0o077:
        raise ValueError('Local provider credential must have mode0600')
    key = args.runpod_key_file.read_text().strip()
    ssh = json.loads(args.ssh_argv.read_text())
    if (not isinstance(ssh, list) or ssh[0] != 'ssh' or not all(isinstance(s, str) for s in ssh)
            or 'StrictHostKeyChecking=yes' not in ssh or 'BatchMode=yes' not in ssh):
        raise ValueError('Previously verified strict private SSH connection required')

    def gcloud(*words):
        result = subprocess.run(['gcloud', 'compute', *words, '--project='+PROJECT,
                                 '--format=json', '--quiet'], check=True, timeout=25,
                                capture_output=True, text=True)
        return json.loads(result.stdout)

    def vm_read():
        return gcloud('instances', 'describe', lease['name'], '--zone='+ZONE)

    def stop_owned():
        current = vm_read()
        if not owned(current, lease):
            raise ValueError('Refusing to stop a different instance')
        subprocess.run(['gcloud', 'compute', 'instances', 'stop', lease['name'],
                        '--zone='+ZONE, '--project='+PROJECT, '--quiet'],
                       check=True, timeout=90, capture_output=True, text=True)

    def record(phase, **extra):
        value = dict(phase=phase, pid=os.getpid(), resource_id=str(lease['id']),
                     at_utc=datetime.now(timezone.utc).isoformat(), **extra)
        tmp = args.status.with_suffix('.tmp')
        tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(args.status)
        with args.status.with_suffix('.events.jsonl').open('a') as writer:
            writer.write(json.dumps(value)+'\n')

    awake = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
    try:
        while True:
            try:
                vm = vm_read()
                if not owned(vm, lease):
                    raise ValueError('Resource identity changed')
                if vm['status'] in STOPPED:
                    record('gpu_not_billing_compute'); return
                if time.time() >= deadline:
                    stop_owned(); record('provider_deadline_stop_requested'); return
                disk = gcloud('disks', 'describe', DISK, '--zone='+ZONE)
                instances = gcloud('instances', 'list')
                request = urllib.request.Request('https://api.runpod.io/v2/pods',
                    headers={'Authorization': 'Bearer '+key})
                with urllib.request.urlopen(request, timeout=15) as response:
                    pods = json.load(response)
                if isinstance(pods, dict): pods = pods['pods']
                if not isinstance(pods, list): raise ValueError('Live provider list required')
                guard = check_live(lease, vm, disk, instances, pods, time.time())
                ip = vm['networkInterfaces'][0]['accessConfigs'][0]['natIP']
                if ssh[-1].split('@')[-1] != ip:
                    raise ValueError('SSH destination differs from recorded GPU')
                local = args.status.with_suffix('.guard.json')
                local.write_text(json.dumps(guard)+'\n')
                subprocess.run(['rsync', '-ar', '--exclude=session-token', '--no-owner', '--no-group',
                    '--timeout=15', '-e', shlex.join(ssh[:-1]), str(local),
                    ssh[-1]+':/workspace/migration/gcp-budget-live-v1.json'],
                    check=True, timeout=20, capture_output=True, text=True)
                record('armed', guard=guard)
            except Exception as error:
                record('guard_failed_no_fresh_execution_permission', error_type=type(error).__name__)
                # Stop only this recorded lease; never retry a model or touch the disk.
                stop_owned(); record('owned_gpu_stopped_after_guard_failure')
                raise
            time.sleep(min(10, max(0.1, deadline-time.time())))
    finally:
        awake.terminate()


if __name__ == '__main__':
    main()
