"""Separate resource admission for the authorized substantial-capacity phase."""
import json
import math
from pathlib import Path
from datetime import datetime

from .large_inputs import sha, PARENT_SHA, MANIFEST_SHA

KEY = 'venus-large-capacity-v65'
AUTH = Path('docs/NATIVE-VENUS-LARGE-CAPACITY-AUTHORIZATION-2026-11-04.json')
ROOT = Path('runs/venus-large-capacity-20261104/v65')
OLD_USD = 165.7900361111111


def authority():
    a = json.loads(AUTH.read_text())
    if (a['status'] != 'APPROVED' or sha(a['proposal_path']) != a['proposal_sha256']
            or sha(a['value_resolution_path']) != a['value_resolution_sha256']
            or a['maximum_new_GCP_USD'] != 150 or a['maximum_project_USD'] != 325
            or a['maximum_aggregate_GPU_seconds'] != 43200
            or a['maximum_GPU_running_intervals'] != 3
            or not a['no_credit_purchase_or_external_model_API_calls']):
        raise ValueError('Exact value-resolved capacity authorization required')
    return a


def resource(lease):
    authority()
    path = Path(lease['execution_plan_path'])
    if path.parent != ROOT or path.name not in ('interval-01.json', 'interval-02.json', 'interval-03.json'):
        raise ValueError('One phase, at most three immutable interval plans required')
    p = json.loads(path.read_text())
    if (lease['resource_key'] != p['resource_key'] or p['resource_key'] != KEY
            or lease['authorization_sha256'] != p['authorization_sha256'] or sha(AUTH) != lease['authorization_sha256']
            or sha(path) != lease['execution_plan_sha256']
            or p['parent_sha256'] != PARENT_SHA or p['cache_manifest_sha256'] != MANIFEST_SHA
            or p['maximum_all_branch_updates'] != 984 or p['external_model_API_calls'] != 0
            or p['prior_project_USD'] != OLD_USD):
        raise ValueError('Exact parent/data/budget/interval binding required')
    for key in ('vm_id', 'disk_id', 'zone', 'creation_timestamp', 'storage_started_unix', 'storage_deadline_unix'):
        if lease[key] != p[key]: raise ValueError('Provider identity or storage clock differs')
    previous = p['previous_intervals']
    if len(previous) != int(path.stem[-2:]) - 1 or len(previous) > 2:
        raise ValueError('All prior actual intervals required; never reset the clock')
    seconds = 0
    for row in previous:
        receipt = Path(row['provider_final_path'])
        if receipt.parent != ROOT or sha(receipt) != row['sha256']:
            raise ValueError('Actual preserved terminal receipt required for each prior interval')
        vm = json.loads(receipt.read_text())
        if vm['status'] != 'TERMINATED' or vm['id'] != row['vm_id']:
            raise ValueError('Previous GPU must actually be stopped')
        elapsed = (datetime.fromisoformat(vm['lastStopTimestamp']) - datetime.fromisoformat(vm['lastStartTimestamp'])).total_seconds()
        upper = math.ceil(elapsed / 100) * 100
        if elapsed < 0 or upper != row['upper_seconds']: raise ValueError('Actual rounded elapsed time required')
        seconds += upper
        if p['optimizer_work_started'] and (row['vm_id'] != p['vm_id'] or row['disk_id'] != p['disk_id']):
            # A pre-optimizer sizing interval may precede a distinct training
            # placement. Once training starts its identity is immutable.
            if row['optimizer_updates'] != 0: raise ValueError('Never migrate an already training arm silently')
    duration = p['maximum_running_seconds']
    if (type(duration) is not int or duration < 1 or seconds + duration + 100 > 43200
            or lease['prior_capacity_GPU_seconds_upper'] != seconds
            or p['storage_deadline_unix'] != p['storage_started_unix'] + 86400
            or p['disk_size_gib'] != 250 or p['disk_id'] == '4375908109499109167'):
        raise ValueError('Aggregate running/storage limits or original-disk protection failed')
    choice = (p['machine'], p['provisioning'], p['zone'].rsplit('-', 1)[0])
    if choice not in (('a2-highgpu-1g', 'FLEX_START', 'us-central1'),
                      ('a2-ultragpu-1g', 'STANDARD', 'europe-west4')):
        raise ValueError('Only value-resolved A10040Flex or A10080Standard admitted')
    if p['zone'] not in ('us-central1-a','europe-west4-a','europe-west4-b','europe-west4-c'):
        raise ValueError('Only value-resolved first sizing zone or original Europe fallback zones admitted')
    if not p['vm_name'].startswith('duplex-venus-large-') or not p['disk_name'].startswith('duplex-venus-large-'):
        raise ValueError('Only exact new phase resources may be controlled')
    return dict(name=p['vm_name'], ident=p['vm_id'], disk=p['disk_id'], zone=p['zone'],
                disk_name=p['disk_name'], machine=p['machine'], authority=AUTH,
                provider_read_limits=(25,20), duration=duration, plan=p)


def validate(vm, instances, pods, lease, now):
    r = resource(lease); p = r['plan']
    start = datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()
    end = start + r['duration']
    if (vm['id'] != r['ident'] or vm['status'] != 'RUNNING' or vm['name'] != r['name']
            or vm['creationTimestamp'] != lease['creation_timestamp']
            or not vm['machineType'].endswith('/' + r['machine']) or not vm['zone'].endswith('/' + r['zone'])
            or set(vm['tags']['items']) != {'duplex-venus-train-bounded'} or vm.get('serviceAccounts')):
        raise ValueError('Actual approved isolated GPU identity required')
    disks = [d for d in vm['disks'] if d.get('type') != 'SCRATCH']
    scratch=[d for d in vm['disks'] if d.get('type')=='SCRATCH']
    if (r['machine']=='a2-ultragpu-1g' and (len(scratch)!=1 or int(scratch[0]['diskSizeGb'])!=375)
            or r['machine']=='a2-highgpu-1g' and scratch):
        raise ValueError('Exact priced local scratch storage required')
    if (len(disks) != 1 or not disks[0]['boot'] or disks[0]['autoDelete']
            or int(disks[0]['diskSizeGb']) != 250 or not disks[0]['source'].endswith('/disks/' + r['disk_name'])):
        raise ValueError('Exact retained250GB boot disk required')
    s = vm['scheduling']
    if (s['provisioningModel'] != p['provisioning'] or int(s['maxRunDuration']['seconds']) != r['duration']
            or s['instanceTerminationAction'] != 'STOP' or s['automaticRestart']
            or lease['stop_unix'] != end or not start <= now < end < lease['storage_deadline_unix']):
        raise ValueError('Actual native STOP and immutable storage window required')
    if not 0 < lease['verified_hourly_price_usd'] <= 8:
        raise ValueError('Fresh bounded price required')
    # 12h at the hard8USD/h maximum + all52USD preservation/network and2USD CPU
    # reserves, carrying old spending; credits never erase recorded consumption.
    if 43200 * 8 / 3600 + 54 > 150 or OLD_USD + 150 > 325:
        raise ValueError('Aggregate project/phase budget exceeded')
    active = [v for v in instances if v['status'] not in ('TERMINATED', 'STOPPED') and
              (v.get('guestAccelerators') or any('/' + m in v.get('machineType', '') for m in ('a2-', 'a3-', 'a4-', 'g2-', 'g4-')))]
    if len(active) != 1 or active[0]['id'] != r['ident'] or any(v['status'] not in ('EXITED', 'STOPPED', 'TERMINATED') for v in pods):
        raise ValueError('One actual GCP GPU and no Runpod GPU required')
    return end
