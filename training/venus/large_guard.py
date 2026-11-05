"""Separate resource admission for the authorized substantial-capacity phase."""
import json
import math
from pathlib import Path
from datetime import datetime

from .large_inputs import sha, PARENT_SHA, MANIFEST_SHA

KEY = 'venus-large-capacity-v65'
AUTH = Path('docs/NATIVE-VENUS-LARGE-CAPACITY-AUTHORIZATION-2026-11-04.json')
MEMORY_AUTH = Path('docs/NATIVE-VENUS-LARGE-CAPACITY-MEMORY-RESUME-AUTHORIZATION-2026-11-04.json')
PREFIX_AUTH = Path('docs/NATIVE-VENUS-LARGE-CAPACITY-PREFIX-CONTINUATION-AUTHORIZATION-2026-11-04.json')
LAYOUT_AUTH = Path('docs/NATIVE-VENUS-LARGE-CAPACITY-LAYOUT-CONTINUATION-AUTHORIZATION-2026-11-04.json')
AGGREGATE_AUTH = Path('docs/NATIVE-VENUS-LARGE-CAPACITY-AGGREGATE-CONTINUATION-AUTHORIZATION-2026-11-04.json')
REPORTING_REVIEW = Path('docs/NATIVE-VENUS-LARGE-CAPACITY-REPORTING-RECOVERY-REVIEW-2026-11-04.json')
REPORTING_REVIEW_SHA = '046b8d504a3ff3d38cc1201660f581319b8824db4d58c412b330622238661b68'
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


def memory_authority():
    authority()
    a=json.loads(MEMORY_AUTH.read_text())
    if (a['status']!='APPROVED' or sha(a['proposal_path'])!=a['proposal_sha256']
            or a['original_authorization_sha256']!=sha(AUTH)
            or a['maximum_GPU_running_intervals']!=4 or a['maximum_new_GCP_USD']!=150
            or a['maximum_project_USD']!=325 or a['maximum_aggregate_GPU_seconds']!=43200
            or a['prior_GPU_seconds_upper']!=1600 or a['maximum_new_interval_seconds']!=41500
            or a['shutdown_rounding_reserve_seconds']!=100 or a['maximum_actual_start_attempts']!=1
            or a['vm_id']!='7521667153519853425' or a['disk_id']!='2297341858915206463'):
        raise ValueError('Exact same-resource fourth-interval amendment required')
    return a


def prefix_authority():
    memory_authority()
    a=json.loads(PREFIX_AUTH.read_text())
    if (a['status']!='APPROVED' or sha(a['proposal_path'])!=a['proposal_sha256']
            or a['prior_authorization_sha256']!=sha(MEMORY_AUTH)
            or a['maximum_GPU_running_intervals']!=6 or a['maximum_new_GCP_USD']!=150
            or a['maximum_project_USD']!=325 or a['maximum_aggregate_GPU_seconds']!=43200
            or a['prior_GPU_seconds_upper']!=2100 or a['next_maximum_interval_seconds']!=41000
            or a['shutdown_rounding_reserve_seconds']!=100
            or a['vm_id']!='7521667153519853425' or a['disk_id']!='2297341858915206463'
            or not a['optional_sixth_only_after_actual_new_updates'] or not a['no_failed_admission_retry']):
        raise ValueError('Exact shared-history continuation authorization required')
    return a


def layout_authority():
    prefix_authority()
    a=json.loads(LAYOUT_AUTH.read_text())
    if (a['status']!='APPROVED' or sha(a['proposal_path'])!=a['proposal_sha256']
            or a['prior_authorization_sha256']!=sha(PREFIX_AUTH)
            or not a['sixth_may_validate_layout_before_new_updates']
            or a['maximum_GPU_running_intervals']!=6 or a['maximum_new_GCP_USD']!=150
            or a['maximum_project_USD']!=325 or a['maximum_aggregate_GPU_seconds']!=43200
            or a['prior_GPU_seconds_upper']!=2600 or a['next_maximum_interval_seconds']!=40500
            or a['shutdown_rounding_reserve_seconds']!=100
            or a['vm_id']!='7521667153519853425' or a['disk_id']!='2297341858915206463'
            or a['maximum_capacity_only_prestart_requests']!=3 or a['minimum_capacity_request_spacing_seconds']!=600):
        raise ValueError('Exact sixth-interval layout-purpose amendment required')
    return a


def aggregate_authority():
    layout_authority()
    a=json.loads(AGGREGATE_AUTH.read_text())
    if (a['status']!='APPROVED' or sha(a['proposal_path'])!=a['proposal_sha256']
            or a['prior_authorization_sha256']!=sha(LAYOUT_AUTH)
            or a['maximum_GPU_running_intervals']!=12 or a['maximum_new_GCP_USD']!=150
            or a['maximum_project_USD']!=325 or a['maximum_aggregate_GPU_seconds']!=43200
            or a['prior_GPU_seconds_upper']!=2700 or a['next_maximum_interval_seconds']!=40400
            or a['shutdown_rounding_reserve_seconds']!=100
            or a['vm_id']!='7521667153519853425' or a['disk_id']!='2297341858915206463'
            or a['maximum_further_failed_kernel_storage_admissions']!=2 or not a['no_unchanged_failed_repeats']):
        raise ValueError('Exact unchanged-budget aggregate continuation authority required')
    return a


def resource(lease):
    authority()
    path = Path(lease['execution_plan_path'])
    if path.parent != ROOT or path.name not in tuple('interval-%02d.json'%i for i in range(1,13)):
        raise ValueError('Only explicitly authorized immutable interval plans required')
    p = json.loads(path.read_text())
    fourth=path.name=='interval-04.json'
    fifth=path.name=='interval-05.json'
    sixth=path.name=='interval-06.json'
    number=int(path.stem[-2:]);sustained=number>=10 and bool(p.get("sustained_authorization_sha256"));aggregate=number>=7 and not sustained
    if sustained:
        from .sustained import AUTH as S_AUTH,authority as s_authority
        a=s_authority()
        if (p["sustained_authorization_sha256"]!=sha(S_AUTH) or lease.get("sustained_authorization_sha256")!=sha(S_AUTH)
                or p["vm_id"]!=a["vm_id"] or p["disk_id"]!=a["disk_id"]
                or p["zone"]!="europe-west4-a" or p["machine"]!="a2-ultragpu-1g" or p["provisioning"]!="STANDARD"
                or p["vm_name"]!="duplex-venus-large-gpu-20261104-eua-v67" or p["disk_name"]!="duplex-venus-large-data-20261104-eua-v67"
                or not 0<p["maximum_running_seconds"]<=64800 or p["storage_deadline_unix"]!=a["storage_deadline_unix"]
                or p.get("activation_memory_policy")!="repeat4-prefix-pageable-v3-layout"
                or number==10 and p["optimizer_work_started"] or number>10 and not p["optimizer_work_started"]):
            raise ValueError("Exact sustained resources, clock and actual state required")
        if p["optimizer_work_started"]:
            receipt=Path(p["trained_state_receipt_path"])
            if receipt.parent!=ROOT or sha(receipt)!=p["trained_state_receipt_sha256"]:raise ValueError("Bound trained state required")
            trained=json.loads(receipt.read_text())
            if trained["actual_new_updates"]<=0 or trained["vm_id"]!=p["vm_id"] or trained["disk_id"]!=p["disk_id"] or trained["cache_manifest_sha256"]!=MANIFEST_SHA:raise ValueError("Actual same-resource trained state required")
    if aggregate:
        a=aggregate_authority()
        if (p.get('aggregate_authorization_sha256')!=sha(AGGREGATE_AUTH)
                or lease.get('aggregate_authorization_sha256')!=sha(AGGREGATE_AUTH)
                or p['vm_id']!=a['vm_id'] or p['disk_id']!=a['disk_id']
                or p['zone']!='europe-west4-a' or p['machine']!='a2-ultragpu-1g'
                or p['vm_name']!='duplex-venus-large-gpu-20261104-eua-v67'
                or p['disk_name']!='duplex-venus-large-data-20261104-eua-v67'
                or p['provisioning']!='STANDARD' or not 0<p['maximum_running_seconds']<=40400
                or p['storage_deadline_unix']!=1793897263.296897
                or p.get('activation_memory_policy')!='repeat4-prefix-pageable-v3-layout'):
            raise ValueError('Aggregate continuation requires exact authority and same GPU/disk')
        if number>8 and not p['optimizer_work_started']:
            # Owner's limit covers kernel/storage failures, not this diagnosed
            # Python reporting exception after successful CONTROL backward.
            # Keep this exception bound to one reviewed recovery, never a loop.
            if (number!=9 or p.get('reporting_recovery_review_sha256')!=REPORTING_REVIEW_SHA
                    or sha(REPORTING_REVIEW)!=REPORTING_REVIEW_SHA):
                raise ValueError('Later intervals require actual trained state or exact ninth reporting recovery')
        if p['optimizer_work_started']:
            receipt=Path(p['trained_state_receipt_path'])
            if receipt.parent!=ROOT or sha(receipt)!=p['trained_state_receipt_sha256']:
                raise ValueError('Bound actual trained checkpoint receipt required')
            trained=json.loads(receipt.read_text())
            if (trained['actual_new_updates']<=0 or trained['vm_id']!=p['vm_id'] or trained['disk_id']!=p['disk_id']
                    or trained['cache_manifest_sha256']!=MANIFEST_SHA):
                raise ValueError('Actual same-resource trained state required, never replay parent')
    if sixth:
        a=layout_authority()
        if (p.get('layout_authorization_sha256')!=sha(LAYOUT_AUTH)
                or lease.get('layout_authorization_sha256')!=sha(LAYOUT_AUTH)
                or p['vm_id']!=a['vm_id'] or p['disk_id']!=a['disk_id']
                or p['zone']!='europe-west4-a' or p['machine']!='a2-ultragpu-1g'
                or p['vm_name']!='duplex-venus-large-gpu-20261104-eua-v67'
                or p['disk_name']!='duplex-venus-large-data-20261104-eua-v67'
                or p['provisioning']!='STANDARD' or p['maximum_running_seconds']!=40500
                or p['storage_deadline_unix']!=1793897263.296897
                or p.get('activation_memory_policy')!='repeat4-prefix-pageable-v3-layout'):
            raise ValueError('Sixth interval requires exact layout authority and same resources')
    if fifth:
        a=prefix_authority()
        if (p.get('prefix_authorization_sha256')!=sha(PREFIX_AUTH)
                or lease.get('prefix_authorization_sha256')!=sha(PREFIX_AUTH)
                or p['vm_id']!=a['vm_id'] or p['disk_id']!=a['disk_id']
                or p['zone']!='europe-west4-a' or p['machine']!='a2-ultragpu-1g'
                or p['vm_name']!='duplex-venus-large-gpu-20261104-eua-v67'
                or p['disk_name']!='duplex-venus-large-data-20261104-eua-v67'
                or p['provisioning']!='STANDARD' or p['maximum_running_seconds']!=41000
                or p['storage_deadline_unix']!=1793897263.296897
                or p.get('activation_memory_policy')!='repeat4-prefix-pageable-v2'):
            raise ValueError('Fifth interval requires exact shared-history authority and same resources')
    if fourth:
        a=memory_authority()
        if (p.get('memory_authorization_sha256')!=sha(MEMORY_AUTH)
                or lease.get('memory_authorization_sha256')!=sha(MEMORY_AUTH)
                or p['vm_id']!=a['vm_id'] or p['disk_id']!=a['disk_id']
                or p['zone']!='europe-west4-a' or p['machine']!='a2-ultragpu-1g'
                or p['vm_name']!='duplex-venus-large-gpu-20261104-eua-v67'
                or p['disk_name']!='duplex-venus-large-data-20261104-eua-v67'
                or p['provisioning']!='STANDARD' or p['maximum_running_seconds']!=41500
                or p['storage_deadline_unix']!=1793897263.296897):
            raise ValueError('Fourth interval is bound to the same80VM/disk and original clock')
    if (lease['resource_key'] != p['resource_key'] or p['resource_key'] != KEY
            or lease['authorization_sha256'] != p['authorization_sha256'] or sha(AUTH) != lease['authorization_sha256']
            or sha(path) != lease['execution_plan_sha256']
            or p['parent_sha256'] != PARENT_SHA or p['cache_manifest_sha256'] != MANIFEST_SHA
            or p['maximum_all_branch_updates'] != (120 if sustained else 984) or p['external_model_API_calls'] != 0
            or p['prior_project_USD'] != OLD_USD):
        raise ValueError('Exact parent/data/budget/interval binding required')
    for key in ('vm_id', 'disk_id', 'zone', 'creation_timestamp', 'storage_started_unix', 'storage_deadline_unix'):
        if lease[key] != p[key]: raise ValueError('Provider identity or storage clock differs')
    previous = p['previous_intervals']
    if len(previous) != number - 1 or len(previous) > (11 if aggregate or sustained else 5 if sixth else 4 if fifth else 3 if fourth else 2):
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
    if fourth and seconds!=1600:raise ValueError('All1600 actual prior GPU upper seconds must carry forward')
    if fifth and seconds!=2100:raise ValueError('All2100 actual prior GPU upper seconds must carry forward')
    if sixth and seconds!=2600:raise ValueError('All2600 actual prior GPU upper seconds must carry forward')
    if aggregate and (seconds<2700 or number==7 and seconds!=2700):raise ValueError('Carry all actual GPU history, at least2700 seconds')
    if (type(duration) is not int or duration < 1 or seconds + duration + 100 > (73600 if sustained else 43200)
            or sustained and (seconds<8700 or number==10 and seconds!=8700)
            or lease['prior_capacity_GPU_seconds_upper'] != seconds
            or not sustained and p['storage_deadline_unix'] != p['storage_started_unix'] + 86400
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
    if p.get('sustained_authorization_sha256'):
        if not 0<lease['verified_hourly_price_usd']<=5.9 or 73600*5.9/3600+54>175 or OLD_USD+175>350:
            raise ValueError('Sustained budget exceeded')
    elif 43200 * 8 / 3600 + 54 > 150 or OLD_USD + 150 > 325:
        raise ValueError('Aggregate project/phase budget exceeded')
    active = [v for v in instances if v['status'] not in ('TERMINATED', 'STOPPED') and
              (v.get('guestAccelerators') or any('/' + m in v.get('machineType', '') for m in ('a2-', 'a3-', 'a4-', 'g2-', 'g4-')))]
    if len(active) != 1 or active[0]['id'] != r['ident'] or any(v['status'] not in ('EXITED', 'STOPPED', 'TERMINATED') for v in pods):
        raise ValueError('One actual GCP GPU and no Runpod GPU required')
    return end
