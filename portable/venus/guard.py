"""Independent Mac provider/budget guard and fail-closed guest lease watchdog."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import urllib.error
from portable.context.gcp_budget_watchdog import read_runpod_pods

PROJECT='aurium-duplex-live-20261024'
ZONE='europe-west4-a'
NAME='duplex-native-saved8-gpu-flex90-20261025-v1'
IDENT='4443302983961393208'
DISK='1828090267781080539'
PRIOR_SECONDS=40351
PRIOR_USD=91.98448055555555


def resource(lease):
    if lease.get("resource_key") in ("venus-adaptive-v34","venus-adaptive-v35","venus-adaptive-v36"):
        from training.venus.adaptive_guard import resource as adaptive_resource
        return adaptive_resource(lease)
    if lease.get('resource_key') in ('venus-paced-continuation-v22','venus-paced-continuation-v23','venus-paced-recovery-v24','venus-paced-recovery-v25','venus-paced-recovery-v26','venus-paced-recovery-v27','venus-paced-recovery-v28','venus-paced-recovery-v29','venus-paced-recovery-v30','venus-paced-recovery-v31','venus-paced-recovery-v32','venus-paced-recovery-v33'):
        from training.venus.paced_guard import resource as paced_resource
        return paced_resource(lease)
    if lease.get('resource_key') in ('venus-focused-audit-v15','venus-voice-reference-v16','venus-voice-reference-v17','venus-voice-reference-v18','venus-cadence-qualify-v19','venus-cadence-qualify-v20','venus-cadence-qualify-v21','venus-paced-continuation-v22','venus-paced-continuation-v23','venus-paced-recovery-v24','venus-paced-recovery-v25','venus-paced-recovery-v26','venus-paced-recovery-v27','venus-paced-recovery-v28','venus-paced-recovery-v29','venus-paced-recovery-v30','venus-paced-recovery-v31','venus-paced-recovery-v32','venus-paced-recovery-v33'):
        from training.venus.focused_guard import resource as focused_resource
        return focused_resource(lease)
    if lease.get('resource_key')=='venus-behavior-v14':
        from training.venus.behavior_guard import resource as behavior_resource
        return behavior_resource(lease)
    if lease.get('resource_key')=='venus-recovery-v13':
        from training.venus.recovery_guard import resource as recovery_resource
        return recovery_resource(lease)
    if lease.get('resource_key')=='venus-memory-v11':
        from training.venus.memory_guard import resource as memory_resource
        return memory_resource(lease)
    if lease.get('resource_key')=='venus-native-gradient-v8':
        from training.venus.restart_guard import resource as restart_resource
        return restart_resource(lease)
    if lease.get('resource_key')=='venus-training-v1':
        from training.venus.guard import resource as training_resource
        return training_resource(lease)
    if lease.get('resource_key')=='us-a10040':
        path=Path('docs/NATIVE-REGIONAL-PLACEMENT-AUTHORIZATION-2026-10-26.json')
        authority=path.read_bytes();a=json.loads(authority)
        if (a['status']!='APPROVED' or a['maximum_additional_gcp_usage_usd']!=35 or
            a['maximum_additional_gpu_seconds']!=14400 or
            lease.get('authorization_sha256')!=hashlib.sha256(authority).hexdigest() or
            lease.get('zone') not in a['resources']['supported_catalog_zones'] or
            not all(str(lease.get(k,'')).isdigit() for k in ('vm_id','disk_id'))):
            raise ValueError('Explicit regional authority and actual resource identities required')
        if lease.get('access_mode')=='iap':
            from .access import approved_iap_proxy,AUTH,VM_ID,DISK_ID
            approved_iap_proxy()
            if (str(lease['vm_id'])!=VM_ID or str(lease['disk_id'])!=DISK_ID or
                lease.get('iap_authorization_sha256')!=hashlib.sha256(AUTH.read_bytes()).hexdigest()):
                raise ValueError('Exact supplemental IAP authority and resource required')
        return dict(name='duplex-venus-us-20261026-v1',zone=lease['zone'],
            ident=str(lease['vm_id']),disk=str(lease['disk_id']),disk_name='duplex-venus-us-20261026-v1',
            machine='a2-highgpu-1g',authority=path)
    if lease.get('resource_key')=='zone-b-a10040':
        path=Path('docs/NATIVE-FOUNDATION-CAPACITY-FALLBACK-AUTHORIZATION-2026-10-26.json')
        authority=path.read_bytes();a=json.loads(authority)
        if (a['status']!='APPROVED' or a['maximum_additional_gcp_usage_usd']!=35 or
            a['maximum_additional_gpu_seconds']!=14400 or
            lease.get('authorization_sha256')!=hashlib.sha256(authority).hexdigest() or
            not str(lease.get('vm_id','')).isdigit()):
            raise ValueError('Explicit fallback authority and actual resource identity required')
        return dict(name='duplex-venus-audio-20261026-b-v1',zone='europe-west4-b',
            ident=str(lease['vm_id']),disk='8246893242108371884',disk_name='duplex-native-stage-20261025-b-v1',
            machine='a2-highgpu-1g',authority=path)
    return dict(name=NAME,zone=ZONE,ident=IDENT,disk=DISK,disk_name='duplex-native-stage-20261024-v1',
        machine='a2-ultragpu-1g',authority=Path('docs/NATIVE-FOUNDATION-COMPARISON-AUTHORIZATION-2026-10-26.json'))


def stamp(s):
    return datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()


def run(argv,timeout=40):
    p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stderr[-1200:])
    return p.stdout


def cloud(*args,timeout=40):
    return json.loads(run(['gcloud','compute',*args,'--project='+PROJECT,'--format=json','--quiet'],timeout))


def bounded_cloud_read(*args):
    # One fresh retry of an idempotent read; never renew from cached state.
    for attempt,limit in enumerate((12,8)):
        try:return cloud(*args,timeout=limit)
        except subprocess.TimeoutExpired:
            if attempt:raise
            print(json.dumps(dict(event='bounded_provider_timeout_retry',read=list(args))),flush=True)
        except RuntimeError as error:
            transient=('RemoteDisconnected','Connection aborted.','Connection reset by peer',
                       'SSLEOFError','UNEXPECTED_EOF_WHILE_READING')
            permanent=('reauth','invalid_grant','permission_denied','unauthenticated')
            if attempt or any(value in str(error).lower() for value in permanent) or not any(value in str(error) for value in transient):raise
            print(json.dumps(dict(event='bounded_provider_connection_retry',read=list(args))),flush=True)


def bounded_runpod_read(key,reader=read_runpod_pods,sleep=time.sleep):
    """Retry transient HTTP failures only; never renew from a cached response."""
    for attempt in range(3):
        try:return reader(key)
        except urllib.error.HTTPError as error:
            if error.code not in (429,500,502,503,504) or attempt==2:raise
            print(json.dumps(dict(event='bounded_runpod_http_retry',status=error.code,attempt=attempt+1)),flush=True)
            sleep(attempt+1)


def provider_snapshot(binding,key):
    started=time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        vm=pool.submit(bounded_cloud_read,'instances','describe',binding['name'],'--zone='+binding['zone'])
        disk=pool.submit(bounded_cloud_read,'disks','describe',binding['disk_name'],'--zone='+binding['zone'])
        instances=pool.submit(bounded_cloud_read,'instances','list')
        pods=pool.submit(bounded_runpod_read,key)
        return vm.result(),disk.result(),instances.result(),pods.result(),started


def validate(vm,instances,runpods,lease,now):
    if lease.get("resource_key") in ("venus-adaptive-v34","venus-adaptive-v35","venus-adaptive-v36"):
        from training.venus.adaptive_guard import validate as adaptive_validate
        return adaptive_validate(vm,instances,runpods,lease,now)
    if lease.get('resource_key') in ('venus-paced-continuation-v22','venus-paced-continuation-v23','venus-paced-recovery-v24','venus-paced-recovery-v25','venus-paced-recovery-v26','venus-paced-recovery-v27','venus-paced-recovery-v28','venus-paced-recovery-v29','venus-paced-recovery-v30','venus-paced-recovery-v31','venus-paced-recovery-v32','venus-paced-recovery-v33'):
        from training.venus.paced_guard import validate as paced_validate
        return paced_validate(vm,instances,runpods,lease,now)
    if lease.get('resource_key') in ('venus-focused-audit-v15','venus-voice-reference-v16','venus-voice-reference-v17','venus-voice-reference-v18','venus-cadence-qualify-v19','venus-cadence-qualify-v20','venus-cadence-qualify-v21','venus-paced-continuation-v22','venus-paced-continuation-v23','venus-paced-recovery-v24','venus-paced-recovery-v25','venus-paced-recovery-v26','venus-paced-recovery-v27','venus-paced-recovery-v28','venus-paced-recovery-v29','venus-paced-recovery-v30','venus-paced-recovery-v31','venus-paced-recovery-v32','venus-paced-recovery-v33'):
        from training.venus.focused_guard import validate as focused_validate
        return focused_validate(vm,instances,runpods,lease,now)
    if lease.get('resource_key')=='venus-behavior-v14':
        from training.venus.behavior_guard import validate as behavior_validate
        return behavior_validate(vm,instances,runpods,lease,now)
    if lease.get('resource_key')=='venus-recovery-v13':
        from training.venus.recovery_guard import validate as recovery_validate
        return recovery_validate(vm,instances,runpods,lease,now)
    if lease.get('resource_key')=='venus-memory-v11':
        from training.venus.memory_guard import validate as memory_validate
        return memory_validate(vm,instances,runpods,lease,now)
    if lease.get('resource_key')=='venus-native-gradient-v8':
        from training.venus.restart_guard import validate as restart_validate
        return restart_validate(vm,instances,runpods,lease,now)
    if lease.get('resource_key')=='venus-training-v1':
        from training.venus.guard import validate as training_validate
        return training_validate(vm,instances,runpods,lease,now)
    b=resource(lease)
    if (str(vm['id'])!=b['ident'] or vm['name']!=b['name'] or
        not vm['zone'].endswith('/'+b['zone']) or vm['status']!='RUNNING' or
        not vm['machineType'].endswith('/'+b['machine']) or
        not any(d.get('source','').endswith('/disks/'+b['disk_name']) and
                d.get('boot') and not d.get('autoDelete') for d in vm['disks'])):
        raise ValueError('Exact approved live GPU/durable disk required')
    # Existing Flex configuration is immutable. Its shorter90min STOP is inside
    # the approved four-hour maximum; never recreate a VM to change that field.
    s=vm['scheduling'];start=stamp(vm['lastStartTimestamp'])
    regional=lease.get('resource_key')=='us-a10040'
    iap=lease.get('access_mode')=='iap'
    fallback=lease.get('resource_key') in ('zone-b-a10040','us-a10040')
    flex_b=fallback and s.get('provisioningModel')=='FLEX_START'
    flex_seconds=int(s.get('maxRunDuration',{}).get('seconds',0)) if flex_b else 0
    end=(start+flex_seconds if flex_b else stamp(s['terminationTime'])) if fallback else start+5400
    if fallback and not iap:
        # One replacement VM's whole wall lifetime bounds all its billed starts.
        # A restart cannot reset the approved additional four-hour allowance.
        end=min(end,stamp(vm['creationTimestamp'])+14400)
    model_ok=((s.get('provisioningModel','STANDARD')=='STANDARD' or
               (flex_b and flex_seconds in (3600,7200,14400))) and
              vm['creationTimestamp']==lease['creation_timestamp'] and 0<end-start<=14400) if fallback else (
              s.get('provisioningModel')=='FLEX_START' and str(s.get('maxRunDuration',{}).get('seconds'))=='5400')
    if (not model_ok or
        s.get('instanceTerminationAction')!='STOP' or s.get('automaticRestart') is not False or
        end!=lease['stop_unix'] or not start<=now<end):
        raise ValueError('Exact existing provider90min STOP deadline required')
    if fallback and b['ident']!='1962162267540853286':
        # Closed startup intervals are carried conservatively, including
        # allocation/stop overhead. Either bounded warm window fits the grant.
        if lease.get('prior_new_gpu_seconds_upper',0)<1200 or flex_seconds not in (3600,7200):
            raise ValueError('Preserve prior bootstrap and shorter recreation allowance')
        if lease['prior_new_gpu_seconds_upper']+flex_seconds>14400:
            raise ValueError('Additional four-hour allowance cannot reset')
    if regional and (flex_seconds>7200 or lease.get('disk_size_gib')!=100):
        raise ValueError('Regional allocation must retain its shorter window and100GB limit')
    if iap and (flex_seconds!=3600 or lease.get('prior_new_gpu_seconds_upper',0)<3600 or
                set(vm.get('tags',{}).get('items',[]))!={'duplex-venus-iap-bounded'}):
        raise ValueError('One newly authorized hour, carried history and exact isolated IAP tag required')
    if (PRIOR_SECONDS+14400>16*3600 or PRIOR_USD+7.25*4+6>135 or
        lease['prior_gpu_seconds']!=PRIOR_SECONDS or lease['prior_scope_usd']!=PRIOR_USD):
        raise ValueError('Cumulative budget cannot reset')
    running=[x for x in instances if x['status'] not in ('TERMINATED','STOPPED','STOPPING') and
        (x.get('guestAccelerators') or any('/'+m in x.get('machineType','') for m in ['a2-','a3-','g2-','a4-']))]
    if len(running)!=1 or str(running[0]['id'])!=b['ident']:
        raise ValueError('One project GPU only')
    if any(x.get('status') not in ('EXITED','TERMINATED','STOPPED') for x in runpods):
        raise ValueError('Runpod overlap; do not start another GPU')
    return end


def guest(args):
    import urllib.request
    req=urllib.request.Request('http://169.254.169.254/computeMetadata/v1/instance/id',headers={'Metadata-Flavor':'Google'})
    lease=json.loads(Path(args.lease).read_text()) if args.lease else {}
    b=resource(lease)
    with urllib.request.urlopen(req,timeout=5) as r:
        if r.read().decode()!=b['ident']:raise ValueError('Never shut down a replacement resource')
    while True:
        try:
            g=json.loads(Path(args.guard).read_text())
            okay=(g['vm_id']==b['ident'] and g['disk_id']==b['disk'] and time.time()<min(g['expires_unix'],g['stop_unix']))
        except (OSError,KeyError,ValueError):okay=False
        if not okay:
            subprocess.run(['/sbin/shutdown','-h','now'],check=True);return
        time.sleep(5)


def publish_guard(ssh,raw,status):
    # Local readiness is an acknowledgement of actual remote publication.
    # Publishing it before SSH completed let the guest read an expired old file.
    cmd='cat > /workspace/migration/venus-live-guard-v1.tmp && mv /workspace/migration/venus-live-guard-v1.tmp /workspace/migration/venus-live-guard-v1.json'
    p=subprocess.run(ssh+[cmd],input=raw,text=True,capture_output=True,timeout=20)
    if p.returncode:raise RuntimeError('Guest guard publication failed')
    Path(status).write_text(raw+'\n')


def mac(args):
    lease=json.loads(Path(args.lease).read_text());b=resource(lease)
    authority=b['authority'].read_bytes()
    a=json.loads(authority)
    expected='APPROVED_WITHIN_EXISTING_TRAINING_SCOPE' if lease.get('resource_key') in ('venus-memory-v11','venus-recovery-v13','venus-behavior-v14') else 'APPROVED'
    maximum=70 if lease.get('resource_key') in ('venus-adaptive-v34','venus-adaptive-v35','venus-adaptive-v36','venus-paced-recovery-v29','venus-paced-recovery-v30','venus-paced-recovery-v31','venus-paced-recovery-v32','venus-paced-recovery-v33') else 40 if lease.get('resource_key') in ('venus-focused-audit-v15','venus-voice-reference-v16','venus-voice-reference-v17','venus-voice-reference-v18','venus-cadence-qualify-v19','venus-cadence-qualify-v20','venus-cadence-qualify-v21','venus-paced-continuation-v22','venus-paced-continuation-v23','venus-paced-recovery-v24','venus-paced-recovery-v25','venus-paced-recovery-v26','venus-paced-recovery-v27','venus-paced-recovery-v28','venus-paced-recovery-v29','venus-paced-recovery-v30','venus-paced-recovery-v31','venus-paced-recovery-v32','venus-paced-recovery-v33') else 35
    assert a['status']==expected and a['maximum_additional_gcp_usage_usd']==maximum
    assert lease['authorization_sha256']==hashlib.sha256(authority).hexdigest()
    ssh=json.loads(Path(args.ssh_argv).read_text())
    if 'StrictHostKeyChecking=yes' not in ssh:raise ValueError('Strict SSH required')
    # A bulk rsync on the controller socket must never starve lease publication.
    # Keep authentication/host checking identical, with a separate multiplexor.
    ssh=[x+'.guard' if x.startswith('ControlPath=') else x for x in ssh]
    key=Path(args.runpod_key).read_text().strip()
    try:
        while True:
            if args.stop_file and Path(args.stop_file).exists():return
            birth=run(['/bin/ps','-p',str(lease['controller_pid']),'-o','lstart=']).strip()
            if birth!=lease['controller_birth']:raise ValueError('Controller identity no longer live')
            vm,disk,instances,pods,snapshot_start=provider_snapshot(b,key)
            if vm['status']=='TERMINATED':return
            assert str(disk['id'])==b['disk']
            if lease.get('resource_key') in ('venus-adaptive-v34','venus-adaptive-v35','venus-adaptive-v36','us-a10040','venus-training-v1','venus-native-gradient-v8','venus-memory-v11','venus-recovery-v13','venus-behavior-v14','venus-focused-audit-v15','venus-voice-reference-v16','venus-voice-reference-v17','venus-voice-reference-v18','venus-cadence-qualify-v19','venus-cadence-qualify-v20','venus-cadence-qualify-v21','venus-paced-continuation-v22','venus-paced-continuation-v23','venus-paced-recovery-v24','venus-paced-recovery-v25','venus-paced-recovery-v26','venus-paced-recovery-v27','venus-paced-recovery-v28','venus-paced-recovery-v29','venus-paced-recovery-v30','venus-paced-recovery-v31','venus-paced-recovery-v32','venus-paced-recovery-v33'):
                assert int(disk['sizeGb'])==100
            pods=pods.get('pods',[]) if isinstance(pods,dict) else pods
            end=validate(vm,instances,pods,lease,time.time())
            expires=min(snapshot_start+90,end)
            if time.time()>=expires:raise ValueError('Provider snapshot is already stale')
            g=dict(vm_id=b['ident'],disk_id=b['disk'],checked_unix=time.time(),snapshot_started_unix=snapshot_start,expires_unix=expires,
                stop_unix=end,authorization_sha256=lease['authorization_sha256'],guard_pid=os.getpid(),
                prior_gpu_seconds=lease['prior_gpu_seconds'],prior_scope_usd=lease['prior_scope_usd'])
            publish_guard(ssh,json.dumps(g),args.status)
            time.sleep(20)
    finally:
        vm=cloud('instances','describe',b['name'],'--zone='+b['zone'])
        if str(vm['id'])==b['ident'] and vm['status'] not in ('TERMINATED','STOPPED'):
            run(['gcloud','compute','instances','stop',b['name'],'--zone='+b['zone'],'--project='+PROJECT,'--quiet','--discard-local-ssd=True'],120)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--guest',action='store_true');p.add_argument('--guard');p.add_argument('--lease');p.add_argument('--ssh-argv');p.add_argument('--runpod-key');p.add_argument('--status');p.add_argument('--stop-file');args=p.parse_args();guest(args) if args.guest else mac(args)
