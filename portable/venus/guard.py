"""Independent Mac provider/budget guard and fail-closed guest lease watchdog."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

PROJECT='aurium-duplex-live-20261024'
ZONE='europe-west4-a'
NAME='duplex-native-saved8-gpu-flex90-20261025-v1'
IDENT='4443302983961393208'
DISK='1828090267781080539'
PRIOR_SECONDS=40351
PRIOR_USD=91.98448055555555


def resource(lease):
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


def cloud(*args):
    return json.loads(run(['gcloud','compute',*args,'--project='+PROJECT,'--format=json','--quiet']))


def validate(vm,instances,runpods,lease,now):
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
    fallback=lease.get('resource_key')=='zone-b-a10040'
    end=stamp(s['terminationTime']) if fallback else start+5400
    model_ok=(s.get('provisioningModel','STANDARD')=='STANDARD' and
              vm['creationTimestamp']==lease['creation_timestamp'] and 0<end-start<=14400) if fallback else (
              s.get('provisioningModel')=='FLEX_START' and str(s.get('maxRunDuration',{}).get('seconds'))=='5400')
    if (not model_ok or
        s.get('instanceTerminationAction')!='STOP' or s.get('automaticRestart') is not False or
        end!=lease['stop_unix'] or not start<=now<end):
        raise ValueError('Exact existing provider90min STOP deadline required')
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


def mac(args):
    from portable.context.gcp_budget_watchdog import load_runpod_pods
    lease=json.loads(Path(args.lease).read_text());b=resource(lease)
    authority=b['authority'].read_bytes()
    a=json.loads(authority)
    assert a['status']=='APPROVED' and a['maximum_additional_gcp_usage_usd']==35
    assert lease['authorization_sha256']==hashlib.sha256(authority).hexdigest()
    ssh=json.loads(Path(args.ssh_argv).read_text())
    if 'StrictHostKeyChecking=yes' not in ssh:raise ValueError('Strict SSH required')
    key=Path(args.runpod_key).read_text().strip()
    try:
        while True:
            if args.stop_file and Path(args.stop_file).exists():return
            birth=run(['/bin/ps','-p',str(lease['controller_pid']),'-o','lstart=']).strip()
            if birth!=lease['controller_birth']:raise ValueError('Controller identity no longer live')
            vm=cloud('instances','describe',b['name'],'--zone='+b['zone'])
            if vm['status']=='TERMINATED':return
            disk=cloud('disks','describe',b['disk_name'],'--zone='+b['zone'])
            assert str(disk['id'])==b['disk']
            pods=load_runpod_pods(key);pods=pods.get('pods',[]) if isinstance(pods,dict) else pods
            end=validate(vm,cloud('instances','list'),pods,lease,time.time())
            g=dict(vm_id=b['ident'],disk_id=b['disk'],checked_unix=time.time(),expires_unix=min(time.time()+90,end),
                stop_unix=end,authorization_sha256=lease['authorization_sha256'],guard_pid=os.getpid(),
                prior_gpu_seconds=PRIOR_SECONDS,prior_scope_usd=PRIOR_USD)
            raw=json.dumps(g);Path(args.status).write_text(raw+'\n')
            # Status contains no credentials. Publish by atomic rename on the guest.
            cmd='cat > /workspace/migration/venus-live-guard-v1.tmp && mv /workspace/migration/venus-live-guard-v1.tmp /workspace/migration/venus-live-guard-v1.json'
            p=subprocess.run(ssh+[cmd],input=raw,text=True,capture_output=True,timeout=20)
            if p.returncode:raise RuntimeError('Guest guard publication failed')
            time.sleep(20)
    finally:
        vm=cloud('instances','describe',b['name'],'--zone='+b['zone'])
        if str(vm['id'])==b['ident'] and vm['status'] not in ('TERMINATED','STOPPED'):
            run(['gcloud','compute','instances','stop',b['name'],'--zone='+b['zone'],'--project='+PROJECT,'--quiet','--discard-local-ssd=True'],120)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--guest',action='store_true');p.add_argument('--guard');p.add_argument('--lease');p.add_argument('--ssh-argv');p.add_argument('--runpod-key');p.add_argument('--status');p.add_argument('--stop-file');args=p.parse_args();guest(args) if args.guest else mac(args)
