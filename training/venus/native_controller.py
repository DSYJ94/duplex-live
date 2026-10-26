"""One warm restart, native-shaped full-gradient training, same hardware/caps."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from datetime import datetime
from . import controller as c
from .restart_guard import AUTH
from portable.context.gcp_budget_watchdog import read_runpod_pods

BASE=Path('runs/venus-training-20261026')
OLD=BASE/'allocation-v2'


def main():
    c.R=BASE/'native-v9';assert not c.R.exists(),'Never duplicate a real warm restart'
    for prior in BASE.glob('native-v*/events.json'):
        assert not any(e['event']=='one_actual_warm_restart' for e in json.loads(prior.read_text())), 'Carry any actual prior warm interval before a further start'
    assert not c.run(['git','status','--porcelain','--','training/venus','data/venus','portable/venus/guard.py']).strip()
    approval=BASE/'data-approval-v7.json';assert json.loads(approval.read_text())['approved']
    a=json.loads(AUTH.read_text());r=a['existing_resource'];zone=r['zone']
    assert a['status']=='APPROVED' and a['total_conservative_scope_usd']<35 and a['hourly_admission_ceiling_usd']==4
    previous=json.loads((BASE/'recovery-v7/provider-final.json').read_text());assert previous['status']=='TERMINATED'
    elapsed=datetime.fromisoformat(previous['lastStopTimestamp']).timestamp()-datetime.fromisoformat(previous['lastStartTimestamp']).timestamp()
    assert 4600+elapsed<=6400
    # The old live watcher must be gone before the new start and new lease.
    oldguard=json.loads((OLD/'guard.json').read_text())['guard_pid']
    assert subprocess.run(['/bin/ps','-p',str(oldguard),'-o','pid='],capture_output=True).returncode!=0
    vms=c.cloud('compute','instances','list');assert all(v['status'] in ('TERMINATED','STOPPED') for v in vms)
    vm=next(v for v in vms if v['name']==c.NAME);assert str(vm['id'])==r['vm_id']
    assert vm['lastStartTimestamp']==previous['lastStartTimestamp'] and vm['lastStopTimestamp']==previous['lastStopTimestamp'],'Unrecorded interval must never reset the ledger'
    disk=c.cloud('compute','disks','describe',c.NAME,'--zone='+zone);assert str(disk['id'])==r['disk_id'] and int(disk['sizeGb'])==100
    pods=read_runpod_pods(Path('runs/local-migration-20261021/.private/runpod-budget-api-key').read_text().strip());pods=pods.get('pods',[]) if isinstance(pods,dict) else pods
    assert all(p['status'] in ('EXITED','STOPPED','TERMINATED') for p in pods)
    price=json.loads(Path('runs/venus-20261026/us-current-price40-v1.json').read_text());assert 0<price['hourly_estimate_usd']<=4 and time.time()-datetime.fromisoformat(price['at_utc']).timestamp()<3600
    c.R.mkdir();c.source=c.run(['git','rev-parse','HEAD']).strip();c.remote=c.ROOT+'/duplex-venus-native-'+c.source[:12]
    c.run(['git','bundle','create',str(c.R/'source.bundle'),'HEAD'],120)
    c.record('warm_restart_preflight',source=c.source,vm_id=r['vm_id'],disk_id=r['disk_id'],prior_new_seconds_upper=6400,next_max_seconds=7200,verified_price=price['hourly_estimate_usd'])
    awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())]);guard=None;started=False
    try:
        # Flex scheduling is immutable; retain the existing authorized7200s STOP.
        check=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone);assert str(check['id'])==r['vm_id'] and check['status']=='TERMINATED' and int(check['scheduling']['maxRunDuration']['seconds'])==7200
        operation=c.cloud('compute','instances','start',c.NAME,'--zone='+zone,'--async');operation=operation[0] if isinstance(operation,list) else operation;started=True
        c.record('one_actual_warm_restart',operation=operation['name'])
        until=time.time()+180
        while operation['status']!='DONE':
            if time.time()>until:raise TimeoutError('Observe same restart operation, do not duplicate')
            time.sleep(3);operation=c.cloud('compute','operations','describe',operation['name'],'--zone='+zone)
        if operation.get('error'):raise RuntimeError(json.dumps(operation['error']))
        vm=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone);assert str(vm['id'])==r['vm_id'] and vm['status']=='RUNNING'
        start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();end=start+7200
        c.record('actual_running',last_start=vm['lastStartTimestamp'],native_stop_unix=end)
        c.ssh_argv=json.loads((OLD/'ssh.json').read_text())
        for _ in range(20):
            try:c.ssh('test -d '+c.ROOT+' && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader',15);break
            except RuntimeError:time.sleep(3)
        else:raise RuntimeError('Existing strict IAP SSH unavailable')
        c.sync(c.R/'source.bundle',c.ROOT+'/native-source-v8.bundle')
        c.ssh('git --git-dir='+c.ROOT+'/source.git fetch '+c.ROOT+'/native-source-v8.bundle HEAD && git --git-dir='+c.ROOT+'/source.git worktree add --detach '+c.remote+' '+c.source)
        lease=dict(resource_key='venus-native-gradient-v8',zone=zone,vm_id=r['vm_id'],disk_id=r['disk_id'],disk_size_gib=100,creation_timestamp=vm['creationTimestamp'],stop_unix=end,prior_new_gpu_seconds_upper=6400,prior_gpu_seconds=40351,prior_scope_usd=a['prior_original_scope_cost_upper_usd'],hourly_admission_ceiling_usd=4,verified_hourly_price_usd=price['hourly_estimate_usd'],authorization_sha256=hashlib.sha256(AUTH.read_bytes()).hexdigest(),controller_pid=os.getpid(),controller_birth=c.run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip())
        (c.R/'lease.json').write_text(json.dumps(lease,indent=2)+'\n');(c.R/'ssh.json').write_text(json.dumps(c.ssh_argv)+'\n');c.sync(c.R/'lease.json',c.ROOT+'/venus-native-gradient-lease-v8.json')
        guard=subprocess.Popen([sys.executable,'-u','-m','portable.venus.guard','--lease',str(c.R/'lease.json'),'--ssh-argv',str(c.R/'ssh.json'),'--runpod-key','runs/local-migration-20261021/.private/runpod-budget-api-key','--status',str(c.R/'guard.json'),'--stop-file',str(c.R/'stop-requested')],stdout=(c.R/'guard.log').open('x'),stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'});c.guard=guard
        until=time.time()+90
        while not (c.R/'guard.json').exists():
            if guard.poll() is not None:raise RuntimeError('Independent Mac guard ended')
            if time.time()>until:raise TimeoutError('No new live verified guard')
            time.sleep(1)
        c.ssh('sudo systemd-run --unit=duplex-venus-native-guest-guard-v8 --property=RuntimeMaxSec=7300 --working-directory='+c.remote+' /usr/bin/python3 -u -m portable.venus.guard --guest --guard '+c.ROOT+'/venus-live-guard-v1.json --lease '+c.ROOT+'/venus-native-gradient-lease-v8.json')
        assert c.props('duplex-venus-native-guest-guard-v8')['ActiveState']=='active'
        c.ssh('sudo systemctl stop duplex-venus-train-bootstrap-stop.timer');c.record('both_new_guards_live',mac_pid=guard.pid)
        py=c.ROOT+'/venus-runtime-v1/venv/bin/python';env='CUBLAS_WORKSPACE_CONFIG=:4096:8 DUPLEX_LIVE_GUARD_FILE='+c.ROOT+'/venus-live-guard-v1.json DUPLEX_LIVE_LEASE_FILE='+c.ROOT+'/venus-native-gradient-lease-v8.json HF_HOME='+c.ROOT+'/venus-runtime-v1/hf-cache'
        def python(cmd):return '/usr/bin/env '+env+' '+py+' -u -m '+cmd
        c.sync(approval,c.OUT+'/data-approval-v7.json')
        proof='duplex-venus-native-proof-v8';c.start_unit(proof,python('training.venus.native_cache --model '+c.MODEL+' --timelines '+c.OUT+'/timelines-v7-repaired --output '+c.OUT+'/native-proof-v8 --proof-only'),600);c.wait_unit(proof,min(end-300,time.time()+650))
        cache='duplex-venus-native-cache-v8';c.start_unit(cache,python('training.venus.native_cache --model '+c.MODEL+' --timelines '+c.OUT+'/timelines-v7-repaired --approval '+c.OUT+'/data-approval-v7.json --output '+c.OUT+'/native-cache-v8'),max(1,int(end-time.time()-300)));c.wait_unit(cache,end-300)
        args=' --model '+c.MODEL+' --cache '+c.OUT+'/native-cache-v8 --approval '+c.OUT+'/data-approval-v7.json --output '+c.OUT+'/candidate-v8 --source '+c.source+' --lease '+c.ROOT+'/venus-native-gradient-lease-v8.json'
        first='duplex-venus-native-numerical-start-v8';c.start_unit(first,python('training.venus.train --stage start'+args),max(1,int(end-time.time()-300)));c.wait_unit(first,end-300)
        sustained='duplex-venus-native-restore-sustain-v8';c.start_unit(sustained,python('training.venus.train --stage restore-continue'+args),max(1,int(end-time.time()-300)));c.wait_unit(sustained,end-300)
        c.record('actual_native_training_complete',updates=304,adopted=False,full_behavior_and_speech_still_required=True)
    except BaseException as error:
        c.record('native_continuation_failure',error_type=type(error).__name__,error=str(error)[-2500:]);raise
    finally:
        if c.ssh_argv:
            try:
                for unit in c.UNITS:
                    state=c.props(unit)
                    if state['ActiveState']=='active' and state['SubState']!='exited':c.ssh('sudo systemctl stop '+unit,30)
                c.copyback()
            except Exception as error:c.record('copyback_error_disk_preserved',error=str(error)[-1500:])
        if guard and guard.poll() is None:
            (c.R/'stop-requested').write_text('closing\n')
            try:guard.wait(timeout=150)
            except subprocess.TimeoutExpired:c.record('guard_stop_observation_timeout',pid=guard.pid)
        if started:
            current=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone);assert str(current['id'])==r['vm_id']
            if current['status'] not in ('TERMINATED','STOPPED'):c.run(['gcloud','compute','instances','stop',c.NAME,'--zone='+zone,'--project='+c.PROJECT,'--quiet'],150)
            current=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone);(c.R/'provider-final.json').write_text(json.dumps(current,indent=2)+'\n');c.record('provider_stopped',vm_id=current['id'],status=current['status'])
        awake.terminate()


if __name__=='__main__':main()
