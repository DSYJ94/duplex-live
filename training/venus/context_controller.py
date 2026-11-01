"""One same-VM context epoch; actual terminals, independent guards and copyback."""
import json,math,os,shlex,subprocess,sys,time
from pathlib import Path
from datetime import datetime
from . import controller as c
from .source_controller import prepare_ssh
from .context_guard import KEY,PLAN,RETRY_KEY,RETRY_PLAN
from .context_objective import ADMISSION,authority,sha
from portable.venus.guard import bounded_runpod_read


def preserve_candidate(candidate):
    """Copy complete new training state before any subsequent evaluation."""
    backup=c.R/'critical-candidate'
    backup.mkdir(parents=True,exist_ok=False)
    code="import pathlib,json,hashlib;r=pathlib.Path("+repr(candidate)+");print(json.dumps([dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(r.rglob('*')) if p.is_file() and not p.is_symlink()]))"
    rows=json.loads(c.ssh('/usr/bin/python3 -c '+shlex.quote(code),60))
    if not rows or 'training-terminal.json' not in {r['path'] for r in rows}:
        raise ValueError('Actual complete new candidate required')
    c.run(['rsync','-ar','--no-owner','--no-group','--exclude=session-token','--exclude=.private','--exclude=credentials','--exclude=owner-conversations','--exclude=API keys.md','--exclude=core','--exclude=core.*','-e',shlex.join(c.ssh_argv[:-1]),c.ssh_argv[-1]+':'+candidate+'/',str(backup)+'/'],120)
    for row in rows:
        path=backup/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Candidate backup hash mismatch; never declare preservation')
    (c.R/'critical-candidate-verified.json').write_text(json.dumps(dict(all_hashes_verified=True,source=c.source,files=rows),indent=2)+'\n')
    c.record('critical_candidate_backed_up_before_evaluation',files=len(rows),bytes=sum(r['bytes'] for r in rows))


def retry_ready(prior,vm,now):
    rows=json.loads((prior/'events.json').read_text())
    failed=[r for r in rows if r['event']=='context_phase_failed']
    if len(failed)!=1 or 'ZONE_RESOURCE_POOL_EXHAUSTED' not in failed[0]['error'] or (prior/'provider-start.json').exists():
        raise ValueError('Only a terminal capacity failure before any GPU/model start admits one retry')
    if any(r['event']=='actual_unit_started' for r in rows):raise ValueError('Never repeat actual model work')
    old=json.loads((prior/'provider-final.json').read_text())
    if vm['status']!='TERMINATED' or any(vm[k]!=old[k] for k in ('id','lastStartTimestamp','lastStopTimestamp')):
        raise ValueError('Original provider identity and terminal state must still match')
    if now<datetime.fromisoformat(failed[0]['at_utc']).timestamp()+300:raise ValueError('One delayed retry only; no capacity polling loop')


def main(capacity_retry=False):
    global KEY,PLAN
    if capacity_retry:KEY,PLAN=RETRY_KEY,RETRY_PLAN
    a=authority();p=json.loads(PLAN.read_text());zone=a['resource']['zone']
    c.R=Path('runs/venus-training-20261101')/('context-v49' if capacity_retry else 'context-v48')
    if c.R.exists():raise ValueError('Observe original controller; never restart this allocation')
    c.NAME=a['resource']['name'];c.source=c.run(['git','rev-parse','HEAD']).strip();c.remote=c.ROOT+'/duplex-context-'+c.source[:12]
    if c.run(['git','-c','core.filemode=false','status','--porcelain','--','training/venus','data/venus','portable/venus']).strip():raise ValueError('Deploy committed code only')
    price=json.loads(Path('runs/venus-training-20261101/current-a100-price.json').read_text())
    assert 0<price['hourly_estimate_usd']<=4 and time.time()-datetime.fromisoformat(price['at_utc']).timestamp()<3600
    instances=c.cloud('compute','instances','list');assert all(v['status']=='TERMINATED' for v in instances)
    vm=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone)
    if capacity_retry:retry_ready(Path('runs/venus-training-20261101/context-v48'),vm,time.time())
    assert vm['id']==a['resource']['vm_id'] and vm['lastStopTimestamp']==p['previous_provider_stop']
    disk=c.cloud('compute','disks','describe','duplex-venus-train-20261026-v1','--zone='+zone)
    assert disk['id']==a['resource']['disk_id'] and int(disk['sizeGb'])==100
    pods=bounded_runpod_read(Path('runs/local-migration-20261021/.private/runpod-budget-api-key').read_text().strip());pods=pods.get('pods',[]) if isinstance(pods,dict) else pods
    assert all(v['status'] in ('EXITED','STOPPED','TERMINATED') for v in pods)
    timelines=Path(p['source_timelines']);assert sha(timelines/'manifest.json')==a['new_timeline_manifest_sha256']
    c.R.mkdir(parents=True)
    c.run(['git','bundle','create',str(c.R/'source.bundle'),'HEAD'],120)
    awake=subprocess.Popen(['caffeinate','-i','-s','-w',str(os.getpid())]);guard=None;broker=None;started=False;end=None
    remote_out=c.OUT+'/context-v48';parent=c.OUT+'/action-candidate-v42/step-0108.pt';candidate=remote_out+'/candidate'
    try:
        c.cloud('compute','os-login','ssh-keys','add','--key-file=runs/local-migration-20261024/.private/gcp-native-ssh.pub','--ttl=2h')
        bootstrap=c.R/'bootstrap.sh';bootstrap.write_text('#!/bin/sh\nset -eu\nsystemd-run --unit=duplex-context48-bootstrap-stop --on-active=600s /sbin/shutdown -h now || /sbin/shutdown -h now\n')
        c.cloud('compute','instances','add-metadata',c.NAME,'--zone='+zone,'--metadata-from-file=startup-script='+str(bootstrap))
        c.cloud('compute','instances','set-scheduling',c.NAME,'--zone='+zone,'--max-run-duration=4000s','--instance-termination-action=STOP')
        op=c.cloud('compute','instances','start',c.NAME,'--zone='+zone,'--async');op=op[0] if isinstance(op,list) else op;started=True
        c.record('one_context_allocation',operation=op['name'],source=c.source,maximum_new_updates=60,prior_focused_seconds=28700,maximum_seconds=4100)
        until=time.time()+1000
        while op['status']!='DONE':
            if time.time()>until:raise TimeoutError('Observe SAME provider operation; never resubmit')
            time.sleep(5);op=c.cloud('compute','operations','describe',op['name'],'--zone='+zone)
        if op.get('error'):raise RuntimeError(json.dumps(op['error']))
        vm=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone);assert vm['status']=='RUNNING'
        end=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()+4000
        (c.R/'provider-start.json').write_text(json.dumps(vm,indent=2)+'\n')
        prepare_ssh(vm,zone,'48');c.sync(c.R/'source.bundle',c.ROOT+'/context48.bundle')
        c.ssh('git --git-dir='+c.ROOT+'/source.git fetch '+c.ROOT+'/context48.bundle HEAD && git --git-dir='+c.ROOT+'/source.git worktree add --detach '+c.remote+' '+c.source)
        lease=dict(resource_key=KEY,zone=zone,vm_id=vm['id'],disk_id=disk['id'],disk_size_gib=100,creation_timestamp=vm['creationTimestamp'],stop_unix=end,prior_new_gpu_seconds_upper=28700,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,verified_hourly_price_usd=price['hourly_estimate_usd'],authorization_sha256=sha(ADMISSION),execution_plan_sha256=sha(PLAN),controller_pid=os.getpid(),controller_birth=c.run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip())
        lp=c.ROOT+'/context48-lease.json';(c.R/'lease.json').write_text(json.dumps(lease,indent=2)+'\n');c.sync(c.R/'lease.json',lp)
        guard=subprocess.Popen([sys.executable,'-u','-m','portable.venus.guard','--lease',str(c.R/'lease.json'),'--ssh-argv',str(c.R/'ssh.json'),'--runpod-key','runs/local-migration-20261021/.private/runpod-budget-api-key','--status',str(c.R/'guard.json'),'--stop-file',str(c.R/'stop-requested')],stdout=(c.R/'guard.log').open('x'),stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'});c.guard=guard
        until=time.time()+90
        while not (c.R/'guard.json').exists():
            if guard.poll() is not None or time.time()>until:raise RuntimeError('Independent Mac guard not ready')
            time.sleep(1)
        live=json.loads(c.ssh('cat '+c.ROOT+'/venus-live-guard-v1.json'));assert live['vm_id']==vm['id'] and live['stop_unix']==end and time.time()<live['expires_unix']
        c.ssh('sudo systemd-run --unit=duplex-context48-guest --property=RuntimeMaxSec=4200 --working-directory='+c.remote+' /usr/bin/python3 -u -m portable.venus.guard --guest --guard '+c.ROOT+'/venus-live-guard-v1.json --lease '+lp)
        assert c.props('duplex-context48-guest')['ActiveState']=='active'
        c.ssh('sudo systemctl stop duplex-context48-bootstrap-stop.timer');c.record('both_guards_live',mac_pid=guard.pid)
        c.ssh('mkdir -p '+remote_out)
        assert int(c.ssh("df -B1 --output=avail /workspace/migration | tail -1").strip())>8*1024**3,'Need sufficient existing disk space; do not add storage'
        for local,remote in ((str(timelines)+'/',remote_out+'/timelines/'),('runs/venus-training-20261101/context-alignment-v1/',remote_out+'/alignment/'),('runs/venus-training-20261101/context-semantic-v1/',remote_out+'/semantic/')):c.sync(local,remote)
        c.sync('runs/venus-training-20261029/action-target-audit-v1/index-verified-v2.json',remote_out+'/old-action-index.json')
        assert c.ssh('sha256sum '+parent).split()[0]==a['parent_checkpoint_sha256']
        py=c.ROOT+'/venus-runtime-v1/venv/bin/python'
        env='/usr/bin/env DUPLEX_LIVE_GUARD_FILE='+c.ROOT+'/venus-live-guard-v1.json DUPLEX_LIVE_LEASE_FILE='+lp+' HF_HOME='+c.ROOT+'/venus-runtime-v1/hf-cache '+py+' -u -m '
        admission=c.remote+'/'+str(ADMISSION);cache=remote_out+'/cache'
        def stage(label,command,reserve,maximum):
            assert not c.ssh('nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits').strip(),'Prior GPU unit remains live'
            assert guard.poll() is None and end-time.time()>reserve+30
            unit='duplex-venus-context48-'+label
            c.start_unit(unit,env+command,min(maximum,int(end-time.time()-reserve)))
            c.wait_unit(unit,end-reserve)
            c.record('context_stage_terminal0',stage=label)
        stage('cache','training.venus.context_cache --model '+c.MODEL+' --timelines '+remote_out+'/timelines --old-cache '+c.OUT+'/paced-prefix-cache-v24 --old-timelines '+c.OUT+'/paced-timelines-v22 --old-index '+remote_out+'/old-action-index.json --tokenizer-config '+c.MODEL+'/tokenizer_config.json --admission '+admission+' --output '+cache+' --source '+c.source,2800,1200)
        common=' --model '+c.MODEL+' --cache '+cache+' --lease '+lp+' --source '+c.source+' --opal-source '+c.ROOT+'/opal-upstream-v34 --context-phase-admission '+admission
        stage('parent-restore','training.venus.native_restore --kind parent --checkpoint '+parent+' --sha256 '+a['parent_checkpoint_sha256']+' --output '+remote_out+'/parent-restore'+common.replace(' --cache '+cache,' --cache '+c.OUT+'/paced-prefix-cache-v24'),2600,180)
        box=remote_out+'/opal-mailbox'
        args=common+' --approval '+cache+'/approval.json --output '+candidate+' --cache-producer '+c.source+' --parent-checkpoint '+parent+' --parent-sha256 '+a['parent_checkpoint_sha256']+' --opal-backend mailbox:'+box+' --opal-agent-budget .01 --maximum-epochs 1 --coverage-manifest '+cache+'/timelines-manifest.json --action-target-index '+cache+'/action-index.json --parent-native-restore '+remote_out+'/parent-restore/restore.json'
        stage('train-start','training.venus.train --stage start'+args,2300,300)
        code="import torch,json;s=torch.load('"+candidate+"/step-0001.pt',map_location='cpu',weights_only=False);print(json.dumps(s['optimizer']['backend']))"
        run_id=json.loads(c.ssh(py+' -c '+shlex.quote(code)))['run_id']
        command=[sys.executable,'-u','-m','training.venus.opal_supervisor','--source','/private/tmp/duplex-opal-upstream-20261028','--box',box,'--repo',str(Path.cwd()),'--decisions',str(c.R/'opal-decisions'),'--workdir',str(c.R/'opal-broker'),'--run-id',run_id,'--ssh-argv',str(c.R/'ssh.json'),'--remote-repo',c.remote]
        broker=subprocess.Popen(command,stdout=(c.R/'opal-broker.log').open('x'),stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        c.record('current_agent_opal_review_live',pid=broker.pid,run_id=run_id,external_api_calls=0)
        stage('train-continue','training.venus.train --stage restore-continue'+args,1900,800)
        terminal=json.loads(c.ssh('cat '+candidate+'/training-terminal.json'));assert terminal['completed'] and terminal['actual_updates']==60 and terminal['total_optimizer_updates']==466
        c.record('actual_new_training_complete',terminal=terminal)
        preserve_candidate(candidate)
        final=candidate+'/step-0060.pt';finalsha=c.ssh('sha256sum '+final).split()[0]
        stage('final-restore','training.venus.native_restore --kind final --checkpoint '+final+' --sha256 '+finalsha+' --output '+remote_out+'/final-restore'+common,1600,240)
        c.ssh('cp '+remote_out+'/final-restore/restore.json '+candidate+'/final-native-restore.json')
        paired='training.venus.paced_behavior --model '+c.MODEL+' --candidate '+candidate+' --parent-checkpoint '+parent+' --parent-sha256 '+a['parent_checkpoint_sha256']+' --plan '+c.remote+'/evals/fixtures/venus-control-repair-source-v2-20261027.json --timelines '+c.OUT+'/paced-timelines-v22 --approval '+c.OUT+'/paced-data-approval-v22.json --word-quality '+c.OUT+'/paced-word-quality-v22.json --old-plan '+c.remote+'/evals/fixtures/venus-control-corpus-v4-20261026.json --old-timelines '+c.OUT+'/timelines-v7-repaired --old-approval '+c.OUT+'/data-approval-v7.json --output '+remote_out+'/native-paired --source '+c.source+' --lease '+lp+' --tts-attention released --voice-reference '+c.OUT+'/george-reference-v33/reference-16k.wav --voice-reference-sha256 40159a59a35053c358d5192b3919a86453c8c6bf1b2019b289119ceca444f674 --reuse-parent-collection '+c.OUT+'/action-native-paired-v42 --context-phase-admission '+admission+' --training-approval '+cache+'/approval.json'
        stage('paired-native',paired,240,1500)
        c.record('complete_native_comparison',adopted=False,new_general_context_native_qualification_still_required=True)
    except BaseException as error:
        c.record('context_phase_failed',error_type=type(error).__name__,error=str(error)[-1400:]);raise
    finally:
        if broker and broker.poll() is None:broker.terminate()
        if c.ssh_argv:
            try:
                for unit in c.UNITS:
                    if c.props(unit).get('ActiveState')=='active':c.ssh('sudo systemctl stop '+unit,30)
                code="import pathlib,json,hashlib;r=pathlib.Path('"+remote_out+"');print(json.dumps([dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(r.rglob('*')) if p.is_file() and not p.is_symlink()]))"
                rows=json.loads(c.ssh('/usr/bin/python3 -c '+shlex.quote(code),90))
                c.run(['rsync','-ar','--no-owner','--no-group','--exclude=session-token','--exclude=.private','--exclude=credentials','--exclude=owner-conversations','--exclude=API keys.md','--exclude=core','--exclude=core.*','-e',shlex.join(c.ssh_argv[:-1]),c.ssh_argv[-1]+':'+remote_out+'/',str(c.R/'copyback')+'/'],180)
                for r in rows:
                    path=c.R/'copyback'/r['path'];assert path.stat().st_size==r['bytes'] and sha(path)==r['sha256']
                (c.R/'copyback-verified.json').write_text(json.dumps(dict(all_hashes_verified=True,files=len(rows),bytes=sum(r['bytes'] for r in rows),inventory=rows),indent=2)+'\n')
                c.record('full_context_copyback_verified',files=len(rows),bytes=sum(r['bytes'] for r in rows))
            except Exception as error:c.record('copyback_failed_disk_preserved',error_type=type(error).__name__,error=str(error)[-600:])
        if started:
            try:
                finalvm=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone)
                if finalvm['status'] not in ('TERMINATED','STOPPED'):c.run(['gcloud','compute','instances','stop',c.NAME,'--zone='+zone,'--project='+c.PROJECT,'--quiet','--discard-local-ssd=True'],150)
                finalvm=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone)
                (c.R/'provider-final.json').write_text(json.dumps(finalvm,indent=2)+'\n')
                if finalvm.get('lastStartTimestamp')!=p['previous_provider_stop'] and 'provider-start.json' in [x.name for x in c.R.iterdir()]:
                    actual=datetime.fromisoformat(finalvm['lastStopTimestamp']).timestamp()-datetime.fromisoformat(finalvm['lastStartTimestamp']).timestamp()
                    c.record('provider_stopped',actual_seconds=actual,conservative_seconds=math.ceil(actual/100)*100,focused_prior=28700)
            except Exception as error:c.record('provider_stop_check_failed',error_type=type(error).__name__)
        if guard and guard.poll() is None:
            try:guard.wait(timeout=35)
            except subprocess.TimeoutExpired:guard.terminate()
        awake.terminate()


if __name__=='__main__':main(capacity_retry='--capacity-retry' in sys.argv)
