"""Wait through read-only SSH outages, then dispatch complete native20/speech20 once.

Existing committed GPU modules own all qualification gates. This local dispatcher
cannot train, extend, adopt, open sealed fresh inputs, or retry an uncertain launch.
"""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

from portable.context.native_control_copyback import ssh_command

STAGES=('whole_workset','complete40_precise_numerical','complete40_independent_BF16_gradient',
        'combined_readiness','bounded8_training','independent_restore','strict_proof_verification')
REASONING_STAGES=('whole_workset','gradient32','combined_readiness','bounded8_training',
                  'independent_restore','strict_proof')


class ObservationUnavailable(RuntimeError):pass


def ready(value,plan):
    c=value['controller'];birth=value['birth']
    if (c['pid']!=plan['upstream_pid'] or c['source_commit']!=plan.get('upstream_source_commit',plan['source_commit'])
            or c['pod_id']!=plan['pod_id'] or birth not in (None,plan['upstream_birth'])):
        raise ValueError('Exact original upstream identity changed')
    if c['phase']=='complete_restored_pilot_only' and c['passed'] is True:
        reasoning=plan.get('candidate_kind')=='reasoning32'
        stages=REASONING_STAGES if reasoning else STAGES
        exit_key='actual_exit' if reasoning else 'exit_code'
        if ([r['name'] for r in c['stages']]!=list(stages) or any(r.get(exit_key)!=0 for r in c['stages'])
                or reasoning and (c.get('reused_numerical',{}).get('actual_exit')!=0
                    or any(len(c['reused_numerical'].get(k,''))!=64 for k in ('controller_sha256','numerical_sha256')))):
            raise ValueError('All numerical/training/independent-restore stages and exact numerical binding required')
        return birth is None
    if c['phase'].startswith(('failed','upstream_failed')) or birth is None:
        raise ValueError('Upstream actually ended without a complete restored pilot')
    return False


def wait_ready(observe,plan,record,clock=time.time,sleep=time.sleep):
    latest=datetime.fromisoformat(plan['latest_launch_utc']).timestamp()
    while clock()<latest:
        try:value=observe()
        except (ObservationUnavailable,OSError,subprocess.SubprocessError) as error:
            record(phase='waiting_read_retry',last_read_succeeded=False,error_type=type(error).__name__,
                   last_observation_error=str(error)[-1500:])
            sleep(min(30,max(0,latest-clock())));continue
        record(phase='waiting_for_complete_independent_restored_pilot',last_read_succeeded=True,
               observed_upstream_phase=value['controller']['phase'])
        if ready(value,plan):return value
        sleep(min(30,max(0,latest-clock())))
    return None


def native_command(p):
    return [p['native_python'],'-u','-m','portable.context.native_development',
        '--source',p['source'],'--source-commit',p['source_commit'],'--out',p['paired_out'],
        '--asr-python',p['asr_python'],'--deadline-utc',p['deadline_utc'],'--pod-id',p['pod_id'],
        '--gpu-mode','serial-one-gpu','--runtime-profile','container-default',
        '--reasoning32-candidate-proof' if p.get('candidate_kind')=='reasoning32' else '--candidate-proof',
        str(Path(p['upstream'])/'training')]


def launch_payload(p):
    # Transfer runtime/public identifiers only, never local credential paths.
    public={k:p[k] for k in ('native_python','asr_python','source','source_commit','pod_id',
        'paired_out','speech_out','deadline_utc','remote_proxy_url','run_id','compiler_driver')}
    public['native_command']=native_command(p)
    return 'P=json.loads('+repr(json.dumps(public))+')\n'+r'''
from pathlib import Path
import datetime,fcntl,os,subprocess,time,urllib.request
source=Path(P['source']);paired=Path(P['paired_out']);speech=Path(P['speech_out'])
receipt=paired.parent/(paired.name+'-launch.json')
if paired.exists() or speech.exists():raise RuntimeError('Existing evaluation output; never duplicate')
with receipt.open('x') as f:f.write(json.dumps(dict(phase='reserved',pod_id=P['pod_id'],native_command=P['native_command'])))
lock=(paired.parent/'native-control40-gpu-owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
if any(x.strip().isdigit() for x in apps.splitlines()):raise RuntimeError('GPU occupied')
with urllib.request.urlopen(P['remote_proxy_url'].replace('/v1/responses','/health'),timeout=10) as response:h=json.load(response)
if h['run_id']!=P['run_id'] or h['real_requests_transport'] is not True:raise RuntimeError('Actual scoped backend unavailable')
env=os.environ.copy()
for key in list(env):
 if key.startswith(('OPENAI_','RUNPOD_','AWS_')) or key in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN'):env.pop(key)
env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',PYTHONPATH='/workspace/control40-vendor-moshi-rag/moshi',OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',CPATH='/usr/local/cuda/include',TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',TRITON_LIBCUDA_PATH=P['compiler_driver'],DUPLEX_OPENAI_LOOPBACK_PROXY=P['remote_proxy_url'],DUPLEX_OPENAI_PROXY_RUN=P['run_id'],TORCHINDUCTOR_CACHE_DIR='/opt/control40-native20-v1-inductor',TRITON_CACHE_DIR='/opt/control40-native20-v1-triton')
with Path(str(paired)+'-outer.log').open('x') as log:
 native=subprocess.Popen(P['native_command'],cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
state=dict(phase='native_launched_waiting_initial_execution',native_pid=native.pid,native_command=P['native_command'],pod_id=P['pod_id'],source_commit=P['source_commit'],launched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),shared_gpu_lease_inherited=True)
receipt.write_text(json.dumps(state,indent=2)+'\n');until=time.monotonic()+60
while not (paired/'execution.json').is_file():
 if native.poll() is not None or time.monotonic()>until:raise RuntimeError('Inspect native launch receipt before any retry')
 time.sleep(.5)
f=Path('/proc/'+str(native.pid)+'/stat').read_text().rsplit(')',1)[1].split()
if f[0]=='Z':raise RuntimeError('Native controller ended at dispatch')
birth=int(f[19]);current=json.loads((paired/'execution.json').read_text())
if current['pid']!=native.pid or current['source_commit']!=P['source_commit'] or current['phase']=='failed':raise RuntimeError('Native initial identity failed')
command=[P['native_python'],'-u','-m','portable.context.native_speech_followup','--source',str(source),'--paired',str(paired),'--python',P['asr_python'],'--out',str(speech),'--controller-pid',str(native.pid),'--controller-birth',str(birth),'--paired-source-commit',P['source_commit'],'--deadline-utc',P['deadline_utc']]
with Path(str(speech)+'-outer.log').open('x') as log:
 auditor=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
state.update(phase='native20_and_independent_speech_followup_launched',native_birth=birth,speech_pid=auditor.pid,speech_command=command,behavior_approved=False,model_adopted=False)
receipt.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state))
'''


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--plan',type=Path,required=True)
    a.add_argument('--status',type=Path,required=True);args=a.parse_args();p=json.loads(args.plan.read_text())
    if args.status.exists():raise ValueError('Existing dispatcher status; inspect before replacing')
    from evals.recovery_transport import endpoint,PROXY_ENV,RUN_ENV
    for url in (p['local_proxy_url'],p['remote_proxy_url']):endpoint({PROXY_ENV:url,RUN_ENV:p['run_id']})
    ssh=ssh_command(Path(p['key']),Path(p['known_hosts']),p['host'],p['port'])
    state=dict(schema='dl-native-control40-development-followup-v2',pid=os.getpid(),
        pod_id=p['pod_id'],source_commit=p['source_commit'],upstream_pid=p['upstream_pid'],
        upstream_birth=p['upstream_birth'],latest_launch_utc=p['latest_launch_utc'],
        evaluation_deadline_utc=p['deadline_utc'],paired_out=p['paired_out'],speech_out=p['speech_out'],
        local_dispatcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        may_extend_or_adopt=False,may_read_sealed_fresh_inputs=False,planned_optimizer_updates=0)
    def record(**values):
        state.update(values,at_utc=datetime.now(timezone.utc).isoformat())
        temp=args.status.with_suffix('.partial');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(args.status)
    def remote(code,timeout=35):
        result=subprocess.run(ssh+[p['host'],'python3 -'],input='import json\n'+code,
                              capture_output=True,text=True,timeout=timeout)
        if result.returncode:raise ObservationUnavailable(result.stderr[-1800:])
        return json.loads(result.stdout)
    query='''from pathlib import Path
q=Path('/proc/'+str(PID)+'/stat');birth=None
if q.exists():
 f=q.read_text().rsplit(')',1)[1].split();birth=None if f[0]=='Z' else int(f[19])
print(json.dumps(dict(controller=json.loads(Path(REPORT).read_text()),birth=birth)))
'''.replace('PID',str(p['upstream_pid'])).replace('REPORT',repr(str(Path(p['upstream'])/'controller.json')))
    record(phase='waiting_for_complete_independent_restored_pilot')
    try:
        if wait_ready(lambda:remote(query),p,record) is None:
            record(phase='not_started_insufficient_remaining_full20_window');return
        now=datetime.now(timezone.utc);end=datetime.fromisoformat(p['deadline_utc'])
        if (end-now).total_seconds()<40*60:raise ValueError('Complete evaluation window required')
        g=json.loads(Path(p['guard']).read_text())
        if (g['phase']!='armed' or g['pod_id']!=p['pod_id']
                or not 0<=(now-datetime.fromisoformat(g['at_utc'])).total_seconds()<=90
                or datetime.fromisoformat(g['deadline_utc'])<end+timedelta(seconds=120)):
            raise ValueError('Actual current same-Pod cost guard must outlast evaluation')
        os.kill(g['pid'],0);proxy=json.loads(Path(p['proxy_receipt']).read_text());os.kill(proxy['pid'],0)
        if (proxy['run_id']!=p['run_id'] or datetime.fromisoformat(proxy['deadline_utc'])
                <end+timedelta(seconds=60)):
            raise ValueError('Exact live proxy must outlast evaluation')
        with urllib.request.urlopen(p['local_proxy_url'].replace('/v1/responses','/health'),timeout=10) as response:h=json.load(response)
        if h['run_id']!=p['run_id'] or h['real_requests_transport'] is not True or h['actual_requests']>=proxy['maximum_requests']:
            raise ValueError('Actual backend transport unavailable')
        record(phase='launching_once_inspect_remote_receipt_before_any_retry',native_command=native_command(p))
        result=remote(launch_payload(p),timeout=100)
        record(phase='launched_native20_and_independent_speech',launch=result)
    except BaseException as error:
        record(phase='failed_inspect_before_retry',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__':main()
