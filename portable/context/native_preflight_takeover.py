"""Replace only the audit supervisor, preserving its exact running CUDA child.

The old supervisor is paused while its child continues, so its superseded timeout
cannot discard the completed work. This supervisor reads the actual zombie exit
status before retiring that parent. A new cost guard must already be armed.
Failures resume the old supervisor; no numerical stage is restarted or approved
from a subset. All original receipts remain intact, including retirement state.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from portable.context.native_control_preflight import native_environment,git


def process(pid):
    try:
        fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return None
    return dict(pid=pid,state=fields[0],birth=int(fields[19]),exit_status=int(fields[49]))


def exact(pid,birth):
    p=process(pid)
    if p is None or p['birth']!=birth:raise RuntimeError('Process identity changed: '+str(pid))
    return p


def pause(pid,birth):
    if exact(pid,birth)['state']=='Z':raise RuntimeError('Supervisor already terminal')
    os.kill(pid,signal.SIGSTOP)
    end=time.monotonic()+2
    while time.monotonic()<end:
        if exact(pid,birth)['state'] in ('T','t'):return
        time.sleep(.01)
    os.kill(pid,signal.SIGCONT)
    raise RuntimeError('Supervisor pause not confirmed')


def resume(pid,birth):
    p=process(pid)
    if p and p['birth']==birth and p['state']!='Z':os.kill(pid,signal.SIGCONT)


def retire(pid,birth):
    p=process(pid)
    if p is None or p['state']=='Z':return
    exact(pid,birth);os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
    end=time.monotonic()+15
    while time.monotonic()<end:
        p=process(pid)
        if p is None or p['state']=='Z':return
        if p['birth']!=birth:raise RuntimeError('Retired supervisor PID was reused')
        time.sleep(.05)
    raise RuntimeError('Old supervisor has not retired; do not launch another GPU stage')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('payload','python','preflight','compiler','out','cost-guard-receipt'):
        ap.add_argument('--'+n,type=Path,required=True)
    for n in ('controller-pid','controller-birth','numerical-pid','numerical-birth'):
        ap.add_argument('--'+n,type=int,required=True)
    ap.add_argument('--payload-commit',required=True)
    ap.add_argument('--original-commit',required=True)
    ap.add_argument('--deadline-utc',required=True)
    args=ap.parse_args()
    deadline=datetime.fromisoformat(args.deadline_utc.replace('Z','+00:00')).timestamp()
    if (deadline<=time.time()+60 or git(args.payload,'rev-parse','HEAD')!=args.payload_commit
            or git(args.payload,'status','--porcelain')):
        raise ValueError('Exact clean payload and future bounded deadline required')
    guard=json.loads(args.cost_guard_receipt.read_text())
    if (guard['phase']!='armed' or guard['pod_id']!='sjf72o8v64mjvb'
            or time.time()-datetime.fromisoformat(guard['at_utc']).timestamp()>90
            or datetime.fromisoformat(guard['deadline_utc']).timestamp()<deadline+30):
        raise ValueError('Longer same-Pod cost guard must already be armed')
    original=json.loads((args.preflight/'controller.json').read_text())
    if (original['pid']!=args.controller_pid or original['source_commit']!=args.original_commit
            or original['phase']!='numerical' or len(original['stages'])!=3
            or any(s.get('exit_code')!=0 for s in original['stages'][:2])
            or original['stages'][2]['pid']!=args.numerical_pid
            or git(Path(original['execution_source']),'rev-parse','HEAD')!=args.original_commit):
        raise ValueError('Exact live original numerical stage and passed codec/Engine required')
    exact(args.controller_pid,args.controller_birth);exact(args.numerical_pid,args.numerical_birth)
    args.out.mkdir(parents=True,exist_ok=False)
    locks=[]
    for name in ('numerical-takeover.lock','gradient-followup.lock'):
        lock=(args.preflight/name).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    state=dict(schema='dl-native-control40-supervisor-takeover-v1',pid=os.getpid(),
               controller_commit=git(Path(__file__).resolve().parents[2],'rev-parse','HEAD'),
               payload_commit=args.payload_commit,original_commit=args.original_commit,
               adopted_numerical_pid=args.numerical_pid,adopted_numerical_birth=args.numerical_birth,
               original_controller_pid=args.controller_pid,original_controller_birth=args.controller_birth,
               original_deadline_utc=original['deadline_utc'],deadline_utc=args.deadline_utc,
               phase='preparing_takeover',numerical_restarted=False,optimizer_updates=0,
               passed=False,training_approved=False,behavior_approved=False)
    (args.out/'original-controller-before-takeover.json').write_text(json.dumps(original,indent=2)+'\n')
    def save():
        state['at_utc']=datetime.now(timezone.utc).isoformat()
        temp=args.out/'controller.partial';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(args.out/'controller.json')
    child=None;paused=False
    def interrupted(signum,_):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    save()
    try:
        pause(args.controller_pid,args.controller_birth);paused=True
        state.update(phase='adopted_original_numerical',old_supervisor_pause_confirmed=True);save()
        while time.time()<deadline:
            actual=exact(args.numerical_pid,args.numerical_birth)
            if actual['state']=='Z':
                state['actual_numerical_exit_code']=os.waitstatus_to_exitcode(actual['exit_status']);save()
                if state['actual_numerical_exit_code']!=0:raise RuntimeError('Original numerical child failed')
                break
            if exact(args.controller_pid,args.controller_birth)['state'] not in ('T','t'):
                raise RuntimeError('Superseded supervisor resumed before child completion')
            try:
                report=json.loads((args.preflight/'numerical/numerical.json').read_text())
                state['numerical_cases_observed']=len(report['cases'])
            except (FileNotFoundError,json.JSONDecodeError):pass
            save();time.sleep(5)
        else:
            os.killpg(args.numerical_pid,signal.SIGTERM)
            raise TimeoutError('Extended numerical deadline reached')
        report=json.loads((args.preflight/'numerical/numerical.json').read_text())
        if not report['passed'] or not report['complete40'] or report['optimizer_updates']!=0:
            raise RuntimeError('Original complete40 numerical proof failed')
        retire(args.controller_pid,args.controller_birth);paused=False
        state.update(old_supervisor_retired_after_child_exit=True,phase='gradient40');save()
        if git(args.payload,'rev-parse','HEAD')!=args.payload_commit or git(args.payload,'status','--porcelain'):
            raise RuntimeError('Gradient execution source changed')
        compiler=json.loads(args.compiler.read_text());env=native_environment(compiler,os.environ)
        env['CUDA_VISIBLE_DEVICES']='0'
        cmd=original['stages'][2]['command']
        def value(name):return cmd[cmd.index('--'+name)+1]
        command=[str(args.python),'-u','-m','evals.native_control_gradient',
                 '--cache',value('cache'),'--engine-inputs',str(args.preflight/'engine'),
                 '--numerical',str(args.preflight/'numerical'),'--assets',value('assets'),
                 '--base',value('base'),'--proof',value('proof'),'--out',str(args.out/'gradient')]
        with (args.out/'gradient.log').open('x') as log:
            child=subprocess.Popen(command,cwd=args.payload,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(gradient_pid=child.pid,gradient_command=command);save()
            child.wait(timeout=max(1,deadline-time.time()))
        if child.returncode:raise RuntimeError('Complete40 gradient stage failed: '+str(child.returncode))
        result=json.loads((args.out/'gradient/gradient.json').read_text())
        if not result['passed'] or not result['complete40']:raise RuntimeError('Incomplete gradient40')
        state.update(phase='complete_gradient_only',passed=True);save()
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        state.update(phase='failed',error_type=type(error).__name__,error=str(error));save();raise
    finally:
        if paused:resume(args.controller_pid,args.controller_birth)


if __name__=='__main__':main()
