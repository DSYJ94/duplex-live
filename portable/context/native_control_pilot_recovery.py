"""Restart only a deadline-stopped readiness stage; never reuse partial49 approval.

The first three completed C9 stages retain their original actual exit evidence.
Only output/readiness paths change; all numerical, gradient, fit, restore and
behavior requirements stay intact. One same-Pod GPU lease; no provider calls.
"""
import argparse
from datetime import datetime,timezone
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from portable.context.native_control_pilot_takeover import guard_gate,gradient_gate
from portable.context.native_control_training_followup import digest,training_environment
from portable.context.native_control_development_wait import STAGES
from portable.context.native_preflight_takeover import process


def recovered_commands(previous,out):
    if (previous['source_commit']!='c9efaaac24d983342f7eded46617e6450bc102ce'
            or previous['pod_id']!='4yx8vahdmjsylx' or previous['phase']!='failed'
            or previous['error_type']!='TimeoutError'
            or previous['error']!='Extended adopted-stage deadline reached'
            or previous['optimizer_update_limit']!=8 or previous['passed']is not False
            or len(previous['stages'])!=4 or list(previous['planned_commands'])!=list(STAGES)
            or [r['name']for r in previous['stages']]!=list(STAGES[:4])
            or any(r.get('exit_code')!=0 for r in previous['stages'][:3])
            or any(r['command']!=previous['planned_commands'][r['name']]for r in previous['stages'])):
        raise ValueError('Exact terminal zero-update deadline failure with three actual passed predecessors required')
    commands=copy.deepcopy(previous['planned_commands']);old=commands['combined_readiness']
    old_ready=old[old.index('--out')+1];old_fit=commands['bounded8_training'][commands['bounded8_training'].index('--out')+1]
    replacements={old_ready:str(out/'combined-readiness'),old_fit:str(out/'training')}
    for name in STAGES[3:6]:
        commands[name]=[replacements.get(x,x)for x in commands[name]]
    commands[STAGES[6]]=[x.replace(repr(old_fit),repr(str(out/'training')))for x in commands[STAGES[6]]]
    for name in STAGES[3:6]:
        original=previous['planned_commands'][name];changed=commands[name]
        if len(original)!=len(changed):raise ValueError('Argument shape changed')
        for i,(a,b)in enumerate(zip(original,changed)):
            if a!=b and (i==0 or original[i-1]not in('--out','--native-audio49-preflight')or a not in replacements):
                raise ValueError('Only versioned output/readiness paths may change')
    return commands


def run(a):
    deadline=datetime.fromisoformat(a.deadline_utc).timestamp()
    if not time.time()+120<deadline<time.time()+18*3600:raise ValueError('Finite funded future deadline required')
    if digest(a.previous)!=a.previous_sha256:raise ValueError('Stopped controller snapshot changed')
    previous=json.loads(a.previous.read_text());commands=recovered_commands(previous,a.out)
    guard_gate(json.loads(a.cost_guard.read_text()),previous['pod_id'],deadline)
    source=Path(previous['execution_source']);commit=previous['source_commit']
    def clean():
        if (subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=commit
                or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()):
            raise ValueError('Immutable exact C9 execution source required')
    clean()
    for pid in (41540,21063,70675):
        p=process(pid)
        if p is not None and p['state']!='Z':raise ValueError('Previous native process still live')
    lease=(a.out.parent/'native-control40-gpu-owner.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if any(x.strip().isdigit()for x in gpu.splitlines()):raise ValueError('GPU occupied; never duplicate work')
    gradient=commands[STAGES[2]]
    arg=lambda n:Path(gradient[gradient.index('--'+n)+1])
    gradient_gate(json.loads((arg('out')/'gradient.json').read_text()),arg('engine-inputs'),arg('numerical'),commit)
    compiler=json.loads(a.compiler.read_text())
    if (compiler['packages_changed']is not False or not compiler['actual_compiled_bf16_gpu_kernel_passed']
            or not compiler['actual_triton_driver_compile_passed']
            or any(digest(compiler[n])!=compiler[n+'_sha256']for n in('header','assembler'))):
        raise ValueError('Unchanged compiled BF16 runtime required')
    env=training_environment(compiler,os.environ,commit)
    a.out.mkdir(parents=True,exist_ok=False)
    state=dict(schema='dl-native-control40-precise-recovery-controller-v1',pid=os.getpid(),
        birth=process(os.getpid())['birth'],source_commit=commit,pod_id=previous['pod_id'],
        execution_source=str(source),deadline_utc=a.deadline_utc,phase='verifying_prior_whole_workset',
        recovery_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        previous_controller_sha256=digest(a.previous),previous_controller=str(a.previous),
        stages=copy.deepcopy(previous['stages'][:3]),planned_commands=commands,
        completed_predecessors_reused_with_original_actual_exits=True,partial49_approval_reused=False,
        readiness_all49_restarted=True,optimizer_update_limit=8,model_adopted=False,
        behavior_qualified=False,extension_approved=False,passed=False)
    def save():
        state['at_utc']=datetime.now(timezone.utc).isoformat();p=a.out/'controller.partial'
        p.write_text(json.dumps(state,indent=2)+'\n');p.replace(a.out/'controller.json')
    save();child=None
    def interrupt(signum,_):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        subprocess.run(commands[STAGES[0]],cwd=source,env=env,check=True,timeout=300,
                       stdout=(a.out/'whole_workset_recheck.log').open('x'),stderr=subprocess.STDOUT)
        state['prior_workset_independently_rechecked']=True;save()
        for name in STAGES[3:]:
            clean()
            if time.time()>=deadline:raise TimeoutError('Funded recovery deadline reached')
            row=dict(name=name,command=commands[name],started_utc=datetime.now(timezone.utc).isoformat())
            state['phase']=name;state['stages'].append(row);save()
            with(a.out/(name+'.log')).open('x')as log:
                child=subprocess.Popen(commands[name],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                row.update(pid=child.pid,birth=process(child.pid)['birth']);save()
                code=child.wait(timeout=max(1,deadline-time.time()))
            row.update(exit_code=code,ended_utc=datetime.now(timezone.utc).isoformat());save()
            if code:raise RuntimeError(name+' failed with '+str(code))
        state.update(phase='complete_restored_pilot_only',passed=True);save();return state
    except BaseException as e:
        if child is not None and child.poll()is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        state.update(phase='failed',error_type=type(e).__name__,error=str(e));save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in('previous','compiler','cost-guard','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--deadline-utc',required=True);p.add_argument('--previous-sha256',required=True);print(json.dumps(run(p.parse_args())))
