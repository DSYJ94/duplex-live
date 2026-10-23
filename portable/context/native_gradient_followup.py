"""Wait for the exact live whole40 preflight, then run whole40 BF16 gradients.

No optimizer work. The original controller and immutable payload are not changed
or restarted. A nonblocking lock prevents duplicate follow-up submissions.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def process_birth(pid):
    path=Path('/proc')/str(pid)/'stat'
    try:fields=path.read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return None
    return None if fields[0]=='Z' else int(fields[19])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','python','preflight','compiler','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--source-commit',required=True)
    ap.add_argument('--controller-pid',type=int,required=True)
    ap.add_argument('--controller-birth',type=int,required=True)
    ap.add_argument('--deadline-utc',required=True)
    args=ap.parse_args()
    deadline=datetime.fromisoformat(args.deadline_utc.replace('Z','+00:00')).timestamp()
    args.out.mkdir(parents=True,exist_ok=False)
    lock=(args.preflight/'gradient-followup.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(schema='dl-native-control40-gradient-followup-v1',pid=os.getpid(),
               source_commit=args.source_commit,preflight=str(args.preflight),
               original_controller_pid=args.controller_pid,original_controller_birth=args.controller_birth,
               phase='waiting_for_exact_live_preflight',deadline_utc=args.deadline_utc,
               optimizer_updates=0,training_approved=False,behavior_approved=False)
    def save():
        state['at_utc']=datetime.now(timezone.utc).isoformat()
        temp=args.out/'controller.partial';temp.write_text(json.dumps(state,indent=2)+'\n')
        temp.replace(args.out/'controller.json')
    child=None
    def stop(*_):
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    signal.signal(signal.SIGTERM,lambda a,b:(stop(),exit(128+a)))
    save()
    try:
        if process_birth(args.controller_pid)!=args.controller_birth:
            raise RuntimeError('Exact original preflight controller is not live at dispatch')
        while time.time()<deadline:
            pre=json.loads((args.preflight/'controller.json').read_text())
            if pre['pid']!=args.controller_pid or pre['source_commit']!=args.source_commit:
                raise RuntimeError('Original preflight identity changed')
            if pre['phase']=='complete_numerical_only':
                if (pre['passed'] is not True or [s['name'] for s in pre['stages']]!=['codec_verify','engine','numerical']
                        or any(s.get('exit_code')!=0 for s in pre['stages'])):
                    raise RuntimeError('Whole40 original preflight did not finish successfully')
                if process_birth(args.controller_pid) is None:break
            elif process_birth(args.controller_pid)!=args.controller_birth:
                raise RuntimeError('Original preflight terminated without complete numerical approval')
            state['observed_original_phase']=pre['phase'];save();time.sleep(10)
        else:raise TimeoutError('No whole40 numerical completion before common deadline')
        if (subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()!=args.source_commit
                or subprocess.check_output(['git','-C',str(args.source),'status','--porcelain'],text=True).strip()):
            raise RuntimeError('Immutable gradient payload changed')
        compiler=json.loads(args.compiler.read_text())
        if not compiler['actual_compiled_bf16_gpu_kernel_passed'] or compiler['packages_changed'] is not False:
            raise RuntimeError('Pinned actual CUDA compiler check required')
        for name in ('header','assembler'):
            if hashlib.sha256(Path(compiler[name]).read_bytes()).hexdigest()!=compiler[name+'_sha256']:
                raise RuntimeError('Pinned CUDA tool changed')
        from portable.context.native_control_preflight import native_environment
        env=native_environment(compiler,os.environ);env['CUDA_VISIBLE_DEVICES']='0'
        numerical=pre['stages'][-1]['command']
        def argument(name):return numerical[numerical.index('--'+name)+1]
        cmd=[str(args.python),'-u','-m','evals.native_control_gradient',
             '--cache',argument('cache'),'--engine-inputs',str(args.preflight/'engine'),
             '--numerical',str(args.preflight/'numerical'),'--assets',argument('assets'),
             '--base',argument('base'),'--proof',argument('proof'),'--out',str(args.out/'gradient')]
        state.update(phase='whole40_bf16_gradient',command=cmd);save()
        with (args.out/'gradient.log').open('x') as log:
            child=subprocess.Popen(cmd,cwd=args.source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['child_pid']=child.pid;save()
            child.wait(timeout=max(1,deadline-time.time()))
        if child.returncode:raise RuntimeError('Whole40 gradient stage exited '+str(child.returncode))
        result=json.loads((args.out/'gradient/gradient.json').read_text())
        if not result['passed'] or not result['complete40'] or result['optimizer_updates']!=0:
            raise RuntimeError('Whole40 gradient readiness did not pass')
        state.update(phase='complete_gradient_only',passed=True,gradient_report_sha256=
            hashlib.sha256((args.out/'gradient/gradient.json').read_bytes()).hexdigest())
        save()
    except BaseException as error:
        stop();state.update(phase='failed',passed=False,error_type=type(error).__name__,error=str(error));save();raise


if __name__=='__main__':main()
