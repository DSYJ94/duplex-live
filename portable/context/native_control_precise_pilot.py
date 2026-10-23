"""One sequential whole40 reference/gradient/retained pilot/restore chain.

No extension or adoption. Each numerical, gradient, readiness, fit and restore
stage has a distinct process, an immutable source and a real bounded deadline.
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

from portable.context.native_control_preflight import git
from portable.context.native_control_training_followup import arguments,training_environment,digest
from portable.context.native_preflight_takeover import process


def diagnostic_gate(path,original,attention_code):
    from evals.native_control_numerical_precise import ORIGINAL_FAILURE_SHA
    d=json.loads(path.read_text())
    if (digest(original)!=ORIGINAL_FAILURE_SHA or d['phase']!='complete_diagnostic_only'
            or d['case']!='native-control3-train-029' or d['optimizer_updates']!=0
            or d['training_approved'] is not False or d['complete40_retested'] is not False
            or d['diagnostic_only'] is not True or d['within_original_threshold'] is not True
            or d['all_model_parameters_unchanged'] is not True or d['execution_policy_unchanged'] is not True
            or d['original_numerical_sha256']!=ORIGINAL_FAILURE_SHA
            or d['execution_source_commit']!='455975c8a882648cb4ae4a18dd478aa36a9b344c'
            or d['reference_module_sha256']!=digest(attention_code)
            or set(d['comparisons'])!={'hidden','text_logits'}
            or any(not 0<=r['maximum_absolute_error']<=.001 for r in d['comparisons'].values())):
        raise ValueError('Completed immutable diagnostic supports only a NEW full40 audit, never training itself')
    return d


def commands(a):
    prefix=[str(a.python),'-u','-m'];out=a.out
    shared=['--cache',str(a.cache),'--assets',str(a.root/'runs/context-spanning-assets-7454205/manifest.json'),
            '--base',str(a.root/'models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309'),
            '--proof',str(a.root/'runs/context-training-proof-7510331')]
    base=arguments(a.root,a.source,a.cache,out,out,out.parent)
    base+=['--native-audio49-prepared',str(a.prepared49),'--native-audio49-source-inventory',str(a.original_inventory)]
    fit=prefix+['portable.context.native_training_cached']+base
    ready=out/'combined-readiness';proof=out/'training'
    return [
        ('whole_workset',prefix+['evals.native_control_snapshot','--inventory',str(a.workset_inventory)]),
        ('complete40_precise_numerical',prefix+['evals.native_control_numerical_precise']+shared+
         ['--engine-inputs',str(out/'engine'),'--out',str(out/'numerical'),
          '--original-failure',str(a.original_failure),'--deadline-utc',a.audit_deadline_utc]),
        ('complete40_independent_BF16_gradient',prefix+['evals.native_control_gradient']+shared+
         ['--engine-inputs',str(out/'engine'),'--numerical',str(out/'numerical'),'--out',str(out/'gradient')]),
        ('combined_readiness',fit+['--out',str(ready),'--native-audio49-preflight-only','--native-audio49-stream-gradient-readiness']),
        ('bounded8_training',fit+['--out',str(proof),'--native-audio49-preflight',str(ready)]),
        ('independent_restore',fit+['--out',str(proof),'--native-audio49-preflight',str(ready),'--reload']),
        ('strict_proof_verification',[str(a.python),'-u','-c',
         'from pathlib import Path; from evals.native_control_stage import verify_checkpoint; verify_checkpoint(Path('+repr(str(proof))+')); print("Full retained154/native49/control40 proof and fresh restore verified")']),
    ]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','python','root','cache','engine','original-failure','diagnostic','compiler',
              'workset-inventory','original-inventory','prepared49','cost-guard-receipt','out'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--deadline-utc',required=True)
    p.add_argument('--audit-deadline-utc',required=True)
    a=p.parse_args();deadline=datetime.fromisoformat(a.deadline_utc.replace('Z','+00:00')).timestamp()
    audit_deadline=datetime.fromisoformat(a.audit_deadline_utc.replace('Z','+00:00')).timestamp()
    if not time.time()+120<audit_deadline<deadline-120:raise ValueError('Separate audit and total finite future deadlines required')
    def clean():
        if git(a.source,'rev-parse','HEAD')!=a.source_commit or git(a.source,'status','--porcelain'):
            raise ValueError('Exact clean immutable execution source required')
    clean()
    guard=json.loads(a.cost_guard_receipt.read_text())
    if (guard['phase']!='armed' or guard['pod_id']!='sjf72o8v64mjvb'
            or time.time()-datetime.fromisoformat(guard['at_utc']).timestamp()>90
            or datetime.fromisoformat(guard['deadline_utc']).timestamp()<deadline+120):
        raise ValueError('Actual fresh same-Pod budget guard must outlast all stages')
    diagnostic=diagnostic_gate(a.diagnostic,a.original_failure,a.source/'evals/native_control_attention_reference.py')
    actual=process(diagnostic['pid'])
    if actual is not None and actual['state']!='Z':raise ValueError('Original diagnostic still live; never duplicate GPU work')
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if any(x.strip().isdigit() for x in gpu.splitlines()):raise RuntimeError('GPU occupied; no concurrent stage allowed')
    compiler=json.loads(a.compiler.read_text())
    if (compiler['packages_changed'] is not False or not compiler['actual_compiled_bf16_gpu_kernel_passed']
            or not compiler['actual_triton_driver_compile_passed']):raise ValueError('Pinned compiled BF16 CUDA runtime required')
    for n in ('header','assembler'):
        if digest(compiler[n])!=compiler[n+'_sha256']:raise ValueError('Changed compiler path')
    a.out.mkdir(parents=True,exist_ok=False)
    lock=(a.out.parent/'native-control40-gpu-owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (a.out/'engine').symlink_to(a.engine.resolve(),target_is_directory=True)
    stages=commands(a)
    state=dict(schema='dl-native-control40-precise-pilot-controller-v1',pid=os.getpid(),source_commit=a.source_commit,
               execution_source=str(a.source),deadline_utc=a.deadline_utc,audit_deadline_utc=a.audit_deadline_utc,
               diagnostic_sha256=digest(a.diagnostic),original_failed_numerical_sha256=digest(a.original_failure),
               engine_alias_target=str(a.engine.resolve()),phase='prepared',stages=[],planned_commands=dict(stages),
               optimizer_update_limit=8,model_adopted=False,behavior_qualified=False,extension_approved=False,passed=False)
    def save():
        state['at_utc']=datetime.now(timezone.utc).isoformat()
        temp=a.out/'controller.partial';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(a.out/'controller.json')
    save();child=None
    def stop():
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    def interrupted(signum,_):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    env=training_environment(compiler,os.environ,a.source_commit)
    if env.get('PYTHONPATH')!='/workspace/control40-vendor-moshi-rag/moshi':
        raise ValueError('Pinned native legacy import environment required')
    try:
        for name,command in stages:
            clean();remaining=deadline-time.time()
            if remaining<=0:raise TimeoutError('Pilot controller deadline reached')
            state['phase']=name;stage=dict(name=name,command=command,started_utc=datetime.now(timezone.utc).isoformat())
            state['stages'].append(stage);save()
            with (a.out/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=a.source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                stage['pid']=child.pid;actual=process(child.pid)
                if actual is not None:stage['birth']=actual['birth']
                save()
                code=child.wait(timeout=remaining);stage.update(exit_code=code,ended_utc=datetime.now(timezone.utc).isoformat());save()
                if code!=0:raise RuntimeError(name+' failed with exit '+str(code))
        state.update(phase='complete_restored_pilot_only',passed=True);save()
    except BaseException as error:
        stop();state.update(phase='failed',error_type=type(error).__name__,error=str(error));save();raise


if __name__=='__main__':main()
