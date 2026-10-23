"""Follow the exact existing audit with one retained154/native49/control40 pilot.

The execution checkout stays immutable across readiness, fit and fresh restore.
This controller owns no provider credentials and cannot extend or adopt a model.
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

from portable.context.native_control_preflight import git, native_environment
from portable.context.native_preflight_takeover import process

AUDIT_SHA='6b03571ab251a7ba22d144ade4ab55b5c575273b5f588b9c61838ca77a0c00f2'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arguments(root, payload, cache, preflight, supervisor, evidence):
    """Reconstruct the exact old28 order from its hash-bound native520 data."""
    parent=root/'runs/context-training-proof-7510331'
    report=json.loads((parent/'report.json').read_text())
    data=parent/'data.json'
    rows=json.loads(data.read_text())['components']
    audit=root/'runs/native-audio49-audit-7572398'
    if (digest(data)!=report['data_sha256'] or len(rows)!=28
            or len({r['component'] for r in rows})!=28 or report['optimizer_steps']!=520
            or digest(audit/'audio49-audit.json')!=AUDIT_SHA):
        raise ValueError('Exact native520 original28 data and approved full49 audit required')
    values=[
        ('assets',root/'runs/context-spanning-assets-7454205/manifest.json'),
        ('base',root/'models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309'),
        ('voice-reference',root/'runs/voice-quality-7414595/codec/reference.wav'),
        ('voice-provenance',root/'runs/voice-quality-7414595/codec/report.json'),
        *[('cache',root/'runs'/n) for n in ('recovery-mixed-codes-7411513','recovery-codes-7434926','recovery-codes-7435538')],
        ('request-policy','native-prelead-epad'),('actual-source-data',root/'runs/recovery-data-7407554'),
        ('warm-start-proof',parent),('actual-fit-updates',8),
        ('actual-sampling','balanced-kind'),('actual-retention-sampling','balanced-kind'),
        ('actual-base-retention-weight',1),('actual-retention-teacher-weight',1),
        ('actual-native-teacher-loss','balanced-spoken'),('actual-response-first-word-weight',16),
        ('actual-native-teacher-coverage','all-approved'),
        ('actual-native-teacher-cache',root/'runs/native-old25-codes-7543730'),
        ('actual-native-teacher-approval',root/'runs/native-old25-native-approval-7544805/report.json'),
        ('actual-native-teacher-scope',root/'runs/native-old25-native-scope-7543889/report.json'),
        ('actual-native-teacher-preflight',root/'runs/native-old25-native-math-7545975'),
        ('native-v2-cache',root/'runs/native-v2-codes-7545905'),
        ('native-v2-scope',root/'runs/native-v2-native-scope-7545907/report.json'),
        ('native-v2-approval',root/'runs/native-v2-native-approval-7545909/report.json'),
        ('native-v2-preflight',root/'runs/native-v2-native-math-7546608'),
        ('native-v2-learning-rate','2e-6'),('native-v2-warm-ordinary-weight',64),
        ('native-v2-warm-ordinary-approval',root/'runs/native-v2-retention-drift-audit-7554946/native-v2-retention-drift-audit.json'),
        ('native-audio49-audit',audit),('native-audio49-weight',1),('native-audio49-step-scale',.1),
        *[('actual-mixed-data',r['component']) for r in rows],
        ('actual-temporal-preflight',report['actual_temporal_preflight']),
        ('control40-cache',cache),('control40-engine',preflight/'engine'),
        ('control40-numerical',preflight/'numerical'),('control40-gradient',supervisor/'gradient'),
        ('control40-parent-review',payload/'docs/NATIVE-CONTROL40-PARENT-SELECTION-2026-10-23.json'),
        ('control40-development-evidence',evidence),
    ]
    result=['--'+n for n in ('actual-temporal-fit','native-v2-full24-per8','native-v2-warm-ordinary-clock',
                             'native-audio49-parent-adam','native-audio49-complete-batch','native-audio49-stream-auxiliary')]
    for name,value in values:result.extend(['--'+name,str(value)])
    return result


def training_environment(compiler, inherited, commit):
    env=native_environment(compiler,inherited)
    # Training requires public pinned assets, never provider or worker access.
    for key in list(env):
        if key.startswith(('OPENAI_','RUNPOD_','AWS_')) or key in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN'):
            env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES='0',DL_TASK_COMMIT=commit)
    return env


def gradient_complete(state, pid, birth, commit):
    if (state['pid']!=pid or state['payload_commit']!=commit
            or state.get('adopted_numerical_birth')!=16421263):
        raise ValueError('Exact adopted numerical/gradient controller changed')
    actual=process(pid)
    if actual and actual['birth']!=birth:raise ValueError('Audit supervisor PID was reused')
    if state['phase']=='failed':raise RuntimeError('Existing numerical/gradient stage failed: '+state.get('error',''))
    if state['phase']=='complete_gradient_only':
        if (state['passed'] is not True or state['actual_numerical_exit_code']!=0
                or not state['old_supervisor_retired_after_child_exit']):
            raise ValueError('Incomplete original numerical/gradient completion')
        return True
    if actual is None or actual['state']=='Z':
        raise RuntimeError('Original audit supervisor terminated before successful completion')
    return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('payload','python','root','cache','preflight','supervisor','compiler','inventory','out','cost-guard-receipt'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--payload-commit',required=True)
    p.add_argument('--supervisor-payload-commit',
                   help='Explicit original audit payload when a metadata-only dependency repair uses a later training source')
    p.add_argument('--supervisor-pid',type=int,required=True)
    p.add_argument('--supervisor-birth',type=int,required=True)
    p.add_argument('--deadline-utc',required=True)
    p.add_argument('--prepare-only',action='store_true',help='Validate command/data bindings without dispatching any stage')
    p.add_argument('--immutable-input-hashes',action='store_true',
                   help='Full first-read hashes plus fail-closed identity checks for immutable inputs; all validation still runs')
    p.add_argument('--native-audio49-prepared',type=Path)
    p.add_argument('--native-audio49-source-inventory',type=Path)
    a=p.parse_args();deadline=datetime.fromisoformat(a.deadline_utc.replace('Z','+00:00')).timestamp()
    if bool(a.native_audio49_prepared)!=bool(a.native_audio49_source_inventory):
        raise ValueError('Prepared49 inputs and exact original source inventory must be supplied together')
    audit_commit=a.supervisor_payload_commit or a.payload_commit
    def clean():
        if git(a.payload,'rev-parse','HEAD')!=a.payload_commit or git(a.payload,'status','--porcelain'):
            raise ValueError('Exact immutable execution payload required')
    clean()
    if deadline<time.time()+60:raise ValueError('Future finite execution deadline required')
    guard=json.loads(a.cost_guard_receipt.read_text())
    if (guard['phase']!='armed' or guard['pod_id']!='sjf72o8v64mjvb'
            or time.time()-datetime.fromisoformat(guard['at_utc']).timestamp()>90
            or datetime.fromisoformat(guard['deadline_utc']).timestamp()<deadline+120):
        raise ValueError('Fresh confirmed same-Pod budget guard must outlast this controller')
    actual=process(a.supervisor_pid)
    if not actual or actual['birth']!=a.supervisor_birth or actual['state']=='Z':
        raise ValueError('Existing exact audit supervisor must be live at dispatch')
    compiler=json.loads(a.compiler.read_text())
    if (compiler['packages_changed'] is not False or not compiler['actual_compiled_bf16_gpu_kernel_passed']
            or not compiler['actual_triton_driver_compile_passed']):
        raise ValueError('Pinned compiled BF16 CUDA runtime required')
    for name in ('header','assembler'):
        if digest(compiler[name])!=compiler[name+'_sha256']:raise ValueError('Changed CUDA compiler')
    base=arguments(a.root,a.payload,a.cache,a.preflight,a.supervisor,a.supervisor.parent)
    if a.native_audio49_prepared:
        base+=['--native-audio49-prepared',str(a.native_audio49_prepared),
               '--native-audio49-source-inventory',str(a.native_audio49_source_inventory)]
    a.out.mkdir(parents=True,exist_ok=False)
    lock=(a.preflight/'training-followup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ready=a.out/'combined-readiness';proof=a.out/'training'
    prefix=[str(a.python),'-u','-m',
            'portable.context.native_training_cached' if a.immutable_input_hashes else 'evals.context_training_proof']
    stages=[
        ('whole_workset',[str(a.python),'-u','-m','evals.native_control_snapshot','--inventory',str(a.inventory)]),
        ('combined_readiness',prefix+base+['--out',str(ready),'--native-audio49-preflight-only','--native-audio49-stream-gradient-readiness']),
        ('bounded8_training',prefix+base+['--out',str(proof),'--native-audio49-preflight',str(ready)]),
        ('independent_restore',prefix+base+['--out',str(proof),'--native-audio49-preflight',str(ready),'--reload']),
        ('strict_proof_verification',[str(a.python),'-u','-c',
            'from pathlib import Path; from evals.native_control_stage import verify_checkpoint; verify_checkpoint(Path('+repr(str(proof))+')); print("Complete retained154/native49/control40 proof and independent restore verified")']),
    ]
    state=dict(schema='dl-native-control40-training-followup-v1',pid=os.getpid(),
               controller_commit=git(Path(__file__).resolve().parents[2],'rev-parse','HEAD'),
               payload_commit=a.payload_commit,execution_source=str(a.payload),deadline_utc=a.deadline_utc,
               followed_supervisor_pid=a.supervisor_pid,followed_supervisor_birth=a.supervisor_birth,
               followed_supervisor_payload_commit=audit_commit,
               immutable_input_hash_reuse=a.immutable_input_hashes,
               prepared49_inputs=str(a.native_audio49_prepared) if a.native_audio49_prepared else None,
               phase='prepared',stages=[],planned_commands=dict(stages),new_stage_optimizer_limit=8,
               passed=False,model_adopted=False,behavior_qualified=False,extension_approved=False)
    def save():
        state['at_utc']=datetime.now(timezone.utc).isoformat()
        temp=a.out/'controller.partial';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(a.out/'controller.json')
    save()
    if a.prepare_only:return
    child=None
    def stop():
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    def interrupted(signum,_):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    env=training_environment(compiler,os.environ,a.payload_commit)
    try:
        state['phase']='waiting_for_exact_numerical_and_gradient40';save()
        while time.time()<deadline:
            s=json.loads((a.supervisor/'controller.json').read_text())
            if gradient_complete(s,a.supervisor_pid,a.supervisor_birth,audit_commit):break
            state['observed_audit_phase']=s['phase'];save();time.sleep(5)
        else:raise TimeoutError('Original full40 audits did not finish before training deadline')
        for name,command in stages:
            clean()
            if time.time()>=deadline:raise TimeoutError('Bounded training controller deadline reached')
            state['phase']=name;row=dict(name=name,command=command,started_utc=datetime.now(timezone.utc).isoformat())
            state['stages'].append(row);save()
            with (a.out/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=a.payload,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                row['pid']=child.pid;save();child.wait(timeout=max(1,deadline-time.time()))
            row.update(exit_code=child.returncode,ended_utc=datetime.now(timezone.utc).isoformat());save()
            if child.returncode:raise RuntimeError(name+' failed with '+str(child.returncode))
            if name=='combined_readiness':
                r=json.loads((ready/'control40-combined-readiness.json').read_text())
                if not r['passed'] or r['optimizer_updates']!=0 or len(r['steps'])!=8:
                    raise ValueError('Full independent readiness must pass before optimizer dispatch')
                state['combined_readiness_sha256']=digest(ready/'control40-combined-readiness.json');save()
        state.update(phase='complete_training_and_restore_only',passed=True,
                     proof=str(proof),checkpoint_sha256=digest(proof/'adapters.safetensors'),
                     training_state_sha256=digest(proof/'training-state.pt'),
                     report_sha256=digest(proof/'report.json'),reload_sha256=digest(proof/'reload-report.json'))
        save()
    except BaseException as error:
        stop();state.update(phase='failed',passed=False,error_type=type(error).__name__,error=str(error));save();raise


if __name__=='__main__':main()
