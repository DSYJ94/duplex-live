"""Isolated runtime bridge for a separately authorized recovery controller.

No CLI and no resource creation. The caller must supply a live provider/budget
guard and run this in a fresh process. The original trainer is never invoked;
only its pinned construction prefix and original input-hash checks are reused.
"""
import fcntl
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evals.native_reasoning_saved_stage_audit import SOURCE,FIT_COMMAND_SHA256
from evals.native_saved_stage_context import bind
from evals.native_saved_stage_recovery import canonical,finish,verified_inputs
from evals.native_saved_stage_proof import PRODUCER_FILES,digest

PURE_HELPERS={'evals.native_reasoning_saved_stage_audit','evals.native_saved_export_metadata',
    'evals.native_saved_stage_context','evals.native_saved_stage_recovery','evals.native_saved_stage_proof'}


def arguments(controller,out):
    command=controller['planned_commands']['bounded8_training']
    if (canonical_command(command)!=FIT_COMMAND_SHA256
            or command[1:4]!=['-u','-m','portable.context.native_training_cached']
            or command.count('--out')!=1 or '--reload' in command or '--resume-training' in command):
        raise ValueError('Only the exact original bounded8 recipe can be reconstructed')
    result=list(command[4:]);result[result.index('--out')+1]=str(out)
    return result


def canonical_command(command):
    import hashlib
    return hashlib.sha256(json.dumps(command,separators=(',',':')).encode()).hexdigest()


def guard_receipt(guard):
    value=guard();now=time.time()
    required=('execution_authorized','live_provider_deadline_verified',
        'live_external_cost_guard_verified','single_gpu_across_providers_verified')
    if (not isinstance(value,dict) or any(value.get(k) is not True for k in required)
            or any(type(value.get(k)) not in (int,float) for k in
                ('checked_unix','deadline_unix','remaining_budget_usd','planned_maximum_cost_usd'))
            or not 0<=now-value.get('checked_unix',0)<=30
            or not now<value.get('deadline_unix',0)<=now+12*3600
            or value.get('provider') not in ('gcp','runpod')
            or not 0<value.get('remaining_budget_usd',0)<=100
            or not 0<value.get('planned_maximum_cost_usd',0)<=value['remaining_budget_usd']
            or not isinstance(value.get('resource_id'),str) or not value['resource_id']
            or not isinstance(value.get('authorization_sha256'),str)
            or len(value['authorization_sha256'])!=64
            or any(c not in '0123456789abcdef' for c in value['authorization_sha256'])):
        raise ValueError('Fresh explicit authorization, provider deadline, cost guard and single GPU required')
    return {k:value[k] for k in (*required,'checked_unix','deadline_unix','provider',
        'remaining_budget_usd','planned_maximum_cost_usd','resource_id','authorization_sha256')}


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


def isolate(source):
    source=Path(source).resolve(strict=True)
    if git(source,'rev-parse','HEAD')!=SOURCE or git(source,'status','--porcelain'):
        raise ValueError('A separate clean original DA checkout is required')
    unexpected=[name for name in sys.modules if (name.startswith('evals.') and name not in PURE_HELPERS)
        or name=='contextspan' or name.startswith(('contextspan.','training.'))]
    if unexpected:raise ValueError('Fresh recovery process required; runtime modules already loaded')
    import evals
    evals.__path__=[str(source/'evals')]
    for name,path in (('portable',source/'portable'),('portable.context',source/'portable/context')):
        if name in sys.modules:sys.modules[name].__path__=[str(path)]
    sys.path.insert(0,str(source));os.chdir(source)
    os.environ['DL_TASK_COMMIT']=SOURCE
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    sys.dont_write_bytecode=True
    return source


def execute(source,snapshot,observation,out,producer_commit,guard,reference=None):
    """One new output per process; an external authorized controller owns timeouts."""
    initial_guard=guard_receipt(guard)
    producer=Path(__file__).resolve().parents[2]
    if git(producer,'rev-parse','HEAD')!=producer_commit or git(producer,'status','--porcelain'):
        raise ValueError('Exact clean separately committed recovery producer required')
    source=isolate(source)
    inputs=verified_inputs(snapshot,observation)
    out=Path(out).resolve()
    roots=[Path('/projects/u6si/duplex-live/runs').resolve(),source/'runs',inputs['root']]
    if out.exists() or any(out==p or p in out.parents or out in p.parents for p in roots):
        raise ValueError('New recovery output must be separate from all immutable input roots')
    lock=(out.parent/'native-control40-gpu-owner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('Another GPU process is active')
    args=arguments(inputs['controller'],out)
    from portable.context.native_training_cached import verify_receipt,CODE,sha,PROFILE
    ready=Path(args[args.index('--native-audio49-preflight')+1])
    verify_receipt(ready/'immutable-input-readiness.json','readiness',SOURCE)
    module=importlib.import_module('evals.context_actual_train')
    if Path(module.__file__).resolve()!=source/'evals/context_actual_train.py':
        raise ValueError('Original model construction resolved outside pinned source')
    captured=[];prefix_receipt=None
    def capture(context):
        if captured:raise ValueError('Only one recovery context may be executed')
        guard_receipt(guard)
        result=finish(context,inputs,prefix_receipt,producer_commit,reference=reference)
        captured.append(result)
        return result
    function,prefix_receipt=bind(Path(module.__file__).read_bytes(),vars(module),capture)
    # Explicit process-local replacement of construction only. The derived
    # function contains no optimizer/reload tail and records its AST identity.
    module.run=function
    from evals.immutable_file_hashes import verified_reads
    sys.argv=['saved-stage-zero-update',*args]
    passed=False;reads=None
    try:
        with verified_reads(roots[:2]) as reads:
            importlib.import_module('evals.context_training_proof').main()
        if len(captured)!=1:raise ValueError('Pinned construction did not reach recovery capture')
        guard_receipt(guard);passed=True
    finally:
        if reads is not None and out.is_dir():
            receipt=dict(schema='dl-native-saved-stage-runtime-inputs-v1',passed=passed,
                source_commit=SOURCE,producer_commit=producer_commit,cache=reads.receipt(),
                profile=PROFILE,code_sha256={name:sha(source/name) for name in CODE},
                context=prefix_receipt,initial_guard=initial_guard,
                result_sha256=canonical(captured[0]) if passed else None,
                producer_implementation_sha256={name:digest(producer/name) for name in PRODUCER_FILES},
                optimizer_updates_during_recovery=0,original_fit_relabelled=False,
                model_adopted=False)
            with (out/'recovery-runtime-inputs.json').open('x') as stream:
                stream.write(json.dumps(receipt,indent=2)+'\n')
    return captured[0]
