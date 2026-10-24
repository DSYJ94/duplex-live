"""Strict distinct proof for recovered saved8; never relabel the failed fit."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

from evals.native_saved_stage_context import construction_tree
from evals.native_saved_stage_recovery import (
    SCHEMA as RESULT_SCHEMA,SOURCE,INVENTORY,OBSERVATION,METADATA,PILOT,PINNED,
    canonical,validate_complete_fingerprints,verified_inputs)

SCHEMA='dl-native-reasoning32-recovered-saved8-proof-v1'
RELOAD_SCHEMA='dl-native-reasoning32-recovered-saved8-reload-v1'
JOB_SCHEMA='dl-native-saved-stage-terminal-job-v1'
PRODUCER_FILES=('evals/native_saved_stage_context.py','evals/native_saved_stage_recovery.py',
    'evals/native_saved_stage_proof.py','portable/context/native_saved_stage_runtime.py',
    'evals/native_saved_export_metadata.py','evals/native_reasoning_saved_stage_audit.py')
INPUT_CODE=('evals/immutable_file_hashes.py','evals/moshi_model_download.py',
    'portable/context/native_training_cached.py')
PROFILE='full-first-read-plus-file-identity-every-reuse-v1'
ARTIFACTS=('adapters.safetensors','training-state.pt','data.json','voice.pt')
RECORDS=('finalization-record.json','finalization-runtime.json','finalization-job.json',
    'independent-record.json','independent-runtime.json','independent-job.json')


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def committed(commit,names,physical=False):
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Exact committed implementation required')
    root=Path(__file__).resolve().parents[1];result={}
    for name in names:
        result[name]=hashlib.sha256(subprocess.check_output(['git','show',commit+':'+name],cwd=root)).hexdigest()
        if physical and digest(root/name)!=result[name]:raise ValueError('Recovery producer code changed: '+name)
    return result


def identity(value):
    return (isinstance(value,dict) and set(value)=={'pid','birth','host','boot_id'}
        and type(value['pid']) is int and value['pid']>0
        and type(value['birth']) is int and value['birth']>0
        and all(isinstance(value[k],str) and value[k] for k in ('host','boot_id')))


def validate_metrics(rows,scenarios):
    if len(rows)!=28 or [r['scenario'] for r in rows]!=scenarios:
        raise ValueError('All28 ordered original-history metrics required')
    for row in rows:
        if any(isinstance(v,float) and not math.isfinite(v) for v in row.values()):
            raise ValueError('Nonfinite measured metric')
        keys=('nll','top1','first_supervised_token_probability')
        if 'first_content_target' in row:keys+=('optimization_nll',)
        for key in keys:
            v=row[key]
            if type(v) not in (float,int) or not math.isfinite(v) or v<0:
                raise ValueError('Finite complete loss and probability metrics required')
        if row['top1']>1 or row['first_supervised_token_probability']>1:
            raise ValueError('Invalid measured probability')


def validate_records(records,inputs,producer_files,input_code,context):
    """Pure semantic gate; callers also bind every record's physical file hash."""
    first,second=records['finalization-record.json'],records['independent-record.json']
    producer=first['producer_commit'];meta=inputs['prefix']['metadata'];evidence=first['evidence']
    if (first['phase']!='finalization' or second['phase']!='independent_restore'
            or first['reference_finalization_sha256'] is not None
            or second['reference_finalization_sha256']!=canonical(first)
            or first['full_independent_restore_verified'] is not False
            or second['full_independent_restore_verified'] is not True
            or second['evidence']!=evidence or first['context']!=context or second['context']!=context):
        raise ValueError('Complete finalization and distinct full independent restore required')
    identities=[r['process_identity'] for r in (first,second)]
    if (any(not identity(v) for v in identities) or identities[0]['pid']==identities[1]['pid']
            or any(identities[0][k]!=identities[1][k] for k in ('host','boot_id'))):
        raise ValueError('Independent processes on the same recorded runtime host required')
    required_true=('passed','optimizer_and_rng_exact','original_partial_fingerprints_exact')
    required_false=('original_fit_passed','quality_qualified','native_behavior_evaluated','model_adopted')
    for prefix,result in (('finalization',first),('independent',second)):
        if (result['schema']!=RESULT_SCHEMA or result['model_source_commit']!=SOURCE
                or result['producer_commit']!=producer or result['context']!=context
                or result['original_inventory_sha256']!=INVENTORY
                or result['original_terminal_observation_sha256']!=OBSERVATION
                or result['training_state_sha256']!=PINNED[PILOT+'/training/training-state.pt']
                or result['adapter_sha256']!=PINNED[PILOT+'/training/adapters.safetensors']
                or type(result['original_fit_actual_exit']) is not int or result['original_fit_actual_exit']!=-15
                or type(result['original_stage_updates']) is not int or result['original_stage_updates']!=8
                or type(result['optimizer_updates_during_recovery']) is not int or result['optimizer_updates_during_recovery']!=0
                or result['adam_parameter_steps']!=[528] or result['pid']!=result['process_identity']['pid']
                or any(result[k] is not True for k in required_true)
                or any(result[k] is not False for k in required_false)):
            raise ValueError('Pinned original failure, zero new updates and complete state comparison required')
        runtime=records[prefix+'-runtime.json'];job=records[prefix+'-job.json']
        cache=runtime['cache'];guard=runtime['initial_guard']
        stamp=datetime.fromisoformat(result['at_utc']).timestamp()
        if (runtime['schema']!='dl-native-saved-stage-runtime-inputs-v1' or runtime['passed'] is not True
                or runtime['source_commit']!=SOURCE or runtime['producer_commit']!=producer
                or runtime['context']!=context or runtime['result_sha256']!=canonical(result)
                or runtime['producer_implementation_sha256']!=producer_files
                or runtime['code_sha256']!=input_code or runtime['profile']!=PROFILE
                or runtime['optimizer_updates_during_recovery']!=0 or runtime['original_fit_relabelled'] is not False
                or runtime['model_adopted'] is not False or cache['mode']!=PROFILE
                or cache['unique_hashed_files']<1 or cache['all_cached_identities_rechecked'] is not True
                or cache['validation_results_or_model_outputs_cached'] is not False):
            raise ValueError('Successful pinned runtime and complete immutable input checks required')
        flags=('execution_authorized','live_provider_deadline_verified',
            'live_external_cost_guard_verified','single_gpu_across_providers_verified')
        if (any(guard[k] is not True for k in flags) or guard['provider'] not in ('gcp','runpod')
                or any(type(guard[k]) not in (int,float) for k in
                    ('checked_unix','deadline_unix','remaining_budget_usd','planned_maximum_cost_usd'))
                or not 0<guard['remaining_budget_usd']<=100
                or not 0<guard['planned_maximum_cost_usd']<=guard['remaining_budget_usd']
                or not guard['checked_unix']<=stamp<guard['deadline_unix']<=guard['checked_unix']+12*3600
                or type(guard['resource_id']) is not str or not guard['resource_id']
                or type(guard['authorization_sha256']) is not str or len(guard['authorization_sha256'])!=64
                or any(c not in '0123456789abcdef' for c in guard['authorization_sha256'])):
            raise ValueError('Bounded authorized single-GPU runtime required')
        if (job['schema']!=JOB_SCHEMA or job['phase']!='complete' or job['kind']!=result['phase']
                or job['actual_exit']!=0 or type(job['actual_exit']) is not int
                or job['observed_terminal'] is not True or job['child_identity']!=result['process_identity']
                or job['producer_commit']!=producer or job['model_source_commit']!=SOURCE
                or job['result_canonical_sha256']!=canonical(result)
                or job['runtime_canonical_sha256']!=canonical(runtime)
                or not job['started_unix']<=stamp<=job['ended_unix']<guard['deadline_unix']):
            raise ValueError('Actual successful terminal child and complete result/runtime binding required')
    guards=[records[p+'-runtime.json']['initial_guard'] for p in ('finalization','independent')]
    if any(guards[0][k]!=guards[1][k] for k in ('provider','resource_id','authorization_sha256','deadline_unix')):
        raise ValueError('The same bounded single-GPU resource and authorization required')
    if (evidence['state_metadata']!=meta or evidence['metadata_sha256']!=METADATA
            or evidence['original_steps']!=inputs['steps']):
        raise ValueError('Original full metadata and all eight update receipts required')
    validate_complete_fingerprints(evidence['all_fingerprints'],meta,inputs['partials'])
    data=read(inputs['training']/'data.json');scenarios=[r['scenario'] for r in data['components']]
    if set(evidence['all_fingerprints']['original28'])!={str(i)+':'+s for i,s in enumerate(scenarios)}:
        raise ValueError('Exact original28 fingerprint identities required')
    for name in ('initial_stage_metrics_recomputed','final_metrics_recomputed'):
        validate_metrics(evidence[name],scenarios)
    expected_modules={name.split('.lora_')[0] for name in inputs['prefix']['parameter_names']}
    modules=evidence['adapter_modules']
    if len(modules)!=13 or len(set(modules))!=13 or set(modules)!=expected_modules:
        raise ValueError('All13 physical adapter modules required')
    return first,second


def record_terminal(child,folder,producer_commit,started_unix,out):
    """Called by an authorized controller only after its real child exits zero."""
    from portable.context.native_preflight_takeover import process
    folder=Path(folder)
    candidates=[p for p in (folder/'saved-stage-finalization.json',folder/'independent-restore.json') if p.is_file()]
    if len(candidates)!=1 or child.poll()!=0:raise ValueError('One actually successful terminal recovery child required')
    result=read(candidates[0]);runtime=read(folder/'recovery-runtime-inputs.json')
    live=process(child.pid);handle=result['process_identity']
    if (child.pid!=handle['pid'] or not identity(handle) or result['producer_commit']!=producer_commit
            or result['passed'] is not True or runtime['passed'] is not True
            or runtime['result_sha256']!=canonical(result)
            or live is not None and (live['birth']!=handle['birth'] or live['state']!='Z')):
        raise ValueError('Actual child identity, terminal state and complete runtime result required')
    value=dict(schema=JOB_SCHEMA,phase='complete',kind=result['phase'],actual_exit=child.returncode,
        observed_terminal=True,child_identity=handle,producer_commit=producer_commit,model_source_commit=SOURCE,
        result_canonical_sha256=canonical(result),runtime_canonical_sha256=canonical(runtime),
        started_unix=started_unix,ended_unix=time.time())
    with Path(out).open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    return value


def description(proof,inputs,records,hashes,parent):
    first=records['finalization-record.json'];second=records['independent-record.json'];meta=inputs['prefix']['metadata']
    recipe=meta['recipe']
    report=dict(schema=SCHEMA,passed=True,repo_commit=SOURCE,source_commit=meta['source_commit'],
        recovery_producer_commit=first['producer_commit'],original_snapshot=str(inputs['root']),
        original_observation=str(inputs['observation']),recovery_record_sha256=hashes,
        original_fit_passed=False,original_fit_actual_exit=-15,original_stage_updates=8,
        updates_this_run=0,optimizer_steps=8,optimizer_update_scope='Eight historical V18 updates; recovery executes zero updates',
        adam_parameter_steps_after=[528],checkpoint_sha256=PINNED[PILOT+'/training/adapters.safetensors'],
        training_state_sha256=PINNED[PILOT+'/training/training-state.pt'],data_sha256=meta['data_sha256'],
        voice_sha256=meta['voice_sha256'],voice_reference_sha256=parent['voice_reference_sha256'],
        assets_manifest_sha256=meta['assets_manifest_sha256'],native_protocol=meta['native_protocol'],
        frozen_hashes=meta['frozen_hashes'],frozen_weights_unchanged=True,base_audio_parameters_frozen=True,
        warm_start=meta['warm_start'],warm_start_optimizer_reset=False,
        rank=8,scaling=2.,last_layers=4,trainable_parameters=1_894_400,
        adapter_modules=first['evidence']['adapter_modules'],additional_native_v2=recipe['additional_native_v2'],
        quality_qualified=False,model_adopted=False)
    reload=dict(schema=RELOAD_SCHEMA,passed=True,fresh_process=True,pid=second['pid'],
        finalization_pid=first['pid'],checkpoint_sha256=report['checkpoint_sha256'],
        training_state_sha256=report['training_state_sha256'],data_sha256=report['data_sha256'],
        optimizer_and_rng_independently_compared_exact=True,parent_adam_parameter_steps_restored=[528],
        finalization_sha256=hashes['finalization-record.json'],independent_sha256=hashes['independent-record.json'],
        all_original_and_supplemental_logits_exact=True,original_fit_relabelled=False)
    return report,reload


def verify_checkpoint(proof):
    from evals.context_rollout_conditioning import verify_checkpoint as verify_parent
    proof=Path(proof).resolve(strict=True);report=read(proof/'report.json')
    if report['schema']!=SCHEMA:raise ValueError('Explicit saved-stage recovery proof required')
    inputs=verified_inputs(report['original_snapshot'],report['original_observation'])
    records={name:read(proof/name) for name in RECORDS}
    hashes={name:digest(proof/name) for name in RECORDS}
    producer=records['finalization-record.json']['producer_commit']
    producer_files=committed(producer,PRODUCER_FILES,physical=True)
    source=Path(__file__).resolve().parents[1]
    original=subprocess.check_output(['git','show',SOURCE+':evals/context_actual_train.py'],cwd=source)
    _,context=construction_tree(original)
    validate_records(records,inputs,producer_files,committed(SOURCE,INPUT_CODE),context)
    meta=inputs['prefix']['metadata'];parent_identity,parent=verify_parent(meta['warm_start']['proof'])
    if (parent_identity!=meta['warm_start'] or parent['optimizer_steps']!=520
            or any(parent[k]!=meta[k] for k in ('data_sha256','voice_sha256','assets_manifest_sha256',
                'frozen_hashes','native_protocol','retention_examples'))):
        raise ValueError('Exact original native520 data, voice, protocol, base and parent required')
    expected,reload=description(proof,inputs,records,hashes,parent)
    if report!=expected or read(proof/'reload-report.json')!=reload:
        raise ValueError('Recovered proof summary differs from complete pinned evidence')
    for name in ARTIFACTS:
        if digest(proof/name)!=PINNED[PILOT+'/training/'+name]:raise ValueError('Recovered artifact changed: '+name)
    identity=dict(proof=str(proof),report_sha256=digest(proof/'report.json'),
        reload_report_sha256=digest(proof/'reload-report.json'),checkpoint_sha256=report['checkpoint_sha256'],
        training_state_sha256=report['training_state_sha256'],native_protocol=report['native_protocol'],
        optimizer_steps=8,recovered_saved_stage=True,recovery_optimizer_updates=0)
    return identity,report


def package(finalization,independent,final_job,independent_job,snapshot,observation,out):
    """Create a new derived proof only from two complete actual exit-zero jobs."""
    from evals.context_rollout_conditioning import verify_checkpoint as verify_parent
    inputs=verified_inputs(snapshot,observation);out=Path(out).resolve()
    if out.exists() or inputs['root'] in out.parents or out in inputs['root'].parents:
        raise ValueError('New separate derived proof directory required')
    origins=dict(zip(RECORDS,(Path(finalization)/'saved-stage-finalization.json',
        Path(finalization)/'recovery-runtime-inputs.json',Path(final_job),
        Path(independent)/'independent-restore.json',Path(independent)/'recovery-runtime-inputs.json',Path(independent_job))))
    records={name:read(path) for name,path in origins.items()};producer=records[RECORDS[0]]['producer_commit']
    source=Path(__file__).resolve().parents[1]
    original=subprocess.check_output(['git','show',SOURCE+':evals/context_actual_train.py'],cwd=source)
    _,context=construction_tree(original)
    validate_records(records,inputs,committed(producer,PRODUCER_FILES,physical=True),committed(SOURCE,INPUT_CODE),context)
    meta=inputs['prefix']['metadata'];parent_identity,parent=verify_parent(meta['warm_start']['proof'])
    if parent_identity!=meta['warm_start']:raise ValueError('Original native520 parent changed')
    out.mkdir()
    for name,path in origins.items():shutil.copyfile(path,out/name)
    for name in ARTIFACTS:shutil.copyfile(inputs['training']/name,out/name)
    report,reload=description(out,inputs,records,{name:digest(out/name) for name in RECORDS},parent)
    for name,value in (('report.json',report),('reload-report.json',reload)):
        with (out/name).open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    return verify_checkpoint(out)
