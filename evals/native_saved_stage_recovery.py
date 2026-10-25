"""Separate zero-update finalization of the pinned, failed V18 export.

Library only: no execution entry point, budget permission or qualification.
The runtime must construct its context from the isolated original DA checkout.
Original failed outputs are read-only; results use a distinct recovery schema.
"""
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path

from evals.native_reasoning_saved_stage_audit import (
    SOURCE,PILOT,READY,PINNED,audit,terminal_timeout,optimizer_rows,confined)
from evals.native_saved_export_metadata import checkpoint_prefix,inspect_export,sha

INVENTORY='318a30c25a47719ba745475a27c7b240f973ce9010a2dfc7873fa4dbb5535135'
OBSERVATION='04d8569f4d343271c7b234e87fc40806cb00b80caf0ef4f97ac33f03185bf136'
CONTROLLER='0605e35166606ae55a5beca5ce56757f69acb68aceeffb989735a4239bda3a32'
METADATA='caae1dfc5ba1c39761189087283aa31ff314532e1db417c6b32407df459a9d8b'
SCHEMA='dl-native-saved-stage-zero-update-finalization-v1'


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def process_identity():
    """Container PID plus Linux boot/start identity; a PID alone is insufficient."""
    from portable.context.native_preflight_takeover import process
    pid=os.getpid();handle=process(pid)
    if handle is None:raise ValueError('Live recovery process identity unavailable')
    return dict(pid=pid,birth=handle['birth'],host=os.uname().nodename,
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def partial_fingerprints(text,ready):
    """Bind the complete surviving prefix, never use it as a passing subset."""
    fields={'native_stream_restore_fingerprint':'native49',
        'control40_restore_fingerprint':'control40','reasoning32_restore_fingerprint':'reasoning32'}
    result={name:{} for name in fields.values()}
    for line in text.splitlines():
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        if not isinstance(row,dict):continue
        markers=[key for key in fields if key in row]
        if not markers:continue
        if len(markers)!=1:raise ValueError('Ambiguous preserved fingerprint')
        marker=markers[0];name=fields[marker];identifier=row[marker]
        if identifier in result[name]:raise ValueError('Duplicate preserved fingerprint')
        result[name][identifier]=(row['sha256'] if name=='native49'
            else {key:value for key,value in row.items() if key!=marker})
    native=[r['id'] for r in ready['native49_policy']['teachers']]
    control=[r['id'] for r in ready['control40_policy']['teachers']]
    if (len(native)!=49 or len(set(native))!=49 or len(control)!=40 or len(set(control))!=40
            or list(result['native49'])!=native or list(result['control40'])!=control[:31]
            or result['reasoning32']):
        raise ValueError('Exact original 49 native and 31 control fingerprint prefix required')
    return result


def validate_complete_fingerprints(actual,metadata,partials):
    recipe=metadata['recipe'];audio=recipe['assistant_audio_auxiliary']
    ids=dict(original28=None,audio49=audio['teacher_ids'],
        native49=[r['id'] for r in audio['native_stream_auxiliary']['teachers']],
        control40=[r['id'] for r in recipe['control40']['teachers']],
        reasoning32=[r['id'] for r in recipe['reasoning32']['teachers']])
    if set(actual)!=set(ids) or not isinstance(actual['original28'],dict) or len(actual['original28'])!=28:
        raise ValueError('All original28 and supplemental fingerprint groups required')
    for name,expected in ids.items():
        if expected is not None and (len(expected)!=len(set(expected)) or set(actual[name])!=set(expected)):
            raise ValueError('Incomplete or substituted '+name+' fingerprints')
    def digest(value):
        return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    for name in ('original28','audio49','native49'):
        if any(not digest(value) for value in actual[name].values()):
            raise ValueError('Invalid '+name+' fingerprint')
    for name in ('control40','reasoning32'):
        if set(recipe[name]['target_counts'])!=set(actual[name]):
            raise ValueError('Complete source target counts required')
        for identifier,value in actual[name].items():
            if (set(value)!={'native_text_calls_sha256','assistant_audio_logits_sha256',
                    'assistant_audio_targets','text_targets'}
                    or not digest(value['native_text_calls_sha256'])
                    or not digest(value['assistant_audio_logits_sha256'])
                    or type(value['text_targets']) is not int or value['text_targets']<=0
                    or not isinstance(value['assistant_audio_targets'],list)
                    or len(value['assistant_audio_targets'])!=8
                    or any(type(n) is not int or n<0 for n in value['assistant_audio_targets'])
                    or sum(value['assistant_audio_targets'])<=0):
                raise ValueError('Invalid native text/acoustic fingerprint')
            target=recipe[name]['target_counts'][identifier]
            if value['text_targets']!=target['text'] or value['assistant_audio_targets']!=target['audio']:
                raise ValueError('Pinned native text/acoustic target counts changed')
    for name,rows in partials.items():
        for identifier,expected in rows.items():
            if actual[name].get(identifier)!=expected:
                raise ValueError('Original preserved '+name+' fingerprint changed')
    return True


def verified_inputs(snapshot,observation):
    root=Path(snapshot).resolve(strict=True);observation=Path(observation)
    if sha(root/'inventory.json')!=INVENTORY or sha(observation)!=OBSERVATION:
        raise ValueError('Exact closed snapshot and original-host observation required')
    checked=audit(root)
    state=json.loads((root/PILOT/'controller.json').read_text())
    observed=json.loads(observation.read_text())
    if (sha(root/PILOT/'controller.json')!=CONTROLLER or observed['controller_sha256']!=CONTROLLER
            or observed['controller']!=state or observed['fit_report_exists'] is not False
            or observed['gpu_processes'].strip() or observed['paired_output_exists']
            or observed['speech_output_exists'] or (root/PILOT/'training/report.json').exists()):
        raise ValueError('Original failed, terminal and unqualified outcome must remain intact')
    terminal_timeout(state,observed['controller_handle'],observed['fit_handle'])
    training=root/PILOT/'training';prefix=checkpoint_prefix(training/'training-state.pt',
        PINNED[PILOT+'/training/training-state.pt'])
    # Pin every packaged JSON artifact consumed by reconstruction comparisons,
    # rather than trusting a snapshot directory simply because its index is pinned.
    entries={r['path']:r for r in json.loads((root/'inventory.json').read_text())['files']}
    packaged={name for name in entries if Path(name).parent==Path(PILOT)/'training'
        and Path(name).suffix=='.json'}
    if packaged!={path.relative_to(root).as_posix() for path in training.glob('*.json')}:
        raise ValueError('Original packaged recovery input set changed')
    for name in packaged:
        entry=entries[name];actual=confined(root,name)
        if actual.stat().st_size!=entry['bytes'] or sha(actual)!=entry['sha256']:
            raise ValueError('Packaged recovery input changed: '+name)
    info=inspect_export(training)
    if info['metadata_sha256']!=METADATA:raise ValueError('Original checkpoint metadata changed')
    text=(root/PILOT/'bounded8_training.log').read_text()
    partials=partial_fingerprints(text,json.loads((root/READY).read_text()))
    return dict(root=root,observation=observation.resolve(strict=True),training=training,prefix=prefix,inspection=info,
        partials=partials,steps=optimizer_rows(text),audit=checked,controller=state)


def validate_context(context,inputs):
    """Compare independently reconstructed physical inputs with the pinned state."""
    args=context['args'];out=Path(args.out).resolve(strict=True);root=inputs['root']
    if out==root or root in out.parents or out in root.parents or args.reload or args.resume_training:
        raise ValueError('A separate new output and no training/reload mode required')
    expected=inputs['prefix']['metadata'];metadata=context['state_metadata'](context['frozen'])
    if metadata!=expected or canonical(metadata)!=METADATA:
        raise ValueError('Physical source/data/voice/recipe reconstruction differs from saved state')
    if list(context['learned'])!=inputs['prefix']['parameter_names']:
        raise ValueError('Physical adapter parameter order changed')
    counts={'batches':28,'retention':4,'native_teacher_batches':25,'additional_batches':24,
        'warm_references':10,'clock_references':14,'audio49_batches':49,
        'stream_bound':49,'control_bound':40,'reasoning_bound':32}
    if any(len(context[key])!=count for key,count in counts.items()):
        raise ValueError('Complete original and supplemental native input sets required')
    for name in ('data.json','voice.pt','approved-temporal-audit.json',
            'warm-ordinary-reference-audit.json','warm-ordinary-clock-reference-audit.json'):
        if sha(out/name)!=sha(inputs['training']/name):
            raise ValueError('Reconstructed source or anchor artifact changed: '+name)
    for path in inputs['training'].glob('approved-*.json'):
        if sha(out/path.name)!=sha(path):
            raise ValueError('Reconstructed complete approval changed: '+path.name)
    return metadata


def finish(context,inputs,context_receipt,producer_commit,reference=None):
    """GPU callback for a future authorized, isolated runner; never steps Adam."""
    inputs=verified_inputs(inputs['root'],inputs['observation'])
    # No Torch import occurs during input auditing or offline tests.
    import torch
    import numpy as np
    import random
    from safetensors.torch import load_file
    from evals.context_training_state import restore
    from evals.context_native_parent_adam import port
    from evals.moshi_training_smoke import tensor_hashes
    from training.control.train_context import assert_equal,cpu_tree
    from evals.context_native_audio49_training import fingerprints as acoustic
    from evals.context_native_stream_training import fingerprints as streaming
    from evals.native_control_training import fingerprints as control
    from evals.native_reasoning_training import fingerprints as reasoning
    metadata=validate_context(context,inputs)
    lm=context['lm'];learned=context['learned'];out=Path(context['args'].out)
    if (context_receipt['original_source_sha256']!='2e504fac5d3171680e3277d1e436f7db51d9403b45a1e77f09d4c2c7ccc893c5'
            or context_receipt['derived_ast_sha256']!='7907f6de7ddbdaa1a22111829a67cfb1a4a2e5df629482f1693c4676549d15f0'
            or context_receipt['optimizer_branch_included'] is not False
            or len(producer_commit)!=40 or any(c not in '0123456789abcdef' for c in producer_commit)):
        raise ValueError('Exact separately recorded recovery producer and construction prefix required')
    if any(p.grad is not None for p in lm.parameters()):raise ValueError('Recovery cannot consume gradients')
    # Reconstruct the actual native520 starting model and Adam import before
    # restoring the saved final state. These are newly measured initial metrics,
    # explicitly distinguished from the missing original in-memory report.
    parent=load_file(context['args'].warm_start_proof/'adapters.safetensors',device='cuda')
    if set(parent)!=set(learned):raise ValueError('Incomplete native520 source parameters')
    with torch.no_grad():
        for name,p in learned.items():
            if parent[name].shape!=p.shape or parent[name].dtype!=p.dtype or not torch.isfinite(parent[name]).all():
                raise ValueError('Changed native520 source parameter')
            p.copy_(parent[name])
    optimizer=torch.optim.AdamW(list(learned.values()),lr=2e-7,weight_decay=0.)
    port(context['args'].warm_start_proof,learned,optimizer,2e-7,out)
    if sha(out/'parent-adam-import.json')!=sha(inputs['training']/'parent-adam-import.json'):
        raise ValueError('Reconstructed exact native520 Adam import changed')
    initial_metrics=context['metrics']()
    saved_path=inputs['training']/'training-state.pt'
    snapshot=restore(saved_path,learned,optimizer,metadata)
    def exact_state():
        if snapshot['completed_steps']!=8 or list(learned)!=snapshot['parameter_names']:
            raise ValueError('Exact eight-update named state required')
        assert_equal(snapshot['adapters'],load_file(inputs['training']/'adapters.safetensors',device='cpu'))
        assert_equal(snapshot['adapters'],cpu_tree(learned))
        assert_equal(snapshot['optimizer'],cpu_tree(optimizer.state_dict()))
        assert_equal(snapshot['torch_cpu_rng'],torch.get_rng_state())
        assert_equal(snapshot['torch_cuda_rng'],torch.cuda.get_rng_state_all())
        assert_equal(snapshot['python_rng'],random.getstate())
        n=np.random.get_state()
        assert_equal(snapshot['numpy_rng'],dict(name=n[0],keys=n[1].tolist(),position=int(n[2]),
            has_gauss=int(n[3]),cached_gauss=float(n[4])))
        if len(optimizer.state)!=26 or any(set(row)!={'step','exp_avg','exp_avg_sq'}
                or float(row['step'])!=528 or bool((row['exp_avg_sq']<0).any()) for row in optimizer.state.values()):
            raise ValueError('All26 named Adam states and clocks528 required')
        if any(p.grad is not None for p in lm.parameters()):raise ValueError('Recovery generated gradients')
    exact_state()
    before=tensor_hashes(learned.items())
    with torch.no_grad():
        final_metrics=context['metrics']()
        fingerprints=dict(original28=context['digests'](),
            audio49=acoustic(lm,context['audio49_batches']),
            native49=streaming(lm,context['stream_bound']),
            control40=control(lm,context['control_bound']),
            reasoning32=reasoning(lm,context['reasoning_bound']))
    if before!=tensor_hashes(learned.items()) or context['frozen']!=tensor_hashes(
            (n,p) for n,p in lm.named_parameters() if not p.requires_grad):
        raise ValueError('Recovery checks changed adapter or frozen base weights')
    assert_equal(snapshot['optimizer'],cpu_tree(optimizer.state_dict()))
    validate_complete_fingerprints(fingerprints,metadata,inputs['partials'])
    if set(fingerprints['original28'])!=set(context['baseline']):
        raise ValueError('Original28 final digest identities changed')
    if fingerprints['original28']==context['baseline']:
        raise ValueError('Saved trained state changed no original valid logits')
    if len(initial_metrics)!=28 or len(final_metrics)!=28:
        raise ValueError('All28 initial and final metrics required')
    expected_scenarios=[row['scenario'] for row in context['audits']]
    if any([r['scenario'] for r in rows]!=expected_scenarios for rows in (initial_metrics,final_metrics)):
        raise ValueError('Original28 metric scenario order changed')
    for row in initial_metrics+final_metrics:
        if any(isinstance(v,float) and not math.isfinite(v) for v in row.values()):
            raise ValueError('Nonfinite reconstructed loss metrics')
    # Evaluation may consume RNG; finish by restoring the exact original state
    # again, then compare every optimizer and RNG stream independently.
    restore(saved_path,learned,optimizer,metadata)
    exact_state()
    identity=process_identity()
    evidence=dict(metadata_sha256=METADATA,initial_stage_metrics_recomputed=initial_metrics,
        final_metrics_recomputed=final_metrics,all_fingerprints=fingerprints,
        state_metadata=metadata,original_steps=inputs['steps'],adapter_modules=context['modules'])
    phase='independent_restore' if reference is not None else 'finalization'
    if reference is not None:
        if (reference['schema']!=SCHEMA or reference['phase']!='finalization'
                or reference['producer_commit']!=producer_commit or reference['pid']==os.getpid()
                or reference['evidence']!=evidence or reference['context']!=context_receipt
                or reference['optimizer_updates_during_recovery']!=0 or not reference['passed']
                or any(reference['process_identity'][k]!=identity[k] for k in ('host','boot_id'))
                or reference['process_identity']==identity):
            raise ValueError('Fresh-process complete state, metrics and fingerprints must match')
    # Re-audit all pinned inputs after computation to detect a concurrent edit.
    verified_inputs(inputs['root'],inputs['observation'])
    result=dict(schema=SCHEMA,phase=phase,pid=os.getpid(),process_identity=identity,
        reference_finalization_sha256=None if reference is None else canonical(reference),
        at_utc=datetime.now(timezone.utc).isoformat(),
        producer_commit=producer_commit,model_source_commit=SOURCE,context=context_receipt,evidence=evidence,
        original_inventory_sha256=INVENTORY,original_terminal_observation_sha256=OBSERVATION,
        training_state_sha256=PINNED[PILOT+'/training/training-state.pt'],
        adapter_sha256=PINNED[PILOT+'/training/adapters.safetensors'],
        original_fit_actual_exit=-15,original_fit_passed=False,original_stage_updates=8,
        optimizer_updates_during_recovery=0,optimizer_and_rng_exact=True,adam_parameter_steps=[528],
        original_partial_fingerprints_exact=True,full_independent_restore_verified=reference is not None,
        quality_qualified=False,native_behavior_evaluated=False,model_adopted=False,passed=True)
    with (out/('independent-restore.json' if reference is not None else 'saved-stage-finalization.json')).open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    return result
