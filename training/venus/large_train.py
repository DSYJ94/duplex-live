"""Complete-epoch substantial-adapter/control training after native admission.

No saved updates are rerun for restoration proofs. Each arm has a new declared
phase clock, with inherited Adam/OPAL programs and immutable parent history.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import time

import torch

from .large_inputs import Corpus, PARENT_SHA, MANIFEST_SHA, sha
from .large_native import guarded, restore_expanded, write, tensor_sha
from .native_cache import load_native
from .native_optimizer import assert_optimizer_equal
from .state import save, restore_rng
from . import train as core


def admission(step_seconds, dev_seconds, available_seconds, checkpoint_bytes,*,workload=None,control=None):
    """Joint whole-epoch arms; measured timing and backup size, never loss."""
    if any(not math.isfinite(v) or v<=0 for v in (step_seconds,dev_seconds,available_seconds,checkpoint_bytes)):
        raise ValueError('Positive actual native timing/size required')
    epoch_cost=None
    if workload is not None or control is not None:
        from .workload import epoch_seconds
        if (workload is None or control is None or control['optimizer_updates']!=0 or control['effective_batch']!=4
                or any(not math.isfinite(control[k]) or control[k]<=0 for k in ('seconds','full_dev_seconds'))):
            raise ValueError('Actual same-corpus control runtime and whole geometry required')
        epoch_cost=epoch_seconds(step_seconds,control['seconds'],workload)
    for epochs in range(8, 0, -1):
        seconds = (epochs*(epoch_cost['main']+epoch_cost['control'])*1.35+(epochs+1)*(dev_seconds+control['full_dev_seconds'])*1.25+4200 if epoch_cost else 2*epochs*60*step_seconds*1.35+2*(epochs+1)*dev_seconds*1.25+4200)
        # Preserve initialization, every whole-epoch full-state checkpoint and
        # one interruption state; reserve5GB for control/audio/receipts. Stay
        # within the approved100GB transfer ceiling, not just local disk size.
        size = (epochs+2)*checkpoint_bytes + 5_000_000_000
        if seconds < available_seconds and size < 100_000_000_000:
            return dict(epochs=epochs, updates_each_arm=epochs*60,
                        maximum_all_branch_updates=984, calibration_branches_used=0,
                        complete_train_per_epoch=240, full_dev_per_review=64,
                        joint_required_seconds=seconds, joint_backup_bytes_upper=size,
                        measured_step_seconds=step_seconds, measured_full_dev_seconds=dev_seconds,
                        actual_initial_checkpoint_bytes=checkpoint_bytes,epoch_runtime_basis=epoch_cost,
                        selection_uses_losses_or_heldout=False)
    raise TimeoutError('No complete useful main/control epoch pair fits time and backup allowance')


def batches(rows, epochs, seed):
    selected = sorted([r for r in rows if r['fold']=='TRAIN'], key=lambda r:r['id'])
    if len(selected) != 240 or len({r['id'] for r in selected}) != 240:
        raise ValueError('Every TRAIN scene required each epoch')
    result = []
    for epoch in range(epochs):
        ordered = list(selected); random.Random(seed+epoch).shuffle(ordered)
        result.extend(ordered[i:i+4] for i in range(0,240,4))
    return result


def main(a):
    guarded(a.lease, 1800)
    root = Path(a.root); out = root/a.arm
    proof = json.loads((root/'fresh-process-restore.json').read_text())
    sizing = json.loads((root/'admission.json').read_text())
    init = root/'expanded-init.pt'
    reused=None
    if (root/'reuse-manifest.json').exists():
        from .sustained import validate
        reused=validate(root,a.source)
    original_source=reused['original_source'] if reused else a.source
    if (not proof['passed'] or (not a.resume and proof['restore_pid'] == os.getpid()) or not proof['actual_CUDA']
            or proof['source'] != a.source or proof['checkpoint_sha256'] != sha(init)
            or proof['new_optimizer_updates'] != 0 or sizing['source'] != original_source):
        raise ValueError('Actual independent expanded native state proof required')
    if sha(a.parent) != PARENT_SHA: raise ValueError('Exact preserved466 parent required')
    parent = torch.load(a.parent, map_location='cpu', weights_only=False)
    corpus = Corpus(a.cache, action_index=a.action_index, admission=a.data_admission)
    if not a.resume: out.mkdir(exist_ok=False)
    elif not out.is_dir(): raise ValueError('Resume only an actual existing arm')
    model, _ = load_native(a.model, language_only=True); llm = model.llm
    llm.eval(); llm.gradient_checkpointing_disable()
    full = a.arm == 'main'
    reader = lambda cache,row: corpus.read(row, full=full)
    recipe = dict(parent['metadata']['recipe'], epochs=8, updates=480, maximum_complete_epochs=8,
                  train_count=240, dev_count=64, seed=20261104, learning_rate=1e-5,
                  warmup_updates=16, activation_offload=sizing['measured']['activation_offload'])
    recipe['activation_offload_cpu_limit_bytes']=sizing['measured']['activation_offload_cpu_limit_bytes']
    recipe['activation_offload_policy']=sizing['measured'].get('activation_offload_policy','legacy-pinned')
    recipe['opal'] = dict(recipe['opal'], backend='keep', agent_budget_usd=0.)
    if full:
        recipe.update(rank=256, alpha=256, last_layers=36,
                      projections=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
                      forward_path='native-segmented-bptt-v1')
    runtime = None
    try:
        if a.resume:
            saved = torch.load(a.resume, map_location='cpu', weights_only=False)
            meta = saved['metadata']; recipe = meta['recipe']
            if (meta['capacity_schema'] != 'native-capacity-trained-v1' or meta['arm'] != a.arm
                    or meta['source'] != a.source or meta['parent_checkpoint_sha256'] != PARENT_SHA
                    or meta['cache_sha256'] != MANIFEST_SHA or not 0 < saved['step'] < recipe['updates']):
                raise ValueError('Exact current arm/step/state required; never repeat saved updates')
            if full:
                runtime = restore_expanded(saved, llm, a.opal_source, out/('resume-'+str(saved['step'])))
            else:
                from .lora import install
                from .checkpoint_restore import restore
                install(llm, rank=64, alpha=64, last_layers=1)
                base = torch.optim.AdamW([p for p in llm.parameters() if p.requires_grad], lr=recipe['learning_rate'])
                runtime, _ = restore(a.resume, sha(a.resume), llm, base, source=a.opal_source,
                                     run_dir=out/('resume-'+str(saved['step'])))
            start = saved['step']
            from .resume_native import validate_state,verify_proof
            lease=json.loads(Path(a.lease).read_text());evidence=json.loads(Path(lease['trained_state_receipt_path']).read_text())
            validate_state(saved,evidence)
            resumed=json.loads((root/'fresh-resume-0008.json').read_text())
            verify_proof(resumed,evidence,lease,os.getpid(),Path('/proc/sys/kernel/random/boot_id').read_text().strip())
            if sha(a.resume)!=resumed['checkpoint_sha256']:raise ValueError('Resumed checkpoint differs')
            from .lora import selected_forward
            ref=next(r for r in corpus.rows if r['id']==resumed['native_reference']['id'])
            with torch.no_grad():ref_logits,_=selected_forward(llm,reader(a.cache,ref),'cuda')
            if tensor_sha(ref_logits)!=resumed['native_reference']['logits_sha256']:raise ValueError('Independent native resume predictions differ')
            restore_rng(saved['rng'])
            meta=copy.deepcopy(meta)
            meta.setdefault('initial_training_native_pid',meta['training_native_pid'])
            meta['training_native_pid']=os.getpid()
            meta.setdefault('resume_executions',[]).append(dict(execution_source=lease['execution_source'],saved_step=start,
                checkpoint_sha256=resumed['checkpoint_sha256'],independent_restore_pid=resumed['restore_pid'],training_pid=os.getpid()))
            # Do not reset OPAL/scheduler or repeat the first8 batches.
            admitted=meta['admitted']
            if admitted['epochs']!=1 or admitted['updates_each_arm']!=60:raise ValueError('Same complete schedule required')
            if lease['stop_unix']-time.time()-180<admitted['joint_required_seconds']:
                raise TimeoutError('Original conservative complete-pair timing no longer fits')
            write(out,'resume-from-0008.json',dict(source=a.source,execution_source=lease['execution_source'],
                actual_saved_step=start,new_updates_before_this_process=8,all_state_exact=True,native_reference_exact=True,
                restore_pid=resumed['restore_pid'],training_pid=os.getpid(),no_saved_updates_replayed=True))
        else:
            if full:
                initialized = torch.load(init, map_location='cpu', weights_only=False)
                runtime = restore_expanded(initialized, llm, a.opal_source, out/'initial-restore')
            else:
                from .lora import install
                from .checkpoint_restore import restore
                install(llm, rank=64, alpha=64, last_layers=1)
                base = torch.optim.AdamW([p for p in llm.parameters() if p.requires_grad], lr=recipe['learning_rate'])
                runtime, _ = restore(a.parent, PARENT_SHA, llm, base, source=a.opal_source, run_dir=out/'parent-restore')
            before = time.monotonic()
            if reused:
                baseline=json.loads((root/'original-proofs/main/dev-baseline.json').read_text())
                if not full:
                    record=json.loads((root/'control-runtime.json').read_text())
                    baseline=dict(record['dev'],source=record['source'],seconds=record['measured']['full_dev_seconds'],scenes=record['full_dev_scenes'])
                dev_seconds=baseline['seconds']
                if baseline['source']!=original_source or baseline['scenes']!=64:raise ValueError('Complete original baseline required')
                write(out,'dev-baseline-reused.json',dict(original_source=original_source,controller_source=a.source,original_path='original-proofs/main/dev-baseline.json' if full else 'original-proofs/control-runtime.json',new_measurement=False,seconds=dev_seconds,scenes=64))
            else:
                baseline = core.dev_loss(llm, a.cache, corpus.rows, True, reader=reader, loss_fn=corpus.loss, recipe=recipe)
                dev_seconds = time.monotonic()-before
                write(out, 'dev-baseline.json', dict(source=a.source, **baseline, seconds=dev_seconds, scenes=64))
            if full:
                lease = json.loads(Path(a.lease).read_text())
                timing={}
                if (root/'control-runtime.json').exists():
                    record=json.loads((root/'control-runtime.json').read_text());work=json.loads((root/'workload-geometry.json').read_text())
                    if record['source']!=original_source or work.pop('source')!=original_source or record['cache_manifest_sha256']!=MANIFEST_SHA or record['full_dev_scenes']!=64:
                        raise ValueError('Actual control/data/source runtime binding required')
                    timing=dict(workload=work,control=record['measured'])
                admitted = admission(sizing['measured']['seconds'],dev_seconds,lease['stop_unix']-time.time()-180,init.stat().st_size,**timing)
                if reused and (admitted['epochs']!=1 or admitted['updates_each_arm']!=60):raise ValueError('Exactly one complete paired epoch approved')
                if reused:admitted['maximum_all_branch_updates']=120
                write(root, 'joint-schedule-admission.json', admitted)
            else:
                admitted = json.loads((root/'joint-schedule-admission.json').read_text())
                terminal = json.loads((root/'main/training-terminal.json').read_text())
                if not terminal['completed'] or terminal['actual_new_updates'] != admitted['updates_each_arm']:
                    raise ValueError('Complete main before the matched control; no overlapping arms')
            recipe.update(epochs=admitted['epochs'], updates=admitted['updates_each_arm'])
            restore_rng(parent['rng'])
            if full:
                from .large_phase import begin
                runtime, transition = begin(parent, runtime, initialized['optimizer'], llm,
                    source=a.opal_source, run_dir=out/'phase-opal', program_dir=out/'phase-programs',
                    recipe=recipe, expansion=initialized['metadata']['expansion'])
            else:
                from .phase_transition import begin
                runtime, transition = begin(parent, runtime, llm, source=a.opal_source,
                    run_dir=out/'phase-opal', program_dir=out/'phase-programs', recipe=recipe)
            write(out, 'phase-transition.json', transition)
            meta = dict(capacity_schema='native-capacity-trained-v1', arm=a.arm, source=a.source,
                        model_revision=corpus.manifest['model_revision'], recipe=copy.deepcopy(recipe),
                        parent_training_steps=466, parent_checkpoint_sha256=PARENT_SHA, cache_sha256=MANIFEST_SHA,
                        initial_programs=parent['optimizer']['files'], admitted=admitted, adopted=False,
                        independent_native_restore=proof, expanded_init_checkpoint_sha256=proof['checkpoint_sha256'],
                        training_native_pid=os.getpid())
            if full: meta['expansion'] = initialized['metadata']['expansion']
            write(out, 'recipe.json', meta)
            start = 0
        ordered = batches(corpus.rows, recipe['epochs'], recipe['seed'])
        if len(ordered) != recipe['updates'] or runtime.optimizer.update != start:
            raise ValueError('Exact admitted schedule/OPAL clock required')
        log = (out/('training-resume-%04d.jsonl'%start if a.resume else 'training.jsonl')).open('x')
        frozen = {n:p._version for n,p in llm.named_parameters() if not p.requires_grad}
        actual = start
        try:
            for step in range(start, recipe['updates']):
                guarded(a.lease, sizing['measured']['seconds']*1.5+180)
                row = core.train_step(llm, runtime.optimizer, a.cache, ordered[step], step, opal=runtime,
                                      reader=reader, loss_fn=corpus.loss, recipe=recipe)
                actual = runtime.optimizer.update
                if any(p._version != frozen[n] for n,p in llm.named_parameters() if n in frozen):
                    raise ValueError('Frozen model mutated during capacity training')
                log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row), flush=True)
                if actual % 8 == 0:
                    runtime.diagnose([row], actual)
                    # Atomic replacement keeps one rolling resumable state;
                    # every complete-epoch state is retained separately below.
                    # This avoids losing a long partial epoch on reauthentication
                    # while respecting the declared artifact-byte admission.
                    live_hash=save(out/'live-resume.pt',llm,runtime.optimizer,actual,meta,
                                   diagnostics_state=runtime.diagnostics_state)
                    log.write(json.dumps(dict(event='live_state_saved',step=actual,sha256=live_hash))+'\n');log.flush()
                if actual % 60 == 0:
                    guarded(a.lease, 180)
                    dev = core.dev_loss(llm, a.cache, corpus.rows, True, reader=reader, loss_fn=corpus.loss, recipe=recipe)
                    runtime.development(dev['ce']); runtime.diagnose([row], actual, dev=dev['per_family'])
                    write(out, f'dev-{actual:04d}.json', dict(step=actual, scenes=64, **dev, adopted=False))
                    save(out/f'step-{actual:04d}.pt', llm, runtime.optimizer, actual, meta,
                         diagnostics_state=runtime.diagnostics_state)
            from .lora import selected_forward
            reference_row=next(r for r in sorted(corpus.rows,key=lambda r:r['id']) if r['fold']=='TRAIN' and r['family']=='ordinary')
            with torch.no_grad():reference_logits,_=selected_forward(llm,reader(a.cache,reference_row),'cuda')
            write(out,'final-native-reference.json',dict(id=reference_row['id'],logits_sha256=tensor_sha(reference_logits),
                  checkpoint_sha256=sha(out/f'step-{actual:04d}.pt'),source=a.source,training_native_pid=os.getpid()))
            write(out, 'training-terminal.json', dict(completed=True, actual_new_updates=actual,
                    complete_epochs=actual//60, train_scenes_per_epoch=240, full_dev_scenes=64,
                    parent_training_steps=466, adopted=False, native_behavior_and_speech_required=True))
        except BaseException:
            # Read the real OPAL clock, including a completed step whose later
            # logging failed. Never guess progress from a loop variable.
            actual = runtime.optimizer.update
            save(out/'live-resume.pt', llm, runtime.optimizer, actual, meta,
                 diagnostics_state=runtime.diagnostics_state)
            raise
        finally:
            log.close()
    finally:
        if runtime is not None: runtime.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model','cache','action-index','data-admission','opal-source','lease','root','source','parent'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--arm', choices=('main','control'), required=True)
    p.add_argument('--resume')
    main(p.parse_args())
