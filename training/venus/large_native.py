"""Native broad-adapter zero-update admission, with separate-process restore.

The producer grants no compute and contains no optimizer step. The training
continuation consumes its actual terminal proofs, rather than a synthetic pass.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import torch

from .large_inputs import Corpus, PARENT_SHA, MANIFEST_SHA, sha
from .large_capacity import expand_broad
from .lora import install, adapter_parameters, adapter_state, load_adapter, selected_forward
from .native_optimizer import NativeOpal, assert_optimizer_equal, gradient_norm
from .state import restore_rng, rng_state, exact_state, save
from .offload import ActivationOffload


def write(folder, name, value):
    path = Path(folder)/name
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def guarded(lease_path, needed=0):
    from portable.venus.native_app import ensure_guard
    from .large_guard import resource
    lease = json.loads(Path(lease_path).read_text()); resource(lease); ensure_guard()
    if time.time() + needed >= lease['stop_unix'] - 180:
        raise TimeoutError('Preserve full state and copyback before native STOP')


def tensor_sha(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def memory_fallback(attempt,cleanup,record,*,start_offloaded=False):
    """Release the exception/traceback graph BEFORE a lossless retry."""
    failed=False
    try:
        result=attempt(start_offloaded)
    except torch.cuda.OutOfMemoryError:
        if start_offloaded:raise
        failed=True
    # Running inside `except` retains the first failed frame and its tensors;
    # empty_cache cannot release those still-live allocations.
    if failed:
        cleanup();record()
        result=attempt(True)
    return result


def restore_expanded(state, llm, source, output):
    """Restore all original and new named states; no uniform-counter shortcut."""
    from .large_phase import write_programs
    spec = state['metadata']['expansion']
    install(llm, rank=64, alpha=64, last_layers=1)
    actual = expand_broad(llm, last_layers=36, rank=256, alpha=256, seed=20261104)
    if actual != spec:
        raise ValueError('Exact declared expansion required for fresh restore')
    load_adapter(llm, state['adapter'])
    params = adapter_parameters(llm)
    base = torch.optim.AdamW(list(params.values()), lr=state['metadata']['recipe']['learning_rate'],
                            betas=(.9,.999), eps=1e-8, weight_decay=.01)
    programs = write_programs(Path(output)/'restored-programs', state['optimizer']['files'])
    runtime = NativeOpal(base, llm, source=source, run_dir=Path(output)/'restored-opal',
                         recipe=state['metadata']['recipe'], initial_program=programs)
    try:
        runtime.optimizer.load_state_dict(state['optimizer'])
        runtime.restore_diagnostics(state['training_diagnostics'])
        restore_rng(state['rng'])
        assert_optimizer_equal(state['optimizer'], runtime.optimizer.state_dict())
        if not exact_state(state['adapter'], adapter_state(llm)) or not exact_state(state['rng'], rng_state()):
            raise ValueError('Expanded adapter or CPU/CUDA RNG restoration differs')
        return runtime
    except BaseException:
        runtime.close()
        raise


def benchmark(llm, runtime, corpus, rows, lease_path, *, offload,host_limit_bytes=48*1024**3,policy='legacy-pinned',progress_path=None,device='cuda'):
    """Real effective batch4, full temporal gradients and materialized Adam."""
    guarded(lease_path, 180)
    params = adapter_parameters(llm)
    frozen = {n: p._version for n,p in llm.named_parameters() if not p.requires_grad}
    runtime.base.zero_grad(set_to_none=True)
    if device not in ('cuda','cpu'):raise ValueError('Explicit CUDA or nonqualifyingCPU fixture required')
    actual_cuda=device=='cuda'
    if actual_cuda:torch.cuda.reset_peak_memory_stats()
    before = time.monotonic(); parts = []
    def progress(value):
        if progress_path is not None:
            with Path(progress_path).open('a') as f:f.write(json.dumps(dict(at_unix=time.time(),**value))+'\n')
            print(json.dumps(value),flush=True)
    for row in rows:
        row_start=time.monotonic();progress(dict(event='calibration_row_start',id=row['id']))
        guarded(lease_path, 180)
        ex = corpus.read(row)
        context = ActivationOffload(llm,maximum_host_bytes=host_limit_bytes,policy=policy)
        from contextlib import nullcontext
        with context.context() if offload else nullcontext():
            loss, metrics = corpus.loss(llm, ex, device)
            if not torch.isfinite(loss): raise ValueError('Nonfinite full native objective')
            (loss/len(rows)).backward()
        parts.append(dict(id=row['id'], family=row['family'], **metrics,
                          offloaded_bytes=context.saved_bytes,logical_saved_bytes=context.logical_bytes,
                          bit_exact_compressed_tensors=context.compressed_tensors,
                          verified_prefix_reuses=context.prefix_reuses))
        progress(dict(event='calibration_row_completed',id=row['id'],seconds=time.monotonic()-row_start,physical_saved_bytes=context.saved_bytes,prefix_reuses=context.prefix_reuses))
        del ex, loss
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values()):
        raise ValueError('Missing or nonfinite full-architecture gradients')
    # New A is intentionally zero-gradient until the zero-initialized B learns.
    b_norms = {n: float(torch.linalg.vector_norm(p.grad)) for n,p in params.items() if n.endswith(('.b','.extra_b'))}
    if not b_norms or any(v <= 0 for v in b_norms.values()):
        raise ValueError('Every intended attention/MLP B path must receive gradients')
    norm = float(gradient_norm(params.values()))
    if actual_cuda:torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() if actual_cuda else 0
    reserved = torch.cuda.max_memory_reserved() if actual_cuda else 0
    total = torch.cuda.get_device_properties(0).total_memory if actual_cuda else 0
    if any(p._version != frozen[n] for n,p in llm.named_parameters() if n in frozen):
        raise ValueError('Frozen foundation weights changed during no-update admission')
    result = dict(seconds=time.monotonic()-before,actual_CUDA=actual_cuda,nonqualifying_CPU_fixture=not actual_cuda,peak_allocated_bytes=peak, peak_reserved_bytes=reserved,
                  device_memory_bytes=total, headroom_bytes=total-reserved,
                  full_temporal_autograd=True, optimizer_updates=0, effective_batch=len(rows),
                  trainable_parameters=sum(p.numel() for p in params.values()),
                  nonzero_B_gradient_paths=len(b_norms), gradient_norm=norm,
                  activation_offload=offload, examples=parts,
                  activation_offload_cpu_limit_bytes=host_limit_bytes,
                  activation_offload_policy=policy,
                  all_Adam_moments_materialized=len(runtime.base.state)==len(params))
    runtime.base.zero_grad(set_to_none=True)
    if not result['all_Adam_moments_materialized'] or actual_cuda and result['headroom_bytes'] < max(2*1024**3, total*.05):
        raise MemoryError('Full workload has less than the declared2GiB/5percent memory headroom')
    return result


def prepare(a):
    from .native_cache import load_native
    from .checkpoint_restore import restore
    from .capacity_state import expand_opal_state
    from .large_phase import write_programs
    guarded(a.lease, 600)
    from .large_guard import resource
    limits=resource(json.loads(Path(a.lease).read_text()))
    host_gib=getattr(a,'host_offload_limit_gib',48)
    if host_gib not in (48,128) or (host_gib==128 and limits['machine']!='a2-ultragpu-1g'):
        raise ValueError('128GiB offload is only admitted on the approved170GiB A10080 host')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    if getattr(a,'activation_memory_policy','legacy-pinned')=='repeat4-prefix-pageable-v3-layout':
        from .layout_preflight import check
        check(report_path=out/'cuda-layout-preflight.json')
    corpus = Corpus(a.cache, action_index=a.action_index, admission=a.data_admission)
    if sha(a.parent) != PARENT_SHA: raise ValueError('Exact candidate466 parent required')
    parent = torch.load(a.parent, map_location='cpu', weights_only=False)
    if (parent['step'] != 60 or parent['metadata']['parent_training_steps'] != 406
            or parent['optimizer']['update'] != 60):
        raise ValueError('Never repeat saved406-to466 updates')
    # Validate ALL304 original hashes/targets before model allocation, not just
    # the four workload examples used below for memory sizing.
    from .workload import geometry,whole_corpus
    calibration=sorted([r for r in corpus.rows if r['fold']=='TRAIN'],key=lambda r:r['tokens']*(1.5 if r['family']=='ordinary' else 1),reverse=True)[:4]
    shapes=[]
    for row in corpus.rows:shapes.append(geometry(row,corpus.read(row)))
    work=whole_corpus(shapes,[r['id'] for r in calibration]);write(out,'workload-geometry.json',dict(source=a.source,**work))
    model, _ = load_native(a.model, language_only=True); llm = model.llm
    llm.eval(); llm.gradient_checkpointing_disable()
    if limits['machine']=='a2-ultragpu-1g' and getattr(a,'activation_memory_policy','legacy-pinned')=='legacy-pinned':
        from .large_memory_probe import probe
        write(out,'whole-causal-native-head-shape-diagnostic.json',probe(model))
    if len(llm.model.layers) != 36: raise ValueError('Pinned native36-layer model required')
    install(llm, rank=64, alpha=64, last_layers=1)
    base = torch.optim.AdamW(list(adapter_parameters(llm).values()), lr=parent['metadata']['recipe']['learning_rate'])
    old, parent_proof = restore(a.parent, PARENT_SHA, llm, base, source=a.opal_source, run_dir=out/'parent-opal')
    if getattr(a,'measured_joint_schedule',False):
        from types import SimpleNamespace
        from . import train as core
        control=benchmark(llm,old,SimpleNamespace(read=lambda r:corpus.read(r,full=False),loss=corpus.loss),calibration,a.lease,offload=True,policy=a.activation_memory_policy,progress_path=out/'control-calibration.jsonl')
        started=time.monotonic()
        dev=core.dev_loss(llm,a.cache,corpus.rows,True,reader=lambda _,r:corpus.read(r,full=False),loss_fn=corpus.loss,recipe=dict(parent['metadata']['recipe'],dev_count=64))
        control['full_dev_seconds']=time.monotonic()-started
        write(out,'control-runtime.json',dict(source=a.source,cache_manifest_sha256=MANIFEST_SHA,new_optimizer_updates=0,measured=control,full_dev_scenes=64,dev=dev))
        assert_optimizer_equal(parent['optimizer'],old.optimizer.state_dict());restore_rng(parent['rng'])
    row = next(r for r in sorted(corpus.rows, key=lambda r:r['id']) if r['fold']=='TRAIN' and r['family']=='ordinary')
    check = corpus.read(row)
    with torch.no_grad(): before = selected_forward(llm, check, 'cuda')[0]
    expansion = expand_broad(llm, last_layers=36, rank=256, alpha=256, seed=20261104)
    if expansion['parameters'] != 698351616: raise ValueError('Exact declared698M profile required')
    with torch.no_grad(): after = selected_forward(llm, check, 'cuda')[0]
    if not torch.equal(before, after): raise ValueError('Expansion changes inherited native predictions before training')
    reference = dict(id=row['id'], logits_sha256=tensor_sha(after))
    del before, after, check
    mapped = expand_opal_state(parent['optimizer'], expansion['before_names'], adapter_parameters(llm))
    old.close(); del old, base
    base = torch.optim.AdamW(list(adapter_parameters(llm).values()), lr=parent['metadata']['recipe']['learning_rate'])
    programs = write_programs(out/'preserved-programs', mapped['files'])
    runtime = NativeOpal(base, llm, source=a.opal_source, run_dir=out/'expanded-opal',
                         recipe=parent['metadata']['recipe'], initial_program=programs)
    try:
        runtime.optimizer.load_state_dict(mapped); runtime.restore_diagnostics(parent['training_diagnostics'])
        restore_rng(parent['rng']); assert_optimizer_equal(mapped, runtime.optimizer.state_dict())
        calibration = sorted([r for r in corpus.rows if r['fold']=='TRAIN'],
                             key=lambda r:r['tokens']*(1.5 if r['family']=='ordinary' else 1), reverse=True)[:4]
        def cleanup():
            runtime.base.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        measured=memory_fallback(
            lambda offload:benchmark(llm,runtime,corpus,calibration,a.lease,offload=offload,host_limit_bytes=host_gib*1024**3,
                                     policy=getattr(a,'activation_memory_policy','legacy-pinned'),progress_path=out/'main-calibration.jsonl'),cleanup,
            lambda:write(out,'no-offload-oom.json',dict(optimizer_updates=0,lossless_offload_fallback=True)),
            start_offloaded=bool(getattr(a,'activation_offload',False)))
        if not measured['actual_CUDA']:raise ValueError('Actual nativeCUDA gradient admission required')
        restore_rng(parent['rng'])
        assert_optimizer_equal(mapped, runtime.optimizer.state_dict())
        metadata = dict(parent['metadata'], expansion=expansion, capacity_schema='native-broad-init-v1',
                        capacity_source=a.source, original_parent_sha256=PARENT_SHA,
                        native_reference=reference, initialization_pid=os.getpid())
        checkpoint_hash = save(out/'expanded-init.pt', llm, runtime.optimizer, parent['step'], metadata,
                               diagnostics_state=runtime.diagnostics_state)
        write(out, 'admission.json', dict(source=a.source, native_process_pid=os.getpid(), actual_CUDA=True,
                    checkpoint_sha256=checkpoint_hash, parent_sha256=PARENT_SHA, full304_hashes_and_masks_verified=True,
                    parent_restore=parent_proof, exact_initial_native_logits=True, native_reference=reference,
                    measured=measured, new_optimizer_updates=0, independent_restore_required=True, adopted=False))
    finally:
        runtime.close()


def restore_only(a):
    from .native_cache import load_native
    guarded(a.lease, 300)
    root = Path(a.output); proof = json.loads((root/'admission.json').read_text())
    checkpoint = root/'expanded-init.pt'
    expected_source=a.source
    if (root/'reuse-manifest.json').exists():
        from .sustained import validate
        expected_source=validate(root,a.source)['original_source']
    if (proof['native_process_pid'] == os.getpid() or proof['source'] != expected_source
            or sha(checkpoint) != proof['checkpoint_sha256'] or proof['new_optimizer_updates'] != 0):
        raise ValueError('Actual independent process and exact native initialization proof required')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    folder = root/'fresh-restore'; folder.mkdir(exist_ok=False)
    corpus = Corpus(a.cache, action_index=a.action_index, admission=a.data_admission)
    model, _ = load_native(a.model, language_only=True); llm = model.llm
    llm.eval(); llm.gradient_checkpointing_disable()
    runtime = restore_expanded(state, llm, a.opal_source, folder)
    try:
        row = next(r for r in corpus.rows if r['id'] == proof['native_reference']['id'])
        with torch.no_grad(): logits, _ = selected_forward(llm, corpus.read(row), 'cuda')
        if tensor_sha(logits) != proof['native_reference']['logits_sha256']:
            raise ValueError('Fresh-process expanded native predictions differ')
        restore_rng(state['rng'])
        if not exact_state(state['rng'], rng_state()): raise ValueError('Fresh CPU/CUDA RNG differs')
        write(root, 'fresh-process-restore.json', dict(passed=True, source=a.source,
              checkpoint_sha256=proof['checkpoint_sha256'], initialization_pid=proof['native_process_pid'],
              restore_pid=os.getpid(), actual_CUDA=True, all_Adam_and_OPAL_state_exact=True,
              rng_exact=True, native_logits_exact=True, new_optimizer_updates=0, adopted=False))
    finally:
        runtime.close()


def final_restore(a):
    from .native_cache import load_native
    guarded(a.lease,300)
    root=Path(a.output);terminal=json.loads((root/'training-terminal.json').read_text())
    reference=json.loads((root/'final-native-reference.json').read_text())
    checkpoint=root/('step-%04d.pt'%terminal['actual_new_updates'])
    if (not terminal['completed'] or reference['source']!=a.source
            or reference['training_native_pid']==os.getpid() or sha(checkpoint)!=reference['checkpoint_sha256']):
        raise ValueError('Actual successful training and distinct source-bound restore process required')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if (state['metadata']['capacity_schema']!='native-capacity-trained-v1'
            or state['step']!=terminal['actual_new_updates'] or state['metadata']['source']!=a.source):
        raise ValueError('Exact saved completed capacity checkpoint required')
    corpus=Corpus(a.cache,action_index=a.action_index,admission=a.data_admission)
    model,_=load_native(a.model,language_only=True);llm=model.llm
    llm.eval();llm.gradient_checkpointing_disable()
    full=state['metadata']['arm']=='main'
    if full:runtime=restore_expanded(state,llm,a.opal_source,root/'final-restore')
    else:
        from .checkpoint_restore import restore
        install(llm,rank=64,alpha=64,last_layers=1)
        base=torch.optim.AdamW(list(adapter_parameters(llm).values()),lr=state['metadata']['recipe']['learning_rate'])
        runtime,_=restore(checkpoint,reference['checkpoint_sha256'],llm,base,source=a.opal_source,run_dir=root/'final-restore')
    try:
        row=next(r for r in corpus.rows if r['id']==reference['id'])
        with torch.no_grad():logits,_=selected_forward(llm,corpus.read(row,full=full),'cuda')
        if tensor_sha(logits)!=reference['logits_sha256']:raise ValueError('Final independently restored native logits differ')
        restore_rng(state['rng'])
        assert_optimizer_equal(state['optimizer'],runtime.optimizer.state_dict())
        write(root,'final-native-restore.json',dict(passed=True,source=a.source,
                checkpoint_sha256=reference['checkpoint_sha256'],training_native_pid=reference['training_native_pid'],
                restore_native_pid=os.getpid(),actual_CUDA=True,optimizer_RNG_OPAL_exact=True,
                native_logits_exact=True,new_optimizer_updates=0,adopted=False))
    finally:runtime.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('prepare','restore','final-restore'), required=True)
    for name in ('model','cache','action-index','data-admission','opal-source','lease','output','source'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--parent')
    p.add_argument('--measured-joint-schedule',action='store_true')
    p.add_argument('--activation-offload',action='store_true',help='Lossless saved-activation offload; original full gradients/objectives remain')
    p.add_argument('--host-offload-limit-gib',type=int,choices=(48,128),default=48)
    p.add_argument('--activation-memory-policy',choices=('legacy-pinned','repeat4-pageable-v1','repeat4-prefix-pageable-v2','repeat4-prefix-pageable-v3-layout'),default='legacy-pinned')
    a = p.parse_args()
    if a.stage == 'prepare' and not a.parent: p.error('--parent required for preparation')
    prepare(a) if a.stage == 'prepare' else restore_only(a) if a.stage=='restore' else final_restore(a)
