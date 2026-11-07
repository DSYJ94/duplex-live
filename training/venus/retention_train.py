"""One complete matched-data epoch from saved646, with exact native state."""
import argparse,copy,json,os,subprocess,time
from pathlib import Path

PARENT_SHA='46e84c6d7b1d02ec16563c1d0e666230f8644963d3406b0f095bab97938950aa'
PARENT_SOURCE='d8161fd40c80b814242f93c8d205cf33a83c4c54'


def validate_parent(state,digest):
    meta=state['metadata'];recipe=meta['recipe']
    if (digest!=PARENT_SHA or state['step']!=120 or state['optimizer']['update']!=120
            or meta['source']!=PARENT_SOURCE or meta['parent_training_steps']!=526
            or meta['capacity_schema']!='native-capacity-continuation-v1'
            or recipe['updates']!=120 or recipe['learning_rate']!=5e-6):
        raise ValueError('Exact completed646 state required; never replay saved updates')


def main(a):
    import torch
    from .isambard_native import BASE,ART,guarded
    from .retention_inputs import Corpus,epoch_batches
    from .large_inputs import sha
    from .native_cache import load_native
    from .isambard_train import restore_runtime
    from .large_native import tensor_sha,write
    from .lora import selected_forward
    from .state import save,restore_rng,exact_state,rng_state
    from .native_optimizer import assert_optimizer_equal
    from .phase_transition import begin
    from . import train as core
    guarded(a.lease,1800)
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=a.source or json.loads(Path(a.lease).read_text())['source']!=a.source:
        raise ValueError('Actual source-bound scheduler process required')
    corpus=Corpus(ART/'cache-v52',a.new_cache,action_index=ART/'cache-v52/action-index.json',
        old_admission='docs/NATIVE-VENUS-CONTEXT-TRAINING-ADMISSION-V3-2026-11-02.json',
        new_manifest_sha=a.new_manifest_sha,semantic_root=a.semantic_root,
        timelines=BASE/'native-retention96-timelines-v3')
    out=Path(a.root)/a.arm
    reference_row=next(r for r in sorted(corpus.old.rows,key=lambda r:r['id']) if r['fold']=='TRAIN' and r['family']=='ordinary')
    if a.final_restore:
        terminal=json.loads((out/'training-terminal.json').read_text())
        reference=json.loads((out/'final-native-reference.json').read_text())
        if not terminal['completed'] or terminal['actual_new_updates']!=84 or reference['source']!=a.source:
            raise ValueError('Actual complete84-update epoch required')
        checkpoint=out/'step-0084.pt';expected=reference['checkpoint_sha256']
    else:
        checkpoint=BASE/'native-training-resume8-v1/main/step-0120.pt';expected=PARENT_SHA
        reference=json.loads((checkpoint.parent/'final-native-reference.json').read_text())
        proof=json.loads((checkpoint.parent/'final-native-restore.json').read_text())
        if not proof['passed'] or proof['source']!=PARENT_SOURCE or proof['checkpoint_sha256']!=expected:
            raise ValueError('Original independent646 native restore required')
        out.mkdir(parents=True,exist_ok=False)
    if sha(checkpoint)!=expected:raise ValueError('Actual full-state checkpoint hash differs')
    parent=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if not a.final_restore:validate_parent(parent,expected)
    elif parent['metadata']['source']!=a.source or parent['metadata']['arm']!=a.arm or parent['step']!=84:
        raise ValueError('Exact new arm state required')
    model,_=load_native(a.model,language_only=True);llm=model.llm;llm.eval();llm.gradient_checkpointing_disable()
    runtime,state,changes=restore_runtime(parent,llm,a.opal_source,out/('final-restore' if a.final_restore else 'parent-restore'))
    try:
        assert_optimizer_equal(state['optimizer'],runtime.optimizer.state_dict())
        if not exact_state(state['rng'],rng_state()):raise ValueError('Fresh native RNG differs')
        with torch.no_grad():logits,_=selected_forward(llm,corpus.read(reference_row),'cuda')
        if tensor_sha(logits)!=reference['logits_sha256']:raise ValueError('Independent native logits differ')
        restore_rng(state['rng']);execution=os.environ['DUPLEX_NATIVE_EXECUTION_ID']
        if a.final_restore:
            if execution==reference['training_execution_id']:raise ValueError('Distinct native final restore required')
            write(out,'final-native-restore.json',dict(passed=True,source=a.source,arm=a.arm,
                checkpoint_sha256=expected,training_execution_id=reference['training_execution_id'],
                restore_execution_id=execution,Adam_RNG_OPAL_exact=True,native_logits_exact=True,
                actual_CUDA=True,new_optimizer_updates=0,adopted=False));return
        recipe=copy.deepcopy(parent['metadata']['recipe'])
        recipe.update(epochs=1,updates=84,warmup_updates=16,seed=20261107,train_count=336,dev_count=64)
        runtime,transition=begin(state,runtime,llm,source=a.opal_source,run_dir=out/'phase-opal',
            program_dir=out/'phase-programs',recipe=recipe)
        metadata=dict(capacity_schema='native-retention96-data-v1',source=a.source,arm=a.arm,
            model_revision=parent['metadata']['model_revision'],recipe=recipe,expansion=parent['metadata']['expansion'],
            parent_training_steps=646,parent_checkpoint_sha256=PARENT_SHA,original_capacity_metadata=parent['metadata'],
            composite_manifest=corpus.manifest,phase_transition=transition,adopted=False)
        write(out,'recipe.json',metadata)
        write(out,'parent-restore.json',dict(passed=True,source=a.source,checkpoint_sha256=expected,
            Adam_RNG_OPAL_exact=True,native_logits_exact=True,actual_CUDA=True,new_optimizer_updates=0,
            original_independent_proof=proof,derived_Adam_format_defaults=changes))
        ordered=epoch_batches(corpus.old.rows,corpus.new_rows,a.arm,recipe['seed'])
        write(out,'sampler.json',dict(batches=[[r['id'] for r in batch] for batch in ordered],
            exposures=336,updates=84,DEV_used=False))
        frozen={n:p._version for n,p in llm.named_parameters() if not p.requires_grad}
        actual=0;last_seconds=600.
        with (out/'training.jsonl').open('x') as log:
            try:
                for step,batch in enumerate(ordered):
                    guarded(a.lease,max(300,last_seconds*1.5)+180)
                    row=core.train_step(llm,runtime.optimizer,ART/'cache-v52',batch,step,opal=runtime,
                        reader=lambda c,r:corpus.read(r),loss_fn=corpus.loss,recipe=recipe)
                    actual=runtime.optimizer.update;last_seconds=row['seconds']
                    if any(p._version!=frozen[n] for n,p in llm.named_parameters() if n in frozen):raise ValueError('Frozen foundation changed')
                    row.update(source=a.source,arm=a.arm,global_optimizer_updates=646+actual)
                    log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                    if actual%8==0:runtime.diagnose([row],actual)
                    digest=save(out/'live-resume.pt',llm,runtime.optimizer,actual,metadata,diagnostics_state=runtime.diagnostics_state)
                    log.write(json.dumps(dict(event='live_state_saved',step=actual,sha256=digest))+'\n');log.flush()
                dev=core.dev_loss(llm,ART/'cache-v52',corpus.rows,True,reader=lambda c,r:corpus.read(r),loss_fn=corpus.loss,recipe=recipe)
                runtime.development(dev['ce']);runtime.diagnose([row],actual,dev=dev['per_family'])
                write(out,'dev-0084.json',dict(step=actual,scenes=64,**dev,adopted=False))
                save(out/'step-0084.pt',llm,runtime.optimizer,actual,metadata,diagnostics_state=runtime.diagnostics_state)
                with torch.no_grad():logits,_=selected_forward(llm,corpus.read(reference_row),'cuda')
                write(out,'final-native-reference.json',dict(source=a.source,training_execution_id=execution,
                    checkpoint_sha256=sha(out/'step-0084.pt'),logits_sha256=tensor_sha(logits)))
                write(out,'training-terminal.json',dict(completed=True,actual_new_updates=actual,
                    total_optimizer_updates=646+actual,TRAIN_exposures=336,DEV=64,arm=a.arm,adopted=False))
            except BaseException:
                save(out/'interrupted-state.pt',llm,runtime.optimizer,runtime.optimizer.update,metadata,diagnostics_state=runtime.diagnostics_state)
                write(out,'training-interrupted.json',dict(completed=False,actual_new_updates=runtime.optimizer.update,source=a.source));raise
    finally:runtime.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','lease','model','opal-source','new-cache','new-manifest-sha','semantic-root'):p.add_argument('--'+name,required=True)
    p.add_argument('--arm',choices=('expanded-data','rehearsal'),required=True)
    p.add_argument('--final-restore',action='store_true');main(p.parse_args())
