"""Complete-epoch continuation of the saved698M adapter; never replay step60.

Same native objective, corpus, architecture and full OPAL/Adam state. A declared
new schedule boundary retains moments and archives the original update history.
"""
import argparse,copy,json,os,time,subprocess
from pathlib import Path

from .isambard_native import BASE,ART,SHA,adam_format,guarded

# Preserve the actual completed evaluation producer when only the Slurm
# transport wrapper evolves. Never relabel the old proof as the new trainer.
EVALUATION_SOURCE='35d40392b641d31023723d87edb72bb224af37b8'

def restore_runtime(state,llm,source,out):
    import torch
    from .large_native import restore_expanded
    mapped,changes=adam_format(state,torch)
    runtime=restore_expanded(mapped,llm,source,out)
    return runtime,mapped,changes

def main(a):
    import torch
    from .large_inputs import Corpus,MANIFEST_SHA,sha
    from .native_cache import load_native
    from .large_native import tensor_sha,write
    from .lora import selected_forward
    from .state import restore_rng,save,exact_state,rng_state
    from . import train as core
    from .large_train import batches
    from .phase_transition import begin
    guarded(a.lease,1800)
    if a.source!=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() or json.loads(Path(a.lease).read_text())['source']!=a.source:raise ValueError('Actual source-bound Slurm execution required')
    if not 1<=a.epochs<=4:raise ValueError('At most four complete new240TRAIN epochs')
    root=Path(a.root);out=root/'main'
    corpus=Corpus(ART/'cache-v52',action_index=ART/'cache-v52/action-index.json',admission='docs/NATIVE-VENUS-CONTEXT-TRAINING-ADMISSION-V3-2026-11-02.json')
    reference_row=next(r for r in sorted(corpus.rows,key=lambda r:r['id']) if r['fold']=='TRAIN' and r['family']=='ordinary')
    if a.final_restore:
        terminal=json.loads((out/'training-terminal.json').read_text());reference=json.loads((out/'final-native-reference.json').read_text())
        if not terminal['completed'] or reference['source']!=a.source:raise ValueError('Actual complete new training required')
        checkpoint=out/('step-%04d.pt'%terminal['actual_new_updates'])
    else:
        evaluation=BASE/'native-evaluation-v5'
        paired=json.loads((evaluation/'native/paired-native-report.json').read_text())
        reference=json.loads((evaluation/'migration/main/reference.json').read_text())
        proof=json.loads((evaluation/'migration/main/restore.json').read_text())
        if not paired['complete'] or len(paired['cases'])!=64 or paired['source']!=EVALUATION_SOURCE or not proof['passed'] or proof['source']!=EVALUATION_SOURCE or reference['source']!=EVALUATION_SOURCE:raise ValueError('Complete actual pairednative64 and independent new-runtime restore required')
        checkpoint=ART/'checkpoints/main-step-0060.pt';out.mkdir(parents=True,exist_ok=False)
    expected=reference['checkpoint_sha256']
    if sha(checkpoint)!=expected:raise ValueError('Exact preserved training state required')
    parent=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if a.final_restore and (parent['metadata']['capacity_schema']!='native-capacity-continuation-v1' or parent['metadata']['source']!=a.source or parent['step']!=terminal['actual_new_updates']):raise ValueError('Exact completed new phase required')
    if not a.final_restore and (expected!=SHA['main'] or parent['step']!=60):raise ValueError('Resume the actual completed60 main state only')
    model,_=load_native(a.model,language_only=True);llm=model.llm;llm.eval();llm.gradient_checkpointing_disable()
    runtime,state,format_changes=restore_runtime(parent,llm,a.opal_source,out/('final-restore' if a.final_restore else 'parent-restore'))
    try:
        with torch.no_grad():logits,_=selected_forward(llm,corpus.read(reference_row),'cuda')
        if tensor_sha(logits)!=reference['logits_sha256']:raise ValueError('Actual training/fresh final restore differs from independent native reference')
        restore_rng(state['rng'])
        if a.final_restore:
            if reference['training_pid']==os.getpid():raise ValueError('Final restore must be a distinct process')
            write(out,'final-native-restore.json',dict(passed=True,source=a.source,checkpoint_sha256=expected,training_pid=reference['training_pid'],restore_pid=os.getpid(),actual_CUDA=True,Adam_RNG_OPAL_exact=True,native_logits_exact=True,new_optimizer_updates=0,derived_format_defaults=format_changes,adopted=False))
            return
        recipe=copy.deepcopy(parent['metadata']['recipe'])
        recipe.update(learning_rate=5e-6,epochs=a.epochs,updates=60*a.epochs,warmup_updates=16,seed=20261106,train_count=240,dev_count=64)
        baseline=core.dev_loss(llm,ART/'cache-v52',corpus.rows,True,reader=lambda c,r:corpus.read(r),loss_fn=corpus.loss,recipe=recipe)
        write(out,'dev-baseline.json',dict(source=a.source,scenes=64,**baseline))
        restore_rng(state['rng'])
        runtime,transition=begin(state,runtime,llm,source=a.opal_source,run_dir=out/'phase-opal',program_dir=out/'phase-programs',recipe=recipe)
        write(out,'phase-transition.json',dict(transition,source=a.source,parent_checkpoint_sha256=SHA['main'],derived_Adam_format_defaults=format_changes))
        metadata=dict(capacity_schema='native-capacity-continuation-v1',arm='main',source=a.source,model_revision=parent['metadata']['model_revision'],recipe=recipe,
            parent_training_steps=526,parent_checkpoint_sha256=SHA['main'],parent_local_step=60,cache_sha256=MANIFEST_SHA,expansion=parent['metadata']['expansion'],
            original_capacity_metadata=parent['metadata'],independent_parent_restore=proof,parent_evaluation_source=EVALUATION_SOURCE,phase_transition=transition,adopted=False)
        write(out,'recipe.json',metadata)
        save(out/'phase-init.pt',llm,runtime.optimizer,0,metadata,diagnostics_state=runtime.diagnostics_state)
        ordered=batches(corpus.rows,a.epochs,recipe['seed']);last_seconds=600.;actual=0
        frozen={name:p._version for name,p in llm.named_parameters() if not p.requires_grad}
        with (out/'training.jsonl').open('x') as log:
            try:
                for step,batch in enumerate(ordered):
                    guarded(a.lease,max(300,last_seconds*1.5)+180)
                    row=core.train_step(llm,runtime.optimizer,ART/'cache-v52',batch,step,opal=runtime,reader=lambda c,r:corpus.read(r),loss_fn=corpus.loss,recipe=recipe)
                    actual=runtime.optimizer.update;last_seconds=row['seconds'];row.update(global_optimizer_updates=526+actual,source=a.source)
                    if any(p._version!=frozen[name] for name,p in llm.named_parameters() if name in frozen):raise ValueError('Frozen foundation weights changed')
                    if hasattr(runtime.optimizer.updater,'last_effective_rates'):row['effective_updater_rates']=list(runtime.optimizer.updater.last_effective_rates)
                    log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                    if actual%8==0:
                        runtime.diagnose([row],actual)
                        digest=save(out/'live-resume.pt',llm,runtime.optimizer,actual,metadata,diagnostics_state=runtime.diagnostics_state)
                        log.write(json.dumps(dict(event='live_state_saved',step=actual,sha256=digest))+'\n');log.flush()
                    if actual%60==0:
                        guarded(a.lease,300)
                        dev=core.dev_loss(llm,ART/'cache-v52',corpus.rows,True,reader=lambda c,r:corpus.read(r),loss_fn=corpus.loss,recipe=recipe)
                        runtime.development(dev['ce']);runtime.diagnose([row],actual,dev=dev['per_family'])
                        write(out,'dev-%04d.json'%actual,dict(step=actual,scenes=64,**dev,adopted=False))
                        save(out/('step-%04d.pt'%actual),llm,runtime.optimizer,actual,metadata,diagnostics_state=runtime.diagnostics_state)
                with torch.no_grad():logits,_=selected_forward(llm,corpus.read(reference_row),'cuda')
                write(out,'final-native-reference.json',dict(source=a.source,training_pid=os.getpid(),checkpoint_sha256=sha(out/('step-%04d.pt'%actual)),logits_sha256=tensor_sha(logits)))
                write(out,'training-terminal.json',dict(completed=True,actual_new_updates=actual,total_optimizer_updates=526+actual,complete_epochs=actual//60,TRAIN=240,DEV=64,adopted=False))
            except BaseException:
                save(out/'interrupted-state.pt',llm,runtime.optimizer,runtime.optimizer.update,metadata,diagnostics_state=runtime.diagnostics_state)
                write(out,'training-interrupted.json',dict(actual_local_updates=runtime.optimizer.update,global_optimizer_updates=526+runtime.optimizer.update,completed=False,source=a.source))
                raise
    finally:runtime.close()

if __name__=='__main__':
    from .large_behavior import parser
    p=parser();p.add_argument('--opal-source',required=True);p.add_argument('--epochs',type=int,default=2);p.add_argument('--final-restore',action='store_true');main(p.parse_args())
