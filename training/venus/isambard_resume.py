"""Independent native restore of the exact interrupted update8; no learning."""
import json,os,subprocess
from pathlib import Path
from .isambard_native import ART,adam_format,guarded

SAVED_SHA='f94985115b571f6ff7e50e186b873e3c7bb9d85cd0cd97e7b4abffa85b763a0a'
SAVED_SOURCE='d9bd2338fcf18e4f10e26d23f638761aa9f208eb'

def validate(state,digest):
    m=state['metadata'];r=m['recipe']
    if (digest!=SAVED_SHA or state['step']!=8 or state['optimizer']['update']!=8 or m['source']!=SAVED_SOURCE
            or m['capacity_schema']!='native-capacity-continuation-v1' or m['parent_training_steps']!=526
            or r['updates']!=120 or r['epochs']!=2 or r['seed']!=20261106 or r['learning_rate']!=5e-6):
        raise ValueError('Exact saved8 state, original phase recipe and clock required')

def main(a):
    import torch
    from .large_inputs import Corpus,sha
    from .native_cache import load_native
    from .large_native import restore_expanded,tensor_sha,write
    from .lora import selected_forward
    from .state import restore_rng,rng_state,exact_state
    from .native_optimizer import assert_optimizer_equal
    guarded(a.lease,600)
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=a.source or json.loads(Path(a.lease).read_text())['source']!=a.source:raise ValueError('Actual source-bound Slurm process required')
    digest=sha(a.resume_checkpoint)
    if digest!=a.resume_sha or a.resume_step!=8:raise ValueError('Exact independently inventoried resume state required')
    original=torch.load(a.resume_checkpoint,map_location='cpu',weights_only=False);validate(original,digest)
    state,changes=adam_format(original,torch)
    corpus=Corpus(ART/'cache-v52',action_index=ART/'cache-v52/action-index.json',admission='docs/NATIVE-VENUS-CONTEXT-TRAINING-ADMISSION-V3-2026-11-02.json')
    model,_=load_native(a.model,language_only=True);llm=model.llm;llm.eval();llm.gradient_checkpointing_disable()
    folder=Path(a.root)/'resume-proof'/a.stage;folder.mkdir(parents=True,exist_ok=False)
    runtime=restore_expanded(state,llm,a.opal_source,folder)
    try:
        assert_optimizer_equal(state['optimizer'],runtime.optimizer.state_dict())
        if not exact_state(state['rng'],rng_state()):raise ValueError('Saved8 RNG differs')
        row=next(r for r in sorted(corpus.rows,key=lambda r:r['id']) if r['fold']=='TRAIN' and r['family']=='ordinary')
        with torch.no_grad():logits,_=selected_forward(llm,corpus.read(row),'cuda')
        restore_rng(state['rng'])
        result=dict(source=a.source,original_training_source=SAVED_SOURCE,checkpoint_sha256=digest,saved_local_update=8,global_update=534,execution_id=os.environ['DUPLEX_NATIVE_EXECUTION_ID'],container_pid=os.getpid(),logits_sha256=tensor_sha(logits),Adam_RNG_OPAL_exact=True,actual_CUDA=True,new_optimizer_updates=0,format_changes=changes)
        if a.stage=='restore':
            ref=json.loads((folder.parent/'reference.json').read_text())
            if ref['execution_id']==result['execution_id'] or any(ref[k]!=result[k] for k in ('source','checkpoint_sha256','saved_local_update','logits_sha256')):raise ValueError('Distinct host-bound fresh restore differs')
            result.update(passed=True,reference_execution_id=ref['execution_id'])
        write(folder.parent,a.stage+'.json',result);print(json.dumps(result),flush=True)
    finally:runtime.close()

if __name__=='__main__':
    from .large_behavior import parser
    p=parser();p.add_argument('--stage',required=True,choices=('reference','restore'));p.add_argument('--opal-source',required=True)
    p.add_argument('--resume-checkpoint',required=True);p.add_argument('--resume-sha',required=True);p.add_argument('--resume-step',type=int,required=True);p.add_argument('--resume-prior-root');main(p.parse_args())
