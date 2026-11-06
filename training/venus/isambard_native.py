"""Same saved candidate and full frozen native cohort on the Slurm runtime.

Legacy proofs keep their original source and bytes. New GH200 reference and
fresh-process restore are separate evidence; cross-hardware bit identity is not
pretended. No optimizer step is possible in this entrypoint.
"""
import argparse,asyncio,copy,gc,json,os,subprocess,time
from pathlib import Path

BASE=Path('/projects/u6ada/duplex-live')
ART=BASE/'artifacts/native-v75'
SHA={'main':'a89a39b7dd1e2a08f06ee2776d92adcd41e285245d6a91fd1d36d85297a42e81','control':'df75747e576c0c72bfede122a4f9e39e8f1fa26964538523a91e59f58bbcc991'}
original_training_source='b39ce3ebe167b059b67fa3d131978e5da7c41ac1'

def guarded(lease,needed=0):
    from .large_native import guarded as check
    check(lease,needed)

def admit(a,require_restore=True):
    from .large_inputs import sha
    from .isambard_guard import resource
    from portable.venus.native_app import ensure_guard
    lease=resource(json.loads(Path(a.lease).read_text()));ensure_guard()
    if a.source!=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() or lease['source']!=a.source:raise ValueError('Actual Git execution producer differs')
    if a.arm:
        checkpoint=ART/'checkpoints'/(a.arm+'-step-0060.pt')
        if sha(checkpoint)!=SHA[a.arm]:raise ValueError('Exact saved60 checkpoint required; no repeated updates')
        if not require_restore:return lease
        proof=Path(a.root)/'migration'/a.arm/'restore.json'
        if not proof.exists():raise ValueError('New independent GH200 restore required before native collection')
        p=json.loads(proof.read_text())
        if not p['passed'] or p['source']!=a.source or p['checkpoint_sha256']!=SHA[a.arm] or not p['actual_CUDA'] or p['new_optimizer_updates']!=0 or p['pid']==p['reference_pid'] or not all(p[k] for k in ('RNG_exact','OPAL_state_exact','all_original_optimizer_fields_exact')):raise ValueError('Fresh runtime proof differs')
    return lease

def adam_format(state,torch):
    """Explicitly derive only a new Torch AdamW format default, never moments."""
    from .state import exact_state
    import io,hashlib
    p=torch.nn.Parameter(torch.ones(1));probe=torch.optim.AdamW([p]);default=probe.param_groups[0].get('decoupled_weight_decay');changes=[]
    scheduler_default=torch.optim.lr_scheduler.LambdaLR(probe,lambda _:1.).state_dict().get('_is_initial')
    def walk(value,path):
        if isinstance(value,dict):
            result={k:walk(v,path+'/'+str(k)) for k,v in value.items()}
            if 'param_groups' in result and default is not None:
                for i,group in enumerate(result['param_groups']):
                    if not {'lr','betas','eps','weight_decay','params'}<=set(group):raise ValueError('Unknown optimizer group in format derivation')
                    if 'decoupled_weight_decay' not in group:
                        if default is not True:raise ValueError('Unknown AdamW default')
                        group['decoupled_weight_decay']=True;changes.append(path+'/param_groups/'+str(i)+'/decoupled_weight_decay=true')
            if {'base_lrs','last_epoch','_step_count','_last_lr','lr_lambdas'}<=set(result) and '_is_initial' not in result and scheduler_default is not None:
                if scheduler_default is not False:raise ValueError('Unknown scheduler initialization default')
                result['_is_initial']=False;changes.append(path+'/_is_initial=false')
            return result
        if isinstance(value,list):return [walk(v,path+'/'+str(i)) for i,v in enumerate(value)]
        if isinstance(value,tuple):return tuple(walk(v,path+'/'+str(i)) for i,v in enumerate(value))
        return value
    mapped=dict(state);mapped['optimizer']=walk(state['optimizer'],'optimizer')
    # All original leaves, including every Adam tensor/counter, remain identical.
    def original_equal(old,new):
        if isinstance(old,dict):return all(k in new and original_equal(v,new[k]) for k,v in old.items())
        if isinstance(old,(list,tuple)):return len(old)==len(new) and all(original_equal(a,b) for a,b in zip(old,new))
        return exact_state(old,new)
    if not original_equal(state['optimizer'],mapped['optimizer']):raise ValueError('Original optimizer field changed during format derivation')
    if isinstance(state['optimizer'].get('memory'),bytes):
        original_memory=torch.load(io.BytesIO(state['optimizer']['memory']),weights_only=False)
        encoded=io.BytesIO();torch.save(original_memory,encoded);new_memory=encoded.getvalue()
        if not exact_state(original_memory,torch.load(io.BytesIO(new_memory),weights_only=False)):raise ValueError('Original OPAL memory values changed during serialization migration')
        mapped['optimizer']['memory']=new_memory
        if new_memory!=state['optimizer']['memory']:
            changes.append('optimizer/memory: exact decoded values; serialization '+hashlib.sha256(state['optimizer']['memory']).hexdigest()+' -> '+hashlib.sha256(new_memory).hexdigest())
    return mapped,changes

def schema_selftest(a):
    import torch
    from .state import exact_state
    from .large_native import write
    guarded(a.lease,180)
    p=torch.nn.Parameter(torch.ones(1));optimizer=torch.optim.AdamW([p]);p.grad=torch.ones_like(p);optimizer.step()
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    legacy=optimizer.state_dict();legacy['param_groups'][0].pop('decoupled_weight_decay',None)
    legacy_scheduler=scheduler.state_dict();legacy_scheduler.pop('_is_initial',None)
    old=dict(optimizer=dict(user_optimizers=[legacy],user_schedulers=[legacy_scheduler],updater=dict(opts=[legacy],scheds=[legacy_scheduler])))
    derived,changes=adam_format(old,torch)
    other=torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))]);other.load_state_dict(derived['optimizer']['user_optimizers'][0])
    if not exact_state(other.state_dict(),derived['optimizer']['user_optimizers'][0]):raise ValueError('Actual Torch optimizer format restore differs')
    other_scheduler=torch.optim.lr_scheduler.LambdaLR(other,lambda _:1.);other_scheduler.load_state_dict(derived['optimizer']['user_schedulers'][0])
    if not exact_state(other_scheduler.state_dict(),derived['optimizer']['user_schedulers'][0]):raise ValueError('Actual Torch scheduler format restore differs')
    out=Path(a.root)/'migration';out.mkdir(parents=True,exist_ok=True)
    write(out,'schema-selftest.json',dict(passed=True,torch=torch.__version__,declared_format_defaults=changes,fixture_optimizer_updates=1,native_model_updates=0,source=a.source))

def restore(a):
    import torch
    from .native_cache import load_native
    from .large_inputs import Corpus,sha
    from .large_native import restore_expanded,tensor_sha,write
    from .checkpoint_restore import restore as restore_control
    from .lora import install,adapter_parameters,selected_forward
    from .state import restore_rng,rng_state,exact_state
    from .native_optimizer import assert_optimizer_equal
    admit(a,require_restore=False)
    guarded(a.lease,600)
    path=ART/'checkpoints'/(a.arm+'-step-0060.pt')
    if sha(path)!=SHA[a.arm]:raise ValueError('Saved checkpoint hash differs')
    original=torch.load(path,map_location='cpu',weights_only=False)
    if original['step']!=60 or original['metadata']['source']!=original_training_source:raise ValueError('Preserve original trained source/step')
    corpus=Corpus(ART/'cache-v52',action_index=ART/'cache-v52/action-index.json',admission='docs/NATIVE-VENUS-CONTEXT-TRAINING-ADMISSION-V3-2026-11-02.json')
    model,_=load_native(a.model,language_only=True);llm=model.llm;llm.eval();llm.gradient_checkpointing_disable()
    state,changes=adam_format(original,torch)
    folder=Path(a.root)/'migration'/a.arm/a.stage;folder.mkdir(parents=True,exist_ok=False)
    if a.arm=='main':runtime=restore_expanded(state,llm,a.opal_source,folder)
    else:
        # Same full restore implementation with explicit derived format defaults.
        from .large_phase import write_programs
        from .native_optimizer import NativeOpal
        install(llm,rank=64,alpha=64,last_layers=1)
        from .lora import load_adapter
        load_adapter(llm,state['adapter'])
        base=torch.optim.AdamW(list(adapter_parameters(llm).values()),lr=state['metadata']['recipe']['learning_rate'])
        programs=write_programs(folder/'programs',state['optimizer']['files'])
        runtime=NativeOpal(base,llm,source=a.opal_source,run_dir=folder/'opal',recipe=state['metadata']['recipe'],initial_program=programs)
        runtime.optimizer.load_state_dict(state['optimizer']);runtime.restore_diagnostics(state['training_diagnostics']);restore_rng(state['rng'])
    try:
        assert_optimizer_equal(state['optimizer'],runtime.optimizer.state_dict())
        from .lora import adapter_state
        if not exact_state(state['adapter'],adapter_state(llm)):raise ValueError('Restored adapter bytes differ')
        if not exact_state(state['rng'],rng_state()):raise ValueError('Restored RNG differs')
        row=next(r for r in sorted(corpus.rows,key=lambda r:r['id']) if r['fold']=='TRAIN' and r['family']=='ordinary')
        with torch.no_grad():logits,_=selected_forward(llm,corpus.read(row,full=a.arm=='main'),'cuda')
        digest=tensor_sha(logits);restore_rng(state['rng'])
        result=dict(source=a.source,original_training_source=original_training_source,checkpoint_sha256=SHA[a.arm],pid=os.getpid(),actual_CUDA=True,new_optimizer_updates=0,logits_sha256=digest,all_original_optimizer_fields_exact=True,derived_Adam_format_defaults=changes,RNG_exact=exact_state(state['rng'],rng_state()),OPAL_state_exact=True,original_checkpoint_unchanged=True)
        if a.stage=='restore':
            reference=json.loads((Path(a.root)/'migration'/a.arm/'reference.json').read_text())
            if reference['pid']==os.getpid() or any(result[k]!=reference[k] for k in ('source','checkpoint_sha256','logits_sha256','derived_Adam_format_defaults')):raise ValueError('Distinct fresh-process native restore differs')
            result.update(passed=True,reference_pid=reference['pid'])
        write(Path(a.root)/'migration'/a.arm,a.stage+'.json',result)
        print(json.dumps(result),flush=True)
    finally:runtime.close()

def main():
    from .large_behavior import parser,collect,combine
    p=parser();p.add_argument('--stage',required=True,choices=('schema-selftest','reference','restore','collect','combine'));p.add_argument('--opal-source',required=True);a=p.parse_args()
    if a.stage=='schema-selftest':schema_selftest(a)
    elif a.stage in ('reference','restore'):restore(a)
    elif a.stage=='collect':
        import sys
        asyncio.run(collect(a,execution_context=sys.modules[__name__]))
    else:admit(a);combine(a)

if __name__=='__main__':main()
