"""Actual native zero-update feature collection and different-process restoration."""
import argparse,gc,json,os,sys,uuid
from pathlib import Path
import torch
from .expanded_guard import FEATURE_AUTH,RESTORE_AUTH,sha
from .native_process import process_identity,validate_restore
from .expanded_capture import PP,VENDOR,WEIGHTS,REFERENCE_SHA,INSTRUCTIONS
from .expanded_runner import OLD_SHA,OLD_REPORT_SHA
from .action_inference import identity
from .action_features import collect
from .native_action_readout import RetReadout
from .action_fit import RECIPE

BASE=Path('/projects/u6ada/duplex-live')
OLD=BASE/'native-moshi-matched-inputs-v1'


def write(path,value):
    with Path(path).open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def prepare_probes(source):
    """Full independently qualified TRAIN user only; no teacher assistant channel."""
    import numpy as np,soundfile as sf
    source=Path(source);manifest=json.loads((source/'manifest.json').read_text())
    if not manifest.get('complete1072')or manifest['TRAIN']!=1008 or manifest['DEV']!=64:raise ValueError('Exact complete admitted source required')
    chosen=sorted((r for r in manifest['rows']if r['fold']=='TRAIN'),key=lambda r:r['id'])[:4];result=[]
    if len(chosen)!=4:raise ValueError('Four distinct TRAIN-only identity probes required')
    for row in chosen:
        path=source/row['path']
        if not path.resolve().is_relative_to(source.resolve())or sha(path)!=row['wav_sha256']or sha(path.with_suffix('.json'))!=row['sidecar_sha256']:raise ValueError('Exact TRAIN wave/sidecar required')
        pcm,rate=sf.read(path,dtype='float32',always_2d=True);users=[p for p in row['parents']if p['id']==row['id']+'-user']
        if rate!=24000 or pcm.shape[1]!=2 or len(users)!=1:raise ValueError('Whole original user-only channel required')
        n=users[0]['resampled_frames']
        if type(n)is not int or not 0<n<=len(pcm)or not np.isfinite(pcm).all():raise ValueError('Finite full TRAIN user recording required')
        user=np.concatenate((pcm[:n,1],np.zeros(8*rate,dtype=np.float32)));user=np.pad(user,(0,(-len(user))%1920))
        result.append(dict(id=row['id'],audio=user,source_sha256=row['wav_sha256'],user_frames=n,proof_only_tail_seconds=8,teacher_audio_provided=False))
    return result


def streaming(engine,probes,voice,out,guard):
    import numpy as np,soundfile as sf
    from contextspan.model.engine import seed_all
    from training.venus.foundation_native import timed_assembly
    out.mkdir(exist_ok=False);rows=[]
    with torch.inference_mode():
        for row in probes:
            guard(180);seed_all(42);engine.reset();engine.set_persona(INSTRUCTIONS,voice)
            trace=[];arrays=[]
            for i,start in enumerate(range(0,len(row['audio']),1920)):
                guard();v=engine.step(row['audio'][start:start+1920]);pcm=np.asarray(v['agent_pcm']if v['agent_pcm']is not None else[],dtype=np.float32)
                if pcm.ndim!=1 or not np.isfinite(pcm).all():raise ValueError('Finite raw native proof output required')
                arrays.append(pcm);trace.append(dict(sequence=i,text_token=v['text_token'],is_ret=v['is_ret'],audio_samples=len(pcm),pcm_sha256=__import__('hashlib').sha256(pcm.tobytes()).hexdigest()))
            folder=out/row['id'];folder.mkdir();sf.write(folder/'user.wav',row['audio'],24000,subtype='FLOAT');sf.write(folder/'agent.wav',timed_assembly(trace,arrays,24000,frame_seconds=.08),24000,subtype='FLOAT');write(folder/'trace.json',trace)
            rows.append(dict(id=row['id'],source_sha256=row['source_sha256'],frames=len(trace),trace=trace))
    write(out/'proof.json',dict(rows=rows,actual_CUDA=True,seed=42,paced_or_live_latency=False,optimizer_updates=0));return rows


def zero_hooks(lm):
    if not isinstance(lm.text_linear,RetReadout):raise ValueError('Actual native zero readout required')
    previous=[];proof=dict(calls=0,positions=0,all_vocabulary_logits_exact=True)
    def original(module,args,value):previous[:]=[value]
    def after(module,args,value):
        if len(previous)!=1 or not torch.equal(previous[0].contiguous().view(torch.uint8),value.contiguous().view(torch.uint8)):raise ValueError('Zero insertion changed actual native logits')
        proof['calls']+=1;proof['positions']+=value.numel()//value.shape[-1];previous.clear()
    return [lm.text_linear.original.register_forward_hook(original),lm.text_linear.register_forward_hook(after)],proof


def new_opal(head,out):
    from .matched_state import materialize_adam
    from training.venus.native_optimizer import NativeOpal
    head.delta.requires_grad_(True)
    base=torch.optim.AdamW(head.delta.parameters(),lr=RECIPE['learning_rate'],betas=RECIPE['adam_betas'],eps=RECIPE['adam_eps'],weight_decay=0.,foreach=False,fused=False,amsgrad=False)
    materialize_adam(base)
    return NativeOpal(base,head.delta,source=BASE/'opal-upstream-v1',run_dir=out,recipe=RECIPE,backend='keep',goal='Zero-update native state restoration only; no learning authorized')


def main(a):
    from .inference_guard import FrameGuard
    from .foundation_context import preflight
    from .matched_native import corpus
    from .expanded_evaluation import make_engine
    from evals.context_execution import math_attention
    from training.venus.state import rng_state,restore_rng,exact_state
    from training.venus.native_optimizer import assert_optimizer_equal
    from training.venus.opal_adapter import checkpoint_manifest
    from .distributed_native import cpu_state
    guard=FrameGuard(a.lease);lease=guard(600)
    recovery=lease.get('restore_authorization_sha256')==sha(RESTORE_AUTH)
    if lease['source']!=a.source or not(recovery or lease.get('feature_authorization_sha256')==sha(FEATURE_AUTH))or torch.cuda.device_count()!=1:raise ValueError('Exact source and separate one-GPU zero-update scope required')
    torch.cuda.set_device(0);os.environ.update(CS_BASE_DIR=str(PP),CS_WEIGHTS_DIR=str(WEIGHTS),CS_VOICES_DIR=str(WEIGHTS/'voices'));sys.path.insert(0,str(VENDOR));math_attention();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    if recovery and(a.stage!='verify'or a.cache_root is None or a.cache_source!=lease['restore_parent']['cache_source']or str(a.cache_root)!=lease['restore_parent']['cache_root']):raise ValueError('Restore-only recovery must reuse its exact backed cache')
    if a.stage=='cache'and(a.cache_root is not None or a.cache_source is not None):raise ValueError('Collection cannot import an old cache')
    cached=a.cache_root or a.root/'cache';cache_source=a.cache_source or a.source
    selection=json.loads(a.plan.read_text())['selection'];bound=identity(selection)
    if selection['checkpoint_sha256']!='96cff747f42fff1c36572800debcee2d435f20efaebf2c8ccfe9720e9bec3c87':raise ValueError('Exact preserved196 required')
    if len(preflight(Path.cwd()))!=32:raise ValueError('Full original public/runtime preflight required')
    groups,report=corpus(OLD,OLD_REPORT_SHA)
    if report['input_sha256']!=OLD_SHA:raise ValueError('Original TRAIN tensor identity changed')
    capture=Path(json.loads(a.plan.read_text())['parent_root'])/'capture/shard-00';record=json.loads((capture/'capture.json').read_text())
    if record['reference_source_sha256']!=REFERENCE_SHA or sha(capture/'voice.pt')!=record['voice_sha256']:raise ValueError('Exact trained RP reference required')
    voice=torch.load(capture/'voice.pt',map_location='cpu',weights_only=True)['codes'];probes=prepare_probes(a.data)
    a.capacity='expanded105M';a.trainer_source=selection['trainer_source'];folder=a.root/a.stage;folder.mkdir(exist_ok=False);execution=uuid.uuid4().hex
    if a.stage=='cache':
        engine=make_engine(a,selection);reference=streaming(engine,probes,voice,folder/'baseline-stream',guard);engine.lm_gen._stop_streaming();del engine;gc.collect();torch.cuda.empty_cache()
    else:
        reference=json.loads((cached/'zero-stream/proof.json').read_text())['rows']
    engine=make_engine(a,selection,zero_readout=True)
    if a.stage=='verify':
        packet=json.loads((cached/'snapshot.json').read_text())
        if sha(cached/'zero-state.pt')!=packet['snapshot_sha256']or sha(cached/'features.pt')!=packet['features_sha256']:raise ValueError('Immutable saved state and feature hashes required')
        saved=torch.load(cached/'zero-state.pt',map_location='cpu',weights_only=False)
        validate_restore(saved,bound,cache_source,execution,lease,process_identity())
        engine.lm.text_linear.restore(saved['readout']);opal=new_opal(engine.lm.text_linear,folder/'opal');opal.optimizer.load_state_dict(saved['optimizer']);opal.restore_diagnostics(saved['diagnostics']);restore_rng(saved['rng'])
        if not exact_state(saved['rng'],rng_state()):raise ValueError('All native RNG fields differ')
        assert_optimizer_equal(saved['optimizer'],opal.optimizer.state_dict())
        if checkpoint_manifest(opal.optimizer)['update']!=0:raise ValueError('No native optimizer updates allowed')
        opal.close();engine.lm.requires_grad_(False)
    actual=streaming(engine,probes,voice,folder/'zero-stream',guard)
    if actual!=reference:raise ValueError('Actual native streaming zero/restored output differs; preserve both traces')
    engine.lm_gen._stop_streaming();hooks,zero=zero_hooks(engine.lm)
    try:features=collect(engine.lm,groups,guard,progress=lambda value:print(json.dumps(dict(event='native_TRAIN_features',stage=a.stage,**value)),flush=True))
    finally:
        for hook in hooks:hook.remove()
    if features['population']['positive']!=89 or not zero['calls']:raise ValueError('All89 original TRAIN positives and actual zero-logit calls required')
    if a.stage=='cache':
        features.update(identity=bound,source=a.source,original_TRAIN_sha256=OLD_SHA);torch.save(features,folder/'features.pt')
        opal=new_opal(engine.lm.text_linear,folder/'opal')
        try:
            state=dict(schema='dl-native-zero-ret-restore-v1',identity=bound,source=a.source,step=0,writer_pid=os.getpid(),execution=execution,process_identity=process_identity(),readout=engine.lm.text_linear.learned_state(),optimizer=cpu_state(opal.optimizer.state_dict()),diagnostics=opal.diagnostics_state,rng=rng_state(),recipe=RECIPE)
            if checkpoint_manifest(opal.optimizer)['update']!=0 or any(torch.count_nonzero(v)for v in state['readout']['state'].values()):raise ValueError('Only true zero-initialized/zero-update state may be saved')
            torch.save(state,folder/'zero-state.pt')
        finally:opal.close()
        write(folder/'snapshot.json',dict(snapshot_sha256=sha(folder/'zero-state.pt'),features_sha256=sha(folder/'features.pt'),source=a.source,step=0,identity=bound))
    else:
        prior=torch.load(cached/'features.pt',weights_only=False,map_location='cpu')
        if not exact_state(prior['population'],features['population'])or not exact_state(prior['rows'],features['rows']):raise ValueError('Complete actual native restored TRAIN features differ')
    write(folder/'terminal.json',dict(passed=True,stage=a.stage,source=a.source,actual_CUDA=True,pid=os.getpid(),execution=execution,process_identity=process_identity(),cache_producer=cache_source,identity=bound,TRAIN_records=len(features['rows']),positive_positions=features['population']['positive'],negative_positions=features['population']['negative'],zero_insertion=zero,streaming_probes=4,all_streamed_tokens_and_PCM_exact=True,all_features_exact=a.stage=='verify',all_Adam_RNG_OPAL_restored_exact=a.stage=='verify',optimizer_updates=0,old_saved_updates_replayed=0,native_policy_acceptance=False))
    print(json.dumps(dict(event='native_feature_stage_complete',stage=a.stage,optimizer_updates=0,pid=os.getpid())),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','lease','plan','data'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cache-root',type=Path);p.add_argument('--cache-source');p.add_argument('--source',required=True);p.add_argument('stage',choices=('cache','verify'));main(p.parse_args())
