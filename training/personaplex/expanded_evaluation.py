"""Zero-update expanded SFT evaluation: matched RP speech and native answers.

Four disjoint workers: baseline/control32, candidate/control32, baseline/DEV64,
candidate/DEV64. No answer/teacher audio or scripted result enters the model.
Native RET is measured as routing, not claimed to be an executed backend task.
"""
import argparse,json,math,os,subprocess,sys
from pathlib import Path
from .expanded_guard import check
from .expanded_capture import INSTRUCTIONS,REFERENCE_SHA,PP,VENDOR,WEIGHTS
from evals.moshi_model_download import sha256 as sha


def completed(root,trainer_source,provider,backup_root):
    root=Path(root)
    if provider.get('JobState')!='COMPLETED' or provider.get('Account')!='brics.u6ada':raise ValueError('Successful terminal provider required; no overlapping evaluation')
    folder=root/('job-'+provider['JobId']);host=json.loads((folder/'host-terminal.json').read_text());terminal=host['training']
    if (host.get('success')is not True or terminal.get('training_complete')is not True or terminal.get('actual_new_step')!=196
            or terminal.get('continuation_ready')is not False or terminal.get('source')!=trainer_source):raise ValueError('All196 actual saved updates and successful host required')
    path=Path(terminal['checkpoint']);receipt=json.loads((folder/'verify'/'verify-terminal.json').read_text())
    ranks=json.loads((folder/'verify'/'restore-all-ranks.json').read_text())
    if (receipt.get('passed')is not True or receipt.get('step')!=196 or receipt.get('new_updates')!=0
            or ranks.get('passed')is not True or len(ranks['rows'])!=4):raise ValueError('Complete final independent native restoration required')
    for row in ranks['rows']:
        if (row.get('source')!=trainer_source or row.get('step')!=196 or row.get('actual_CUDA')is not True
                or row.get('native_fresh_restore_passed')is not True or row.get('native_data_qualified')is not True
                or row.get('all_Adam_RNG_OPAL_native_fields_exact')is not True or row['execution']==row['reference_execution']):raise ValueError('Every exact fresh native rank proof required')
    if len({row['pid']for row in ranks['rows']})!=4:raise ValueError('Four actual restore processes required')
    if path.resolve().parent!=(root/'checkpoints').resolve()or sha(path)!=terminal['checkpoint_sha256']:raise ValueError('Exact final checkpoint bytes required')
    from .adaptive_evidence import backup
    from .matched_archive import independent_filesystems
    if not independent_filesystems(root,Path(backup_root)):raise ValueError('Original checkpoint backup must be on an independent filesystem')
    backed_step,backed_hash,receipt_hash=backup(Path(backup_root)/'step-0196'/'backup.json')
    if backed_step!=196 or backed_hash!=terminal['checkpoint_sha256']:raise ValueError('Independent full final backup required')
    return dict(backup_receipt_sha256=receipt_hash,checkpoint=str(path),checkpoint_sha256=terminal['checkpoint_sha256'],trainer_source=trainer_source,step=196,optimizer_updates=0)


def named_adapter(state,selection):
    """Return a new inference view; retain original full optimizer/RNG files."""
    identity=state.get('identity',{});training=state.get('training',{})
    if (state.get('schema')!='dl-distributed-native-state-v2' or state.get('step')!=196 or state.get('world_size')!=4
            or identity.get('source')!=selection['trainer_source']or identity.get('base')!='personaplex'
            or identity.get('method')!='broad-lora64'or identity.get('trainable_parameters')!=105070592
            or training.get('method')!='broad-lora64'or identity.get('old_saved_updates_replayed')!=0
            or identity.get('recipe',{}).get('new_TRAIN')!=1008 or identity.get('recipe',{}).get('new_DEV')!=64):raise ValueError('Exact complete expanded native identity required')
    learned=training['model']
    if not learned or any(not name.startswith('lm.')for name in learned):raise ValueError('Complete declared DDP native namespace required')
    result={name[3:]:value for name,value in learned.items()}
    if len(result)!=len(learned)or set(result)!=set(training.get('ownership',{}).get('trainable_names',[])):
        raise ValueError('Complete exact declared adapter membership required')
    if sum(v.numel()for v in result.values())!=105070592:raise ValueError('All105070592 learned coefficients required')
    return result


def make_engine(a,selection,*,readout_state=None,zero_readout=False):
    import torch
    from .foundation_context import raw_model
    from .matched_generation import construct_engine
    from .distributed_native import configure_lm
    from .matched_state import load_adapter
    from evals.context_spanning_probe import audited_loads
    from contextspan.model import engine as module,weights
    state=None
    if zero_readout and readout_state is not None:raise ValueError('Zero proof and learned state are separate conditions')
    if (readout_state is not None or zero_readout)and a.capacity!='expanded105M':raise ValueError('Readout requires its exact expanded predecessor, never the raw base')
    if a.capacity!='baseline':
        if sha(selection['checkpoint'])!=selection['checkpoint_sha256']:raise ValueError('Final adapter identity changed')
        state=named_adapter(torch.load(selection['checkpoint'],map_location='cpu',weights_only=False),selection)
    def loader(checkpoint=None,device='cuda',cpu_offload=False):
        if cpu_offload:raise ValueError('No evaluation-only offload confound')
        lm=raw_model(weights,device)
        if state is not None:
            ownership=configure_lm(lm,'broad-lora64')
            if ownership['trainable_parameters']!=105070592:raise ValueError('Full expanded adapter ownership required')
            load_adapter(lm,state)
        if readout_state is not None:
            from .action_inference import identity,install_fit
            install_fit(lm,readout_state,identity(selection))
        elif zero_readout:
            from .action_inference import identity,install_zero
            install_zero(lm,identity(selection))
        return lm.requires_grad_(False).eval(),weights.load_mimi(device),weights.load_tokenizer()
    loads=[]
    with audited_loads(loads):engine=construct_engine(module,loader,checkpoint=str(WEIGHTS/'context_spanning_7b.pt'),device='cuda',temp=.8,temp_text=.7)
    own=[r for r in loads if r['module_identity']==id(engine.lm)]
    if not own or not own[-1]['strict']or own[-1]['missing']or own[-1]['unexpected']:raise ValueError('Exact published raw base load required')
    if engine.lm.dep_q!=16 or engine.frame_size!=1920 or engine.frame_rate!=12.5:raise ValueError('Unchanged full native geometry required')
    if state is not None:
        from .action_inference import adapter_parameters
        actual=adapter_parameters(engine.lm)
        if actual.keys()!=state.keys()or any(not torch.equal(v.detach().cpu(),state[n])for n,v in actual.items()):raise ValueError('Engine warmup changed learned adapter values')
    else:
        from evals.personaplex_adapters import adapter_parameters
        if adapter_parameters(engine.lm):raise ValueError('Baseline must remain untouched')
    from evals.context_native_layout import install
    install(engine);engine.evaluation_model_loads=own
    from evals.context_execution import snapshot
    engine.expanded_inference_proof=dict(actual_adapter_coefficients=0 if state is None else sum(v.numel()for v in state.values()),all_weights_frozen=not any(p.requires_grad for p in engine.lm.parameters()),execution=snapshot(engine.lm))
    if readout_state is not None:
        from .action_inference import identity
        engine.expanded_inference_proof['learned_action_readout']=dict(identity=identity(selection),mode=readout_state['metadata']['mode'],new_readout_updates=128,native_free_running_qualified=False)
    if zero_readout:
        from .action_inference import identity
        engine.expanded_inference_proof['learned_action_readout']=dict(identity=identity(selection),mode='zero-insertion-proof',new_readout_updates=0,native_free_running_qualified=False)
    return engine


def dev_cases(data,out):
    import numpy as np,soundfile as sf
    data=Path(data);manifest=json.loads((data/'manifest.json').read_text());all_rows=manifest['rows']
    if not manifest.get('complete1072')or manifest['TRAIN']!=1008 or manifest['DEV']!=64 or len(all_rows)!=1072 or len({r['id']for r in all_rows})!=1072:raise ValueError('Original full qualified source manifest required')
    rows=[r for r in all_rows if r['fold']=='DEV']
    if len(rows)!=64:raise ValueError('Whole exact64 DEV required')
    out=Path(out);out.mkdir(exist_ok=False);result=[]
    for row in rows:
        source=data/row['path']
        if not source.resolve().is_relative_to(data.resolve())or sha(source)!=row['wav_sha256']or sha(source.with_suffix('.json'))!=row['sidecar_sha256']:raise ValueError('Exact DEV waveform/sidecar required')
        pcm,rate=sf.read(source,dtype='float32',always_2d=True)
        users=[p for p in row['parents']if p['id']==row['id']+'-user']
        if rate!=24000 or pcm.shape[1]!=2 or len(users)!=1 or not np.isfinite(pcm).all():raise ValueError('Original user channel required')
        length=users[0]['resampled_frames']
        if type(length)is not int or not 0<length<=len(pcm):raise ValueError('Full original user audio length required')
        # Fixed60-second response window, independent of teacher answer length.
        user=np.concatenate([pcm[:length,1],np.zeros(60*rate,dtype=np.float32)])
        folder=out/row['id'];folder.mkdir();path=folder/'user.wav';sf.write(path,user,rate,subtype='FLOAT')
        result.append(dict(id=row['id'],family='expanded-DEV',folder=str(folder),input_rate=24000,frames=math.ceil(len(user)/rate),input_sha256=sha(path),source_fold='DEV',original_source_sha256=row['wav_sha256'],teacher_audio_in_model_input=False))
    return result


def complete_reports(root,source):
    root=Path(root);reports={};total=0;paired={}
    for cohort,count in (('control32',32),('dev64',64)):
        for capacity in ('baseline','expanded105M'):
            path=root/cohort/('personaplex-'+capacity)/'report.json';report=json.loads(path.read_text())
            if not report['complete']or report['source']!=source or len(report['cases'])!=count or report['optimizer_updates']!=0 or report['external_API_calls']!=0:
                raise ValueError('Every complete independent speech report required')
            ids={r['id']:r['input_sha256']for r in report['cases']}
            if len(ids)!=count:raise ValueError('Duplicate comparison output')
            if cohort in paired and paired[cohort]!=ids:raise ValueError('Candidate and baseline input identities differ')
            paired[cohort]=ids;total+=count;reports[cohort+'/'+capacity]=dict(path=str(path),sha256=sha(path),summary=report['summary'])
    if total!=192:raise ValueError('Complete192 native outputs required')
    result=dict(source=source,complete192=True,reports=reports,optimizer_updates=0,external_API_calls=0,
        same_RP_reference_and_persona=True,semantic_correctness_review_pending=True,naturalness_and_accent_listening_pending=True,
        connected_backend_tested=False,native_RET_is_not_backend_execution=True,adopted=False)
    with (root/'comparison.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    return result


def run(a):
    import torch
    from . import matched_evaluation as evaluation
    from evals.context_execution import math_attention
    lease=check(a.lease,600)
    if lease.get('evaluation_authorization_sha256')!=sha('docs/NATIVE-EXPANDED1072-EVALUATION-2026-11-16.json'):raise ValueError('Exact zero-update evaluation authority required')
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=a.source:raise ValueError('Exact committed evaluation worker source required')
    rank=int(os.environ['LOCAL_RANK'])
    if os.environ.get('WORLD_SIZE')!='4'or not 0<=rank<4 or torch.cuda.device_count()!=4:raise ValueError('All four allocated native evaluation ranks required')
    torch.cuda.set_device(rank)
    os.environ.update(CS_BASE_DIR=str(PP),CS_WEIGHTS_DIR=str(WEIGHTS),CS_VOICES_DIR=str(WEIGHTS/'voices'));sys.path.insert(0,str(VENDOR))
    math_attention();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    plan=json.loads(a.plan.read_text())
    if plan['source']!=a.source or plan['required_outputs']!=192 or plan['optimizer_updates']!=0:raise ValueError('Exact zero-update whole comparison plan required')
    selection=plan['selection'];a.trainer_source=selection['trainer_source'];a.base='personaplex';a.capacity='baseline'if rank<2 else'expanded105M'
    if not plan.get('public_assets_and_native_geometry_preflight_passed'):raise ValueError('Full public-asset CPU preflight required')
    name='control-cases.json'if rank%2==0 else'dev-cases.json'
    expected_hash=plan['control32_sha256']if rank%2==0 else plan['dev64_sha256']
    if sha(a.root/name)!=expected_hash:raise ValueError('Pinned input cohort changed')
    rows=json.loads((a.root/name).read_text())
    expected=32 if rank%2==0 else 64
    if len(rows)!=expected:raise ValueError('Complete declared cohort required')
    a.root=a.root/('control32'if rank%2==0 else'dev64');a.root.mkdir(exist_ok=True)
    capture=Path(plan['parent_root'])/'capture'/'shard-00';report=json.loads((capture/'capture.json').read_text());voice_path=capture/'voice.pt'
    if report['reference_source_sha256']!=REFERENCE_SHA or sha(voice_path)!=report['voice_sha256']or report['instructions']!=INSTRUCTIONS:raise ValueError('Same approved RP voice/persona as training required')
    voice=torch.load(voice_path,map_location='cpu',weights_only=True)['codes']
    evaluation.collect(a,rows,selection,factory=make_engine,persona=INSTRUCTIONS,voice_codes=voice,voice_sha256=report['voice_sha256'],profile_extra=dict(same_training_RP_reference=REFERENCE_SHA,complete_cohort=expected,teacher_forcing=False,comparison='untouched-base versus expanded196; no cross-model adapter transfer'),guard=check)
    from training.venus.voice_compare import recognizer
    evaluation.audit(a,rows,required=expected,guard=check,require_spoken_content=True,recognizer_factory=lambda:recognizer(device=rank))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','plan','lease'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source',required=True);run(p.parse_args())
