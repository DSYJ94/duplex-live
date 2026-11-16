"""Frozen native audio/context cause separation; no optimizer or answer input.

The text modality is the model's supported persona/context prefix, NOT a text-
chat interface. Audio-only results are reused, never regenerated. A narrow
explicit local tool runs only after actualRET; forced contract tests are separate.
"""
import argparse,asyncio,json,os,math,time,subprocess
from pathlib import Path
from evals.moshi_model_download import sha256 as sha
from .expanded_capture import INSTRUCTIONS,REFERENCE_SHA


def prepare(parent,root,source):
    parent=Path(parent);root=Path(root)
    if root.exists():raise ValueError('New immutable probe namespace required')
    comparison=json.loads((parent/'comparison.json').read_text())
    if not comparison['complete192']or comparison['optimizer_updates']!=0:raise ValueError('Reuse complete original audio-only comparison')
    control=json.loads((parent/'control-cases.json').read_text());dev=json.loads((parent/'dev-cases.json').read_text())
    refs={r['id']:r for r in json.loads((parent/'scoring-only-reference.json').read_text())}
    questions=[]
    for case in control:
        if case['family']!='ordinary':continue
        if len(case['user_turns'])!=1:raise ValueError('Single complete ordinary input required')
        questions.append(dict(id=case['id'],kind='basic',question=case['user_turns'][0]['text'],audio=str(Path(case['folder'])/'user.wav'),rate=16000,audio_sha256=case['input_sha256']))
    for case in dev:
        questions.append(dict(id=case['id'],kind='quantitative-DEV',question=refs[case['id']]['question'],audio=str(Path(case['folder'])/'user.wav'),rate=24000,audio_sha256=case['input_sha256']))
    if len(questions)!=76 or len({r['id']for r in questions})!=76:raise ValueError('Complete12basic+64DEV required')
    if any(set(r)!={'id','kind','question','audio','rate','audio_sha256'}or not r['question'].strip()for r in questions):raise ValueError('Only actual user inputs, no answer/route/score fields')
    root.mkdir();(root/'questions.json').write_text(json.dumps(questions,indent=2)+'\n')
    result=dict(source=source,parent_comparison=str(parent),questions=76,required_new_outputs=304,questions_sha256=sha(root/'questions.json'),old_audio_only_reused=True,old_audio_comparison_sha256=sha(parent/'comparison.json'),optimizer_updates=0,external_API_calls=0,text_interface='supplied question in supported persona/context prefix; not text-chat equivalence')
    (root/'preparation.json').write_text(json.dumps(result,indent=2)+'\n');return result


def supplied_context(question):
    return lambda seconds,assistant:[dict(role='user',text=question,source='Actual supplied user text in persona/context prefix; available before audio playback')]


def ret_summary(logits):
    import torch
    z=logits.detach().float().reshape(-1,logits.shape[-1]);p=(z[:,4]-z.logsumexp(-1)).exp();rank=(z>z[:,4,None]).sum(-1)+1
    return dict(positions=len(p),RET_probability_sum=float(p.sum()),RET_probability_max=float(p.max()),RET_top1=int((rank==1).sum()),RET_top25=int((rank<=25).sum()))


def teacher_ret(engine,out,guard):
    """New measurements of existing originalTRAIN positives, not training/replay."""
    import torch
    from .matched_losses import RET_GROUPS
    groups=torch.load('/projects/u6ada/duplex-live/native-moshi-matched-inputs-v1/inputs.pt',weights_only=True,map_location='cpu')
    engine.lm_gen._stop_streaming();lm=engine.lm;records=[]
    with torch.inference_mode():
        for group in sorted(RET_GROUPS):
            for row in groups[group]:
                positive=row['mask']&(row['targets']==4)
                if not positive.any():continue
                guard(180);x=row['inputs'].cuda();values=[]
                if group in ('stream49','control40','reasoning32'):
                    with lm.streaming(1):
                        for call in row['calls']:
                            guard(180);start,n=call['start'],call['frames'];_,z=lm.forward_codes(x[:,:,start:start+n]);m=positive[start:start+n]
                            if m.any():values.append(z[0,0,m].detach())
                else:
                    _,z=lm.forward_codes(x);values.append(z[0,0,positive].detach())
                score=ret_summary(torch.cat(values));records.append(dict(group=group,id=row['id'],**score));del x,values,z
    if sum(r['positions']for r in records)!=89:raise ValueError('All89originalTRAIN positiveRET positions required')
    result=dict(source_training_corpus='unchanged retained323',positions=89,records=records,teacher_forced_not_behavior=True,optimizer_updates=0)
    (out/'teacher-RET.json').write_text(json.dumps(result,indent=2)+'\n');engine.reset()


async def case(engine,row,condition,out,guard,voice):
    import numpy as np,soundfile as sf
    from scipy.signal import resample_poly
    from contextspan.model.engine import seed_all
    from portable.context.connected_inference import NativeConnection
    from portable.context.local_math_worker import run as worker
    from training.venus.foundation_native import timed_assembly
    if sha(row['audio'])!=row['audio_sha256']:raise ValueError('Original whole input audio changed')
    audio,rate=sf.read(row['audio'],dtype='float32')
    if rate!=row['rate']or audio.ndim!=1:raise ValueError('Original mono user audio required')
    if rate==16000:audio=resample_poly(audio,3,2).astype(np.float32)
    audio=np.pad(audio,(0,(-len(audio))%1920))
    if condition=='prefix-text-only':audio=np.zeros_like(audio)
    guard(180+math.ceil(len(audio)/24000*2));seed_all(42);engine.reset()
    prompt=INSTRUCTIONS+' The user has supplied this written question: '+row['question']+' Answer that question.'
    engine.set_persona(prompt,voice)
    forbidden=lambda *args:(_ for _ in ()).throw(RuntimeError('No fabricated ASR; this probe has actual supplied text'))
    connection=NativeConnection(engine,forbidden,worker,maximum_seconds=180,context_provider=supplied_context(row['question']))
    folder=out/row['id'];folder.mkdir(exist_ok=False);trace=[];arrays=[];tokens=[];events=[];start=time.monotonic()
    try:
        for i,offset in enumerate(range(0,len(audio),1920)):
            guard();before=time.monotonic();value,emitted=await connection.step(audio[offset:offset+1920]);elapsed=time.monotonic()-before
            pcm=np.asarray(value['agent_pcm']if value['agent_pcm']is not None else [],dtype=np.float32)
            if pcm.ndim!=1 or not np.isfinite(pcm).all():raise ValueError('Finite unmodified native output required')
            arrays.append(pcm);tokens.append(value['text_token']);events.extend(dict(frame=i,**e)for e in emitted)
            trace.append(dict(sequence=i,audio_samples=len(pcm),text_token=value['text_token'],is_ret=value['is_ret'],elapsed=elapsed,wall_since_start=time.monotonic()-start))
            await asyncio.sleep(max(0,start+(i+1)*.08-time.monotonic()))
        raw=np.concatenate(arrays);timed=timed_assembly(trace,arrays,24000,frame_seconds=.08)
        sf.write(folder/'user.wav',audio,24000,subtype='FLOAT');sf.write(folder/'raw-emitted.wav',raw,24000,subtype='FLOAT');sf.write(folder/'timed-emitted.wav',timed,24000,subtype='FLOAT')
        (folder/'trace.jsonl').write_text(''.join(json.dumps(t)+'\n'for t in trace))
        result=dict(id=row['id'],kind=row['kind'],condition=condition,complete=True,generated_text=engine.decode_text(tokens),actual_native_RET=sum(t['is_ret']for t in trace),actual_events=events,requests=connection.requests,backend='explicit narrow local exact-rational tool, not a general LLM',wall_seconds=time.monotonic()-start,audio_clock_seconds=len(audio)/24000,original_audio_sha256=row['audio_sha256'],presented_question=row['question'],teacher_or_answer_in_input=False,optimizer_updates=0,external_API_calls=0,files={p.name:sha(p)for p in folder.iterdir()})
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
    finally:connection.close()


async def run(a):
    import torch
    from .inference_guard import FrameGuard
    from .expanded_evaluation import make_engine
    from .expanded_capture import PP,VENDOR,WEIGHTS
    from evals.context_execution import math_attention
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);guard=FrameGuard(a.lease);lease=guard(600)
    if os.environ['WORLD_SIZE']!='4'or torch.cuda.device_count()!=4:raise ValueError('Exact four native probe workers required')
    if lease['source']!=a.source or lease.get('causal_scope_sha256')!=sha('docs/NATIVE-CAUSAL-PROBE-EXECUTION-2026-11-16.json'):raise ValueError('Exact probe source/scope required')
    prep=json.loads((a.root/'preparation.json').read_text());questions=json.loads((a.root/'questions.json').read_text());plan=json.loads(a.plan.read_text())
    if prep['source']!=a.source or sha(a.root/'questions.json')!=prep['questions_sha256']or len(questions)!=76:raise ValueError('Pinned complete input-only question cohort required')
    os.environ.update(CS_BASE_DIR=str(PP),CS_WEIGHTS_DIR=str(WEIGHTS),CS_VOICES_DIR=str(WEIGHTS/'voices'));__import__('sys').path.insert(0,str(VENDOR));math_attention();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    model='baseline'if rank<2 else'expanded105M';condition='prefix-text-only'if rank%2==0 else'audio-plus-prefix-text';a.capacity=model;a.trainer_source=plan['selection']['trainer_source']
    out=a.root/(model+'-'+condition);out.mkdir(exist_ok=False);engine=make_engine(a,plan['selection'])
    if rank%2==0:teacher_ret(engine,out,guard)
    capture=Path(plan['parent_root'])/'capture/shard-00';record=json.loads((capture/'capture.json').read_text())
    if record['reference_source_sha256']!=REFERENCE_SHA or sha(capture/'voice.pt')!=record['voice_sha256']:raise ValueError('Exact originalRP reference required')
    voice=torch.load(capture/'voice.pt',map_location='cpu',weights_only=True)['codes']
    (out/'profile.json').write_text(json.dumps(dict(source=a.source,model=model,condition=condition,actual_inference=engine.expanded_inference_proof,reference_sha256=REFERENCE_SHA,question_interface=prep['text_interface'],optimizer_updates=0,external_API_calls=0),indent=2)+'\n')
    results=[]
    with torch.inference_mode():
        for row in questions:
            result=await case(engine,row,condition,out,guard,voice);results.append(result)
            (out/'results.json').write_text(json.dumps(dict(source=a.source,required=76,cases=results,complete=len(results)==76,optimizer_updates=0),indent=2)+'\n')
            print(json.dumps(dict(event='causal_native_case_complete',model=model,condition=condition,completed=len(results),required=76)),flush=True)
    if len(results)!=76:raise ValueError('Complete condition required')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path);p.add_argument('--source',required=True);p.add_argument('--lease',type=Path);p.add_argument('--plan',type=Path);a=p.parse_args()
    if a.stage=='prepare':prepare(a.parent,a.root,a.source)
    else:asyncio.run(run(a))
