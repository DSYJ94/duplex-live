"""Missing quantitative conditions only; prior official32 is reused, no learning."""
import argparse,json,math,time
from pathlib import Path
from .expanded_guard import sha
from .local_reasoner_reference import question_rows


def quantitative_rows(path):
    rows=[r for r in question_rows(path)if r['kind']!='basic']
    if len(rows)!=64:raise ValueError('Exact missing64 quantitative questions required')
    return rows


def run(a):
    import numpy as np,soundfile as sf,torch
    from scipy.signal import resample_poly
    from .inference_guard import FrameGuard
    from .official_personaplex import OfficialEngine
    from training.venus.foundation_native import timed_assembly
    guard=FrameGuard(a.lease)
    if guard.lease['source']!=a.source:raise ValueError('Exact admitted source required')
    rows=quantitative_rows(a.questions)
    with torch.inference_mode():
        engine=OfficialEngine(guard)
        for voice in ('NATF2','RPBen-normalized'):
            out=a.root/voice;out.mkdir(exist_ok=False);engine.select_voice(voice)
            profile=dict(source=a.source,voice=voice,voice_sha256=engine.voice_sha256,model_schema=engine.schema,prior_official32_reused='8203137',optimizer_updates=0,external_API_calls=0,live_microphone_latency=False,official_server_pin_prompt=True)
            (out/'profile.json').write_text(json.dumps(profile,indent=2)+'\n');results=[]
            for row in rows:
                if sha(row['audio'])!=row['audio_sha256']:raise ValueError('Original user audio changed')
                audio,rate=sf.read(row['audio'],dtype='float32')
                if rate!=row['rate']or audio.ndim!=1:raise ValueError('Original mono input required')
                if rate==16000:audio=resample_poly(audio,3,2).astype(np.float32)
                elif rate!=24000:raise ValueError('Unknown source clock')
                audio=np.pad(audio,(0,(-len(audio))%1920));guard(math.ceil(len(audio)/24000*2)+180)
                engine.reset();folder=out/row['id'];folder.mkdir();trace=[];arrays=[];tokens=[];start=time.monotonic()
                for i,offset in enumerate(range(0,len(audio),1920)):
                    before=time.monotonic();value=engine.step(audio[offset:offset+1920]);pcm=np.asarray(value['agent_pcm']if value['agent_pcm']is not None else[],dtype=np.float32)
                    if pcm.ndim!=1 or not np.isfinite(pcm).all():raise ValueError('Finite unmodified native output required')
                    arrays.append(pcm);tokens.append(value['text_token']);trace.append(dict(sequence=i,audio_samples=len(pcm),text_token=value['text_token'],is_ret=value['is_ret'],elapsed=time.monotonic()-before,wall_since_start=time.monotonic()-start))
                    time.sleep(max(0,start+(i+1)*.08-time.monotonic()))
                sf.write(folder/'user.wav',audio,24000,subtype='FLOAT');sf.write(folder/'raw-emitted.wav',np.concatenate(arrays),24000,subtype='FLOAT');sf.write(folder/'timed-emitted.wav',timed_assembly(trace,arrays,24000,frame_seconds=.08),24000,subtype='FLOAT')
                (folder/'trace.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in trace))
                result=dict(id=row['id'],complete=True,generated_text=engine.decode_text(tokens),actual_native_RET=sum(t['is_ret']for t in trace),backend_connected=False,actual_backend_requests=0,wall_seconds=time.monotonic()-start,audio_clock_seconds=len(audio)/24000,original_audio_sha256=row['audio_sha256'],question_or_gold_in_prompt=False,optimizer_updates=0,files={p.name:sha(p)for p in folder.iterdir()})
                (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');results.append(result)
                (out/'results.json').write_text(json.dumps(dict(source=a.source,voice=voice,required=64,complete=len(results)==64,cases=results,optimizer_updates=0),indent=2)+'\n')
                print(json.dumps(dict(event='official_quantitative_complete',voice=voice,completed=len(results),required=64)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','questions','lease'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source',required=True);run(p.parse_args())
