"""Complete ten-case real-time native diagnostic. Records evidence, never scores a pass."""
import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import time
import wave


def preflight(manifest):
    path=Path(manifest).resolve();plan=json.loads(path.read_text());files=0
    if len(plan['cases'])!=10:raise ValueError('Complete ten-case plan required')
    for case in plan['cases']:
        for turn in case['turns']:
            declared=Path(turn['path'])
            relative=declared.relative_to('/inputs') if declared.is_absolute() else Path(case['id'])/declared.name
            actual=(path.parent/relative).resolve()
            if path.parent not in actual.parents:raise ValueError('Input path escapes the manifest root')
            if hashlib.sha256(actual.read_bytes()).hexdigest()!=turn['sha256']:raise ValueError('Input hash changed')
            with wave.open(str(actual)) as w:
                if (w.getframerate(),w.getnchannels(),w.getsampwidth())!=(16000,1,2) or w.getnframes()==0:
                    raise ValueError('Nonempty16kHz mono PCM16 required')
            turn['path']=str(actual);files+=1
    if files!=18:raise ValueError('Complete eighteen input turns required')
    return plan


async def run_case(case, endpoint, output):
    import numpy as np
    import soundfile as sf
    import websockets
    folder=output/case['id'];folder.mkdir(parents=True,exist_ok=False)
    clips=[]
    for turn in case['turns']:
        path=Path(turn['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==turn['sha256']
        with wave.open(str(path)) as w:
            assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(16000,1,2)
            clips.append(w.readframes(w.getnframes()))
    events=[];sequence=0;turn_index=0;position=0;next_turn_at=None;last_end=None
    started_work=None;saw_speech=False;trigger_valid=True
    async with websockets.connect(endpoint,max_size=4*1024*1024,open_timeout=30) as ws:
        ready=json.loads(await asyncio.wait_for(ws.recv(),120))
        assert ready.get('ready'),ready
        (folder/'profile.json').write_text(json.dumps(ready,indent=2)+'\n')
        origin=time.monotonic()
        with (folder/'trace.jsonl').open('w') as trace:
            for sequence in range(100):
                clock=time.monotonic()-origin
                if turn_index<len(clips) and position>=len(clips[turn_index]):
                    last_end=sequence if last_end is None else last_end
                    if turn_index==len(clips)-1:
                        if sequence-last_end>=20:break
                    elif case['id'].startswith('quiet-'):
                        next_turn_at=last_end+8 if next_turn_at is None else next_turn_at
                    elif case['id']=='pause-resume':
                        if turn_index==0 and (saw_speech or sequence-last_end>=20):
                            trigger_valid &= saw_speech;next_turn_at=sequence
                        elif turn_index==1:next_turn_at=last_end+8
                    elif case['id'] in ('pending-followup','cancel-pending'):
                        if started_work is not None or sequence-last_end>=20:
                            trigger_valid &= started_work is not None
                            if started_work is None or time.monotonic()-started_work>=1:
                                next_turn_at=sequence
                    if next_turn_at is not None and sequence>=next_turn_at:
                        turn_index+=1;position=0;last_end=None;next_turn_at=None
                pcm=bytes(32000)
                if turn_index<len(clips) and position<len(clips[turn_index]):
                    pcm=clips[turn_index][position:position+32000];position+=len(pcm);pcm=pcm.ljust(32000,b'\0')
                await ws.send(pcm)
                result=json.loads(await asyncio.wait_for(ws.recv(),90))
                if result.get('error'):raise RuntimeError(result)
                audio=np.frombuffer(base64.b64decode(result.pop('audio')),dtype='<f4')
                assert np.isfinite(audio).all()
                sf.write(folder/f'chunk-{sequence:03d}.wav',audio,24000,subtype='FLOAT')
                if not result['is_listen'] and result['audio_samples'] and result['peak']>.001:
                    saw_speech=True
                for e in result.get('events',[]):
                    if e['type']=='worker_started' and started_work is None:started_work=e['monotonic']
                    events.append(e)
                result.update(input_sha256=hashlib.sha256(pcm).hexdigest(),input_turn=turn_index,
                    wall_since_start=clock,chunk_file=f'chunk-{sequence:03d}.wav')
                trace.write(json.dumps(result)+'\n');trace.flush()
                await asyncio.sleep(max(0,origin+sequence+1-time.monotonic()))
    result=dict(case=case['id'],complete=True,chunks=sequence,wall_seconds=time.monotonic()-origin,
        dynamic_trigger_observed=trigger_valid,worker_events=events,qualified=False,
        audio_review_required=True,word_audit_complete=all(x['independently_word_audited'] for x in case['turns']))
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


async def main(args):
    plan=preflight(args.manifest)
    if args.preflight_only:
        print(json.dumps(dict(preflight_passed=True,cases=10,input_files=18,model_loaded=False)));return
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    results=[]
    for case in plan['cases']:
        try:result=await run_case(case,args.endpoint,out)
        except Exception as e:result=dict(case=case['id'],complete=False,error=str(e),qualified=False)
        results.append(result);(out/'results.json').write_text(json.dumps(dict(cases=results,all10_complete=len(results)==10 and all(x['complete'] for x in results),qualified=False),indent=2)+'\n')
        print(json.dumps(result),flush=True)
    if not all(x['complete'] for x in results):raise RuntimeError('Incomplete ten-case diagnostic; no passing subset')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--output');p.add_argument('--preflight-only',action='store_true');p.add_argument('--endpoint',default='ws://127.0.0.1:8032/ws');args=p.parse_args()
    if not args.preflight_only and not args.output:p.error('--output required for inference')
    asyncio.run(main(args))
