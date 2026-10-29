"""Native streaming timings using one existing synthetic TRAIN/DEV WAV.

Uses the existing loopback service; does not provision, load another model,
update weights, save owner speech, or make model API calls. Optional saved outputs are synthetic-only. Serial measurements alone
are not an end-to-end conversation latency or concurrent backlog benchmark.
"""
import argparse,hashlib,json,time,math,base64
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import build_opener,ProxyHandler


def summary(rows):
    groups={}
    for label in ('listen','speak'):
        selected=[r for r in rows if r['is_listen']==(label=='listen')]
        values=[r['processing_seconds'] for r in selected]
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):
            raise ValueError('Invalid native processing timing; no performance claim')
        groups[label]=dict(frames=len(values),processing_min=min(values) if values else None,
            processing_max=max(values) if values else None,
            processing_mean=sum(values)/len(values) if values else None)
    return groups


def main(a):
    import numpy as np,soundfile as sf
    from websockets.sync.client import connect
    url=a.url.rstrip('/');parts=urlsplit(url)
    if parts.scheme!='http' or parts.hostname not in ('127.0.0.1','localhost') or parts.username or parts.password or parts.path:
        raise ValueError('Existing local tunnel URL only')
    root=Path('runs/venus-training-20261027/control-paced-timelines-v3').resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    row=next(r for r in manifest['scenes'] if r['id']==a.scene)
    if row['fold'] not in ('TRAIN','DEV'):raise ValueError('Synthetic TRAIN/DEV timings only; no heldout inputs')
    wav=root/row['id']/'user.wav'
    if hashlib.sha256(wav.read_bytes()).hexdigest()!=row['user_wav_sha256']:raise ValueError('Exact approved synthetic waveform required')
    wave,sr=sf.read(wav,dtype='float32')
    if sr!=16000 or wave.ndim!=1 or len(wave)!=row['frames']*16000 or not np.isfinite(wave).all() or np.abs(wave).max()>.995:
        raise ValueError('Complete original one-second native framing required')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    opener=build_opener(ProxyHandler({}));records=[];profiles=[]
    def health():
        with opener.open(url+'/health',timeout=5) as response:return json.load(response)
    for variant in a.variants.split(','):
        if variant not in ('released','candidate'):raise ValueError('Explicit original or candidate only')
        if health()['busy']:raise RuntimeError('Owner conversation active; benchmark does not interrupt it')
        began=time.monotonic()
        with connect(url.replace('http:','ws:')+'/ws?variant='+variant,origin=url,proxy=None,open_timeout=10,close_timeout=3) as ws:
            ready=json.loads(ws.recv(timeout=40))
            if not ready.get('ready'):raise RuntimeError('Native conversation slot unavailable')
            profiles.append(dict(variant=variant,ready_seconds=time.monotonic()-began,profile=ready['profile']))
            due=time.monotonic();audio_parts=[];generated_text=[]
            for index in range(row['frames']):
                if time.monotonic()<due:time.sleep(due-time.monotonic())
                pcm=np.rint(wave[index*16000:(index+1)*16000]*32767).astype('<i2').tobytes()
                sent=time.monotonic();ws.send(pcm);frame=json.loads(ws.recv(timeout=40))
                if 'error' in frame:raise RuntimeError('Native frame failed: '+frame.get('type','Error'))
                if frame['sequence']!=index:raise ValueError('Native frame order changed')
                event=dict(variant=variant,sequence=index,is_listen=frame['is_listen'],processing_seconds=frame['elapsed'],
                    roundtrip_seconds=time.monotonic()-sent,output_audio_seconds=frame['audio_samples']/24000,
                    wire_audio_bytes=frame.get('wire_audio_bytes'),timings=frame.get('timings'))
                if a.save_synthetic_audio:
                    if frame['audio']:
                        samples=np.frombuffer(base64.b64decode(frame['audio']),dtype='<f4').copy()
                        if len(samples)!=frame['audio_samples']:raise ValueError('Native waveform length differs')
                    elif frame.get('wire_silence_omitted'):samples=np.zeros(frame['audio_samples'],dtype=np.float32)
                    else:samples=np.zeros(0,dtype=np.float32)
                    audio_parts.append(samples);generated_text.append(frame.get('text',''))
                records.append(event);print(json.dumps(event),flush=True);due+=1
            if a.save_synthetic_audio:
                audio=np.concatenate(audio_parts);sf.write(out/(variant+'-native.wav'),audio,24000,subtype='FLOAT')
                (out/(variant+'-generated-text.json')).write_text(json.dumps(dict(text=''.join(generated_text),source='model response to exact authored synthetic TRAIN input',owner_audio=False),indent=2)+'\n')
        for _ in range(50):
            if not health()['busy']:break
            time.sleep(.1)
        else:raise RuntimeError('Native slot did not reset')
    result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),complete=len(records)==row['frames']*len(profiles),
        scene=row['id'],fold=row['fold'],source_wav_sha256=row['user_wav_sha256'],profiles=profiles,frames=records,
        summaries={p['variant']:summary([r for r in records if r['variant']==p['variant']]) for p in profiles},
        microphone_used=False,speech_saved=a.save_synthetic_audio,optimizer_updates=0,new_gpu_allocations=0,
        concurrent_backlog_benchmark=False,first_audio_latency_benchmark=False,adoption=False)
    (out/'timings.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://127.0.0.1:8037')
    p.add_argument('--scene',default='venus-control-v1-train-ordinary-0145');p.add_argument('--variants',default='released,candidate');p.add_argument('--output',required=True);p.add_argument('--save-synthetic-audio',action='store_true');main(p.parse_args())
