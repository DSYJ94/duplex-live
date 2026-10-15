"""Vocode verified native Lychee tokens, offline or at recorded arrival times.

The replay measures the published streaming vocoder only. It is explicitly not
concurrent LM+PCM, browser playback, microphone or delegation acceptance.
"""
import argparse
import io
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from evals.moshi_model_download import sha256
from evals.lychee_model_download import SOURCE_COMMIT


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','native-run','voice','voice-provenance','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--mode',choices=('offline','stream-replay'),default='offline')
    ap.add_argument('--hop',type=int,choices=(8,25),default=25)
    args=ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from evals.lychee_environment_check import configure_bf16
    configure_bf16()
    assets=json.loads(args.assets.read_text());source=Path(assets['source'])
    if (assets['source_commit']!=SOURCE_COMMIT or subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)):
        raise RuntimeError('Accepted vocoder source changed')
    voice=json.loads(args.voice_provenance.read_text())
    if (voice['schema']!='dl-voice-codec-probe-v1' or voice['source_model']!='gpt-live-1'
            or voice['reference_sha256']!=sha256(args.voice)):
        raise RuntimeError('Synthetic reference no longer matches accepted receipt')
    model=assets['models']['vocoder'];root=Path(model['path'])
    for name,record in model['files'].items():
        if sha256(root/name)!=record['sha256']:
            raise RuntimeError('Pinned vocoder asset changed: '+name)
    native=json.loads((args.native_run/'report.json').read_text())
    loading=native['strict_loading']
    if (loading['checkpoint_tensor_count']!=982 or not loading['model']['all_parameter_shards_loaded']
            or not loading['audio']['strict_keys_and_shapes'] or loading['model']['side_head_fallback_used']):
        raise RuntimeError('Native token source has no complete released-model loading proof')
    events=json.loads((args.native_run/'events.json').read_text())
    if not events or any(not e['audio'] or any(not isinstance(t,int) or not 0<=t<6561 for t in e['audio']) for e in events):
        raise RuntimeError('Missing/out-of-vocabulary native speech tokens')
    sys.path[:0]=[str(source),str(source/'third_party/Step-Audio2')]
    from evals.minicpm_recovery_probe import read_reference_wav
    def wav_load(*pos,**kw):
        pcm,rate=read_reference_wav(*pos,**kw)
        return torch.from_numpy(pcm),rate
    torchaudio.load=wav_load
    from token2wav import Token2wav
    args.out.mkdir(exist_ok=False)
    vocoder=Token2wav(str(root/'token2wav'))
    provenance={'schema':'dl-lychee-native-vocode-v1','mode':args.mode,'source_commit':SOURCE_COMMIT,
        'repo_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'assets_manifest_sha256':sha256(args.assets),'voice_sha256':sha256(args.voice),
        'native_report_sha256':sha256(args.native_run/'report.json'),
        'native_events_sha256':sha256(args.native_run/'events.json'),
        'flow_implementation':{'file':inspect.getfile(type(vocoder.flow)),
            'sha256':sha256(Path(inspect.getfile(type(vocoder.flow))))},
        'vocoder_weights_strict':True,'concurrent_lm_tested':False,'microphone_acceptance':False,
        'training_approved':False,'voice_quality_verdict':'pending independent audit/listening'}
    packets,rows,packet_groups=[],[],[]
    if args.mode=='offline':
        torch.manual_seed(424242)
        audio=np.zeros(round(native['paced_tokens']['input_seconds']*24000),dtype=np.float32)
        started=time.monotonic()
        with torch.inference_mode():
            for i,event in enumerate(events):
                blob=vocoder(event['audio'],str(args.voice))
                pcm,rate=sf.read(io.BytesIO(blob),dtype='float32')
                if rate!=24000 or pcm.ndim!=1 or not np.isfinite(pcm).all():raise RuntimeError('Invalid native vocoder PCM')
                sf.write(args.out/f'utterance-{i:03d}.wav',pcm,rate)
                offset=round(event['start_time']*rate);end=offset+len(pcm)
                audio=np.pad(audio,(0,max(0,end-len(audio))));audio[offset:end]+=pcm
        provenance['vocoder_wall_s']=time.monotonic()-started
        provenance['output_timing']='offline placement by native token positions; not actual availability'
    else:
        from transformers import AutoTokenizer
        from lychee_fd.runtime.vllm_generation import RealtimeEventAdapter
        from evals.audio_playout import render_packets
        duplex=Path(assets['models']['duplex']['path']);config=json.loads((duplex/'config.json').read_text())
        tokenizer=AutoTokenizer.from_pretrained(duplex,local_files_only=True,trust_remote_code=True)
        fw=SimpleNamespace(text_pad_token_id=config['text_pad_token_id'],
            tts_pad_id=tokenizer(['<tts_pad>']).input_ids[0][0],
            stoken_delay_token_id=config['stoken_delay_token_id'],stoken_pad_token_id=config['stoken_pad_token_id'])
        adapter=RealtimeEventAdapter(tokenizer,fw,tts_chunk_size=1)
        lookahead=int(getattr(vocoder.flow,'pre_lookahead_len',3))
        # Published startup warmup; caches are reset before real output. No user
        # waveform or future generated tokens enter this warmup.
        with torch.inference_mode():
            vocoder.set_stream_cache(str(args.voice))
            vocoder.stream([0]*(args.hop+lookahead+1),str(args.voice),last_chunk=False)
            vocoder.stream([0],str(args.voice),last_chunk=True)
        torch.cuda.synchronize();torch.manual_seed(424242)
        records=[json.loads(line) for line in (args.native_run/'incremental-events.jsonl').read_text().splitlines()]
        records.append({'available_s':native['paced_tokens']['wall_seconds_excluding_load_and_vocoder'],
            'event':{'type':'generation_complete','reason':'recorded input/session exhaustion'}})
        started=time.monotonic();buffer=[];active=None;groups={};consumed=0;interrupted=0
        def emit(tokens,last,advance,due):
            nonlocal consumed
            begin=time.monotonic()-started
            with torch.inference_mode():blob=vocoder.stream(tokens,str(args.voice),last_chunk=last)
            torch.cuda.synchronize();available=time.monotonic()-started
            pcm=np.frombuffer(blob,dtype='<i2').astype(np.float32)/32768.
            row={'event_id':active,'tokens':len(tokens),'advance_tokens':advance,'last':last,
                'tokens_available_s':due,'started_s':begin,'pcm_available_s':available,'compute_s':available-begin,'samples':len(pcm)}
            rows.append(row);consumed+=advance
            if len(pcm):
                packets.append((available,pcm));packet_groups.append(active);groups[active].append((available,pcm))
        for record in records:
            due=record['available_s'];time.sleep(max(0.,started+due-time.monotonic()))
            for event in adapter.feed(record['event']):
                if event['type']=='event_start':
                    if active is not None:raise RuntimeError('Overlapping streaming vocoder events')
                    active=event['event_id'];buffer=[];groups[active]=[]
                    vocoder.set_stream_cache(str(args.voice))
                elif event['type']=='audio_chunk':
                    if active is None:raise RuntimeError('Speech tokens outside a native speaking event')
                    values=event['stoken_ids']
                    if any(not 0<=t<6561 for t in values):raise RuntimeError('Invalid codec token; never drop silently')
                    buffer.extend(values);window=args.hop+lookahead
                    while len(buffer)>window:
                        emit(buffer[:window],False,args.hop,due);buffer=buffer[args.hop:]
                elif event['type']=='event_end' and active is not None:
                    if event.get('interrupt'):
                        interrupted+=len(buffer);buffer=[]
                    elif buffer:
                        emit(buffer,True,len(buffer),due);buffer=[]
                    active=None
        if active is not None:raise RuntimeError('Native trace did not finalize its last speaking event')
        if not packets:raise RuntimeError('Streaming vocoder emitted no PCM')
        audio,ledger=render_packets(packets,24000,.16)
        per_event={key:render_packets(value,24000,.16)[1] for key,value in groups.items() if value}
        provenance.update(hop_tokens=args.hop,lookahead_tokens=lookahead,
            recorded_token_trace_sha256=sha256(args.native_run/'incremental-events.jsonl'),
            replay_wall_s=time.monotonic()-started,first_pcm_available_s=packets[0][0],
            total_unique_tokens_consumed=consumed,interrupted_unplayed_tokens=interrupted,
            within_event_underrun_samples=sum(r['underrun_samples'] for i,r in enumerate(ledger['packets'])
                if i and packet_groups[i]==packet_groups[i-1]),
            terminal_flush='recorded input exhaustion, not a sampled model turn-end decision',
            vocoder_compute_p95_s=float(np.percentile([r['compute_s'] for r in rows],95)),
            output_timing='actual vocoder availability replaying recorded LM arrivals; no concurrent LM')
        (args.out/'playout.json').write_text(json.dumps({'global_including_intended_idle_gaps':ledger,'per_event':per_event},indent=2)+'\n')
        (args.out/'pcm-packets.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        sf.write(args.out/'packet-stream.wav',np.concatenate([p for _,p in packets]),24000)
    if not np.isfinite(audio).all() or not np.any(np.abs(audio)>.001) or np.max(np.abs(audio))>1:
        raise RuntimeError('Missing, invalid or clipping vocoder speech')
    sf.write(args.out/'model.wav',audio,24000)
    provenance.update(text=native['text'],audio_sha256=sha256(args.out/'model.wav'))
    (args.out/'report.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance),flush=True)


if __name__=='__main__':
    main()
