"""Reconstruct TTS packet playout with per-utterance buffering and actual stops.

Recorded components only. No PCM interpolation, word replacement or model-
quality claim; simultaneous LM/vocoder/transport still needs measurement.
"""
import argparse
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def recorded_stops(records):
    counter=0;active=None;stops={}
    for record in records:
        event=record['event']
        if event['type']=='state_change':
            before,after=event['from'].upper(),event['to'].upper()
            if active is not None and after=='L':
                if event.get('interrupt'):stops[active]=record['available_s']
                active=None
            if (before=='L' and after in ('S','B')) or (before=='B' and after=='S'):
                counter+=1;active=f'evt-{counter}'
        elif event['type']=='speaking_done':
            active=None
    return stops


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--native-run',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--buffer',type=float,default=.24)
    args=ap.parse_args()
    import numpy as np
    import soundfile as sf
    from evals.audio_playout import render_packets
    from evals.utterance_playout import render_utterances
    report=json.loads((args.source/'report.json').read_text())
    if report['mode']!='stream-replay' or report['recorded_token_trace_sha256']!=sha256(args.native_run/'incremental-events.jsonl'):
        raise RuntimeError('Original recorded token arrival proof changed')
    if report['audio_sha256']!=sha256(args.source/'model.wav'):
        raise RuntimeError('Original component capture changed')
    stream,rate=sf.read(args.source/'packet-stream.wav',dtype='float32')
    original,original_rate=sf.read(args.source/'model.wav',dtype='float32')
    if stream.ndim!=1 or rate!=24000 or original_rate!=rate:
        raise RuntimeError('Invalid original PCM capture')
    rows=[json.loads(line) for line in (args.source/'pcm-packets.jsonl').read_text().splitlines()]
    cursor=0;packets=[]
    for row in rows:
        if not row['samples']:continue
        end=cursor+row['samples']
        if end>len(stream):raise RuntimeError('Packet samples exceed actual component WAV')
        packets.append({'event_id':row['event_id'],'available_s':row['pcm_available_s'],'pcm':stream[cursor:end]})
        cursor=end
    if cursor!=len(stream):raise RuntimeError('Unaccounted original PCM samples')
    legacy,_=render_packets([(p['available_s'],p['pcm']) for p in packets],rate,.16)
    if not np.array_equal(legacy,original):raise RuntimeError('Original packet ledger does not reproduce recorded WAV exactly')
    records=[json.loads(line) for line in (args.native_run/'incremental-events.jsonl').read_text().splitlines()]
    stops=recorded_stops(records)
    output,playout=render_utterances(packets,rate,args.buffer,stops)
    args.out.mkdir(exist_ok=False)
    sf.write(args.out/'model.wav',output,rate)
    result={'schema':'dl-lychee-utterance-playout-v1','source':str(args.source),
        'source_report_sha256':sha256(args.source/'report.json'),'source_packet_ledger_sha256':sha256(args.source/'pcm-packets.jsonl'),
        'original_capture_reproduced_exactly':True,'per_utterance_startup_buffer_s':args.buffer,
        'recorded_native_interrupts':stops,'control_timing':'recorded native LM decisions; concurrent delivery untested',
        'playout':playout,'text':report['text'],'audio_sha256':sha256(args.out/'model.wav'),
        'pcm_values_changed':False,'training_approved':False,'microphone_acceptance':False,
        'independent_speech_and_listening_verdict':'pending'}
    (args.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('original_capture_reproduced_exactly','per_utterance_startup_buffer_s',
        'recorded_native_interrupts')}),flush=True)
    print(json.dumps({k:playout[k] for k in ('source_samples','played_samples','cancelled_before_playback_samples',
        'within_utterance_underrun_samples','sample_accounting_exact')}),flush=True)


if __name__=='__main__':
    main()
