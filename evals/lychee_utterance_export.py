"""Export complete, uncancelled speech from actual concurrent playout.

Each source packet must equal its actual final playout samples exactly. Audible
gaps remain in exported clips and are reported as failed continuity evidence.
Interrupted or input-exhausted utterances remain recorded exclusions, not passes.
"""
import argparse
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def export(capture, out):
    import numpy as np
    import soundfile as sf
    source=capture/'vocoder'
    report=json.loads((source/'report.json').read_text())
    if (report['schema']!='dl-lychee-concurrent-pcm-v1'
            or not report['actual_concurrent_lm_execution']
            or not report['playout']['sample_accounting_exact']
            or sha256(source/'model.wav')!=report['audio_sha256']):
        raise RuntimeError('Concurrent playout provenance is invalid')
    utterances=json.loads((source/'utterance-events.json').read_text())
    rows=[json.loads(line) for line in (source/'pcm-packets.jsonl').read_text().splitlines()]
    raw,rate=sf.read(source/'packet-stream.wav',dtype='float32')
    played,played_rate=sf.read(source/'model.wav',dtype='float32')
    if rate!=24000 or played_rate!=rate or raw.ndim!=1 or played.ndim!=1:
        raise RuntimeError('Expected native mono 24kHz concurrent PCM')
    if not np.isfinite(raw).all() or not np.isfinite(played).all():raise RuntimeError('Nonfinite PCM')
    groups={};offset=0
    for row in rows:
        size=row['samples']
        if not isinstance(size,int) or size<0:raise RuntimeError('Invalid packet sample count')
        if size:groups.setdefault(row['event_id'],[]).append(raw[offset:offset+size])
        offset+=size
    if offset!=len(raw):raise RuntimeError('Packet ledger does not cover all source samples')
    bindings={name:sha256(source/name) for name in ('report.json','utterance-events.json',
        'pcm-packets.jsonl','packet-stream.wav','model.wav')}
    seen=set();accepted=[];excluded=[]
    out.mkdir(parents=True,exist_ok=False)
    for event in utterances:
        identity=event['event_id']
        if identity in seen:raise RuntimeError('Duplicate actual native utterance')
        seen.add(identity)
        details=report['playout']['events'].get(identity)
        if (event['native_interrupt'] or event['source_ended_by_input_exhaustion']
                or details is None or details['actual_stop_s'] is not None):
            excluded.append({'event':event,'reason':'interrupted, input-exhausted or no played PCM'})
            continue
        if (not event['text'].strip() or details['source_samples']!=details['played_samples']):
            raise RuntimeError('Uncancelled native utterance has incomplete text/PCM playout')
        pcm=np.concatenate(groups[identity])
        if len(pcm)!=details['source_samples']:raise RuntimeError('Per-utterance sample count mismatch')
        begin=details['packets'][0]['playback_start_sample'];cursor=0
        for packet in details['packets']:
            start=packet['playback_start_sample'];size=packet['samples']
            if not np.array_equal(pcm[cursor:cursor+size],played[start:start+size]):
                raise RuntimeError('Source packet PCM differs from actual playout')
            cursor+=size
        if cursor!=len(pcm):raise RuntimeError('Played packet ledger lost source samples')
        end=details['packets'][-1]['playback_start_sample']+details['packets'][-1]['samples']
        clip=played[begin:end]
        folder=out/identity;folder.mkdir()
        sf.write(folder/'model.wav',clip,rate,subtype='PCM_16')
        result={'schema':'dl-concurrent-complete-utterance-v1','text':event['text'],
            'event':event,'source_capture':str(capture),'source_bindings':bindings,
            'source_sample_range':[begin,end],'sample_equality_verified':True,
            'actual_within_utterance_gap_samples':details['within_utterance_underrun_samples'],
            'audio_continuity_pass':details['within_utterance_underrun_samples']==0,
            'training_approved':False,'task_success_verified':False,
            'audio_sha256':sha256(folder/'model.wav')}
        (folder/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        accepted.append(str(folder))
    if set(groups)-seen:raise RuntimeError('PCM without actual final utterance metadata')
    manifest={'schema':'dl-concurrent-utterance-export-v1','source_bindings':bindings,
        'capture_continuity_pass':report['playout']['within_utterance_underrun_samples']==0,
        'accepted':accepted,'excluded':excluded,'training_approved':False,'task_success_verified':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'audit-runs.txt').write_text(''.join(path+'\n' for path in accepted))
    if not accepted:raise RuntimeError('No complete concurrent utterances available to audit')
    print(json.dumps({'complete':len(accepted),'excluded':len(excluded)}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();export(args.capture,args.out)


if __name__=='__main__':main()
