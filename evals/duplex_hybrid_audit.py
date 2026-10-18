"""Independent speech diagnostic on actual causal hybrid playout per utterance.

Cancelled generated material is excluded exactly as in the captured playback.
Partial utterances get transcripts and time-bound checks, never a full-text WER.
No speech intelligibility score grants task or microphone acceptance.
"""
import argparse
import json
import math
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.utterance_playout import render_utterances


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--include-completed-cases-from-failed-suite',action='store_true')
    args=ap.parse_args()
    import numpy as np
    import soundfile as sf
    import whisper_timestamped as whisper
    from scipy.signal import resample_poly
    from data.synthetic.filter import norm_words,wer
    report=json.loads((args.run/'report.json').read_text())
    if (report['schema']!='dl-concurrent-hybrid-development-v1' or (not report['component_execution_completed'] and not args.include_completed_cases_from_failed_suite)
            or not report['same_process_same_gpu'] or report['hybrid_system_qualified'] is not False
            or any(c['reference_transcript_supplied'] for c in report['results'])):
        raise RuntimeError('Audit requires the explicit unqualified actual concurrent hybrid source')
    if not report['results']:raise RuntimeError('No completed case is available for actual-played speech audit')
    args.out.mkdir(parents=True,exist_ok=False);model=whisper.load_model('turbo',device='cuda');results=[]
    for case in report['results']:
        folder=args.run/case['scenario'];audio,rate=sf.read(folder/'model.wav',dtype='float32')
        if sha256(folder/'model.wav')!=case['audio_sha256'] or rate!=24000:raise RuntimeError('Changed hybrid capture')
        ledger=json.loads((folder/'packets.json').read_text());saved=np.load(folder/'generated-packets.npz')
        if set(saved.files)!={str(i) for i in range(len(ledger))}:raise RuntimeError('Incomplete generated packet inventory')
        packets=[dict(event_id=p['event_id'],available_s=p['available_s'],pcm=saved[str(i)]) for i,p in enumerate(ledger)]
        events=json.loads((folder/'events.json').read_text())
        stops={e['event_id']:e['actual_stop_s'] for e in events if e['kind']=='speech_interrupted'}
        reproduced,playout=render_utterances(packets,rate,.16,stops)
        if not np.array_equal(audio,reproduced) or playout!=case['playout']:raise RuntimeError('Captured interrupted playout cannot be reproduced exactly')
        complete={r['event_id'] for r in case['fully_played_responses']}
        starts={e['event_id']:e for e in events if e['kind']=='speech_started'}
        caseout=args.out/case['scenario'];caseout.mkdir()
        for identity,group in playout['events'].items():
            if not group['played_samples']:continue
            positions=[]
            for p in group['packets']:
                a=p['playback_start_sample'];b=a+p['samples']
                if identity in stops:b=min(b,math.ceil(stops[identity]*rate))
                if b>a:positions.append((a,b))
            first=min(p[0] for p in positions);last=max(p[1] for p in positions);clip=audio[first:last]
            if len(clip)!=last-first:raise RuntimeError('Played clip exceeds actual capture')
            sf.write(caseout/(identity+'-played.wav'),clip,rate,subtype='FLOAT')
            g=math.gcd(rate,16000);resampled=resample_poly(clip,16000//g,rate//g).astype(np.float32)
            result=whisper.transcribe(model,resampled,language='en',vad=False,beam_size=5,best_of=5,temperature=0.,verbose=None)
            duration=len(resampled)/16000;bounds=True
            for segment in result['segments']:
                for item in [segment]+segment.get('words',[]):
                    if not 0<=item['start']<=item['end']<=duration+.02:bounds=False
            expected=starts[identity]['text'];fully=identity in complete
            row=dict(scenario=case['scenario'],event_id=identity,task_id=starts[identity]['task_id'],
                played_start_s=first/rate,played_end_s=last/rate,fully_played=fully,planned_text=expected,
                transcript=result['text'],word_error_rate=wer(norm_words(expected),norm_words(result['text'])) if fully else None,
                valid_timestamp_bounds=bounds,played_audio_sha256=sha256(caseout/(identity+'-played.wav')),
                cancelled_generated_material_excluded=True,task_qualified=False)
            (caseout/(identity+'-whisper.json')).write_text(json.dumps(result,indent=2)+'\n');results.append(row)
    audit=dict(schema='dl-concurrent-hybrid-speech-audit-v1',source_report_sha256=sha256(args.run/'report.json'),
        results=results,excluded_failed_cases=report.get('failures',[]),
        explicitly_partial_suite_diagnostic=args.include_completed_cases_from_failed_suite,
        source_suite_execution_completed=report['component_execution_completed'],
        actual_causal_playout_reproduced_exactly=True,independent_model='whisper turbo',
        partial_utterance_wer_computed=False,hybrid_system_qualified=False,microphone_acceptance=False)
    (args.out/'report.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
