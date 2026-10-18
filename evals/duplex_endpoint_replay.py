"""Replay a causal endpoint policy on captured real ASR outputs and user PCM.

No model rerun, reference text input, backend dispatch or training occurs.
This is development policy replay, not fresh full-system acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import sentencepiece
from evals.duplex_endpoint import TurnEndpoint
from evals.moshi_model_download import sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('asr-run','assets','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();original=json.loads((args.asr_run/'report.json').read_text())
    assets=json.loads(args.assets.read_text())
    if (original['schema']!='dl-kyutai-streaming-asr-probe-v1' or not original['passed']
            or original['assets_sha256']!=sha256(args.assets)
            or original['reference_transcript_supplied'] is not False):
        raise RuntimeError('Endpoint replay lacks a complete causal ASR source')
    tokenizer=next(r for r in assets['assets'] if r['role']=='tokenizer');path=Path(tokenizer['path'])
    if sha256(path)!=tokenizer['sha256']:raise RuntimeError('Changed source ASR tokenizer')
    spm=sentencepiece.SentencePieceProcessor(str(path));results=[];args.out.mkdir(parents=True,exist_ok=False)
    for case in original['results']:
        if case['source_split']!='train' or not case['all_live_user_rows_exact_at_temporal_forward']:
            raise RuntimeError('Endpoint development replay requires complete approved TRAIN inputs')
        folder=args.asr_run/case['scenario'];pcm,rate=sf.read(folder/'input-user.wav',dtype='float32')
        if rate!=24000 or pcm.ndim!=1 or hashlib.sha256(pcm.tobytes()).hexdigest()!=case['user_pcm_sha256']:
            raise RuntimeError('Changed captured actual user PCM')
        events=json.loads((folder/'events.json').read_text());policy=TurnEndpoint(spm.decode)
        output=[];starts=[];deferred=0
        for i,event in enumerate(events):
            frame=pcm[i*1920:(i+1)*1920]
            frame=np.pad(frame,(0,1920-len(frame)))
            rms=float(np.sqrt(np.mean(frame**2)))
            score=event['heads'][2][0] if event['heads'] is not None else None
            r=policy.update(event['token'],score,rms,event['input_available_s'])
            if r['speech_started']:starts.append(event['input_available_s'])
            deferred+=int(r['candidate_deferred'])
            if r['utterance']:output.append(dict(r['utterance'],recorded_available_wall_s=event['available_wall_s']))
        results.append(dict(scenario=case['scenario'],raw_endpoint_count=len(case['end_of_turn_events']),
            endpoints=output,speech_starts=starts,deferred_candidate_frames=deferred,
            unended_text=spm.decode(policy.tokens),source_events_sha256=sha256(folder/'events.json'),
            source_user_pcm_sha256=case['user_pcm_sha256'],reference_transcript_supplied=False))
    report=dict(schema='dl-duplex-endpoint-development-replay-v1',passed=True,results=results,
        source_asr_report_sha256=sha256(args.asr_run/'report.json'),policy_code_sha256=sha256(Path(__file__).with_name('duplex_endpoint.py')),
        policy=dict(energy_floor=.003,quiet_frames=3,text_idle_frames=2,terminal_punctuation_required=True),
        model_executed_again=False,backend_executed=False,training_approved=False,
        microphone_acceptance=False,duplex_system_qualified=False,fresh_holdout=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
