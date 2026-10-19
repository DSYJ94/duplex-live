"""Stronger independent full20 same-audio diagnostic; never an input approval."""
import argparse
import json
import math
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.native_fresh_inputs import load_inputs, waveform, PLAN_SHA256

# An explicit spelling equivalence only. Warm/worn remains a word substitution.
SPELLINGS = {'color': 'colour'}


def agreement(reference, transcript):
    from data.synthetic.filter import norm_words, wer
    a, b = norm_words(reference), norm_words(transcript)
    return dict(raw_wer=wer(a,b), spelling_only_wer=wer(
        [SPELLINGS.get(w,w) for w in a], [SPELLINGS.get(w,w) for w in b]),
        spelling_equivalents=SPELLINGS.copy())


def run(data, prior_path, out, device):
    import numpy as np
    import soundfile as sf
    import whisper_timestamped as whisper
    import whisper as package
    from scipy.signal import resample_poly
    from evals.moshi_speech_audit import asr_window, restore_clock
    manifest = load_inputs(data); prior = json.loads(prior_path.read_text())
    if (prior['schema'] != 'dl-native-fresh-quality-v1' or prior['passed']
            or prior['inputs_manifest_sha256'] != sha256(data/'inputs.json')
            or prior['plan_sha256'] != PLAN_SHA256 or len(prior['results']) != 20
            or prior['threshold'] != .05 or prior['peak_bar'] != .995):
        raise RuntimeError('Retain the exact complete failed full20 input audit')
    out.mkdir(parents=True, exist_ok=False)
    model = whisper.load_model('large-v3', device=device)
    rows = []
    for row in manifest['rows']:
        reference = json.loads(Path(row['reference_path']).read_text()); folder = Path(row['path']).parent; clips = []
        for i, utterance in enumerate(reference['utterances']):
            path = folder/utterance['source_file']; pcm, rate = sf.read(path, dtype='float32')
            window, head, tail, silent = asr_window(resample_poly(pcm,2,3).astype(np.float32),True,True,True)
            t = dict(text='',segments=[]) if silent else whisper.transcribe(model,window,
                language='en',vad=False,beam_size=5,best_of=5,temperature=0.,verbose=None)
            t = restore_clock(t,head/16000)
            name = row['scenario_id']+'-%d-asr.json'%i
            (out/name).write_text(json.dumps(t,indent=2)+'\n')
            words = [w for s in t['segments'] for w in s.get('words',[])]
            valid = bool(words) and all(math.isfinite(w['start']) and math.isfinite(w['end'])
                and 0 <= w['start'] <= w['end'] <= len(pcm)/rate+.02
                for s in t['segments'] for w in [s]+s.get('words',[]))
            scores = agreement(utterance['text'],t['text']); wave = waveform(pcm,rate)
            clips.append(dict(ordinal=i,raw_wav_sha256=sha256(path),asr_file=name,
                asr_sha256=sha256(out/name),valid_bounds=valid,waveform=wave,agreement=scores,
                diagnostic_meets_same_bars=wave['passed'] and valid and scores['spelling_only_wer']<=.05))
        rows.append(dict(scenario_id=row['scenario_id'],wav_sha256=row['wav_sha256'],clips=clips,
            diagnostic_meets_same_bars=all(c['diagnostic_meets_same_bars'] for c in clips)))
        print(json.dumps(rows[-1]),flush=True)
    report = dict(schema='dl-native-fresh-full20-asr-diagnostic-v1',diagnostic_only=True,
        source_code_sha256=sha256(Path(__file__)),plan_sha256=PLAN_SHA256,
        inputs_manifest_sha256=sha256(data/'inputs.json'),original_failed_quality_sha256=sha256(prior_path),
        asr_model='large-v3',asr_weight_sha256=package._MODELS['large-v3'].split('/')[-2],
        all20_retained=True,complete_cases=len(rows),complete_clips=sum(len(r['clips']) for r in rows),
        threshold=.05,peak_bar=.995,expected_text_prompt_used=False,original_waveforms_changed=False,
        generation_calls=0,rerolls=0,passing_subset_accepted=False,training_approved=False,
        inference_approved=False,original_failed_verdict_preserved=True,results=rows,
        all20_diagnostic_meets_same_bars=all(r['diagnostic_meets_same_bars'] for r in rows))
    assert len(rows)==20
    (out/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--prior',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=ap.parse_args();run(args.data,args.prior,args.out,args.device)


if __name__=='__main__':main()
