"""Sealed second full20 collection; original failed collection remains immutable."""
import argparse
import json
import math
from pathlib import Path
import shutil
from evals.moshi_model_download import sha256

PLAN = Path(__file__).with_name('fixtures')/'native-fulltask-qualification-v2-20261019.json'
PLAN_SHA256 = '6e44cac1d6bfa19c3453f813abaf4613651f8d051d2c2a0f3dd80c35c1121979'

TAIL_S = 20.
ASR_WEIGHT = 'e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb'
SPELLINGS = {'color': 'colour'}


def agreement(reference, transcript):
    from data.synthetic.filter import norm_words, wer
    a, b = norm_words(reference), norm_words(transcript)
    return dict(raw_wer=wer(a,b), spelling_only_wer=wer(
        [SPELLINGS.get(w,w) for w in a], [SPELLINGS.get(w,w) for w in b]),
        spelling_equivalents=SPELLINGS.copy())


def specifications():
    plan = json.loads(PLAN.read_text())
    if (sha256(PLAN) != PLAN_SHA256 or not plan['do_not_train'] or plan['training_approved']
            or len(plan['cases']) != 20 or plan['observation_tail_s'] != TAIL_S
            or [c['id'] for c in plan['cases']] != ['native-fresh2-%03d'%i for i in range(20)]):
        raise RuntimeError('Fixed full20 evaluation plan changed')
    return plan['cases']


def waveform(pcm, rate):
    import numpy as np
    finite = bool(np.isfinite(pcm).all())
    peak = float(np.abs(pcm).max()) if len(pcm) and finite else None
    return dict(sample_rate=rate, samples=len(pcm), finite=finite, peak=peak,
                passed=rate == 24000 and pcm.ndim == 1 and finite
                and len(pcm) >= 2400 and peak is not None and .001 < peak <= .995)


def generate(out):
    import numpy as np
    import soundfile as sf
    from data.synthetic.recovery_curriculum import synthesize, MODEL
    from data.synthetic import recovery_curriculum as producer
    specs = specifications();out.mkdir(parents=True, exist_ok=False);rows=[];by_id={}
    for case in specs:
        folder=out/case['id'];folder.mkdir();prior_end=0.;clips=[];utterances=[];receipts=[]
        paired=case.get('same_user_audio_as')
        if paired:
            parent=next(c for c in specs if c['id']==paired)
            if case['turns'] != parent['turns'] or paired not in by_id:
                raise RuntimeError('Counterfactual inputs must reuse exactly the same user stimulus')
            prior=by_id[paired]
            for item in prior['generation_receipts']:
                for name in (item['file'],Path(item['file']).with_suffix('.receipt.json').name):
                    shutil.copyfile(out/paired/name,folder/name)
            shutil.copyfile(out/paired/'conversation.wav',folder/'conversation.wav')
            shutil.copyfile(out/paired/'reference.json',folder/'reference.json')
            receipts=prior['generation_receipts'];utterances=json.loads((folder/'reference.json').read_text())['utterances']
            duration=prior['duration_s']
        else:
            for i,(nominal,text) in enumerate(case['turns']):
                path=folder/('user-%d.wav'%i);receipt=synthesize(text,'coral',path)
                if receipt['source_model'] != MODEL or receipt['input_text'] != text:
                    raise RuntimeError('Wrong user speech source')
                pcm,rate=sf.read(path,dtype='float32')
                if rate!=24000 or pcm.ndim!=1 or not np.isfinite(pcm).all():
                    raise RuntimeError('Invalid user speech dimensions; retain failed attempt')
                first=math.ceil(max(float(nominal),prior_end+.8)*rate);last=first+len(pcm);prior_end=last/rate
                clips.append((first,pcm));receipts.append(dict(file=path.name,**receipt))
                utterances.append(dict(text=text,start_s=first/rate,end_s=last/rate,source_file=path.name))
            pcm=np.zeros(math.ceil((prior_end+TAIL_S)*24000),dtype=np.float32)
            for first,clip in clips:pcm[first:first+len(clip)]=clip
            sf.write(folder/'conversation.wav',pcm,24000,subtype='FLOAT');duration=len(pcm)/24000
            (folder/'reference.json').write_text(json.dumps(dict(utterances=utterances),indent=2)+'\n')
        row=dict(scenario_id=case['id'],split='fresh-evaluation',path=str(folder/'conversation.wav'),
                 wav_sha256=sha256(folder/'conversation.wav'),reference_path=str(folder/'reference.json'),
                 reference_sha256=sha256(folder/'reference.json'),generation_receipts=receipts,
                 duration_s=duration,utterance_count=len(utterances),same_user_audio_as=paired)
        rows.append(row);by_id[case['id']]=row
    manifest=dict(schema='dl-native-fresh-inputs-v2',rows=rows,plan_sha256=PLAN_SHA256,
                  generator_sha256=sha256(Path(__file__)),speech_producer_sha256=sha256(Path(producer.__file__)),do_not_train=True,training_approved=False,
                  assistant_audio_generated=False,source_model=MODEL,user_voice='coral',
                  raw_samples_preserved=True,gain_changed=False,observation_tail_s=TAIL_S,invalid_inputs='freeze unscored; no reroll or subset')
    (out/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(scenarios=len(rows),unique_clips=36,plan_sha256=PLAN_SHA256)),flush=True)


def load_inputs(data):
    """Verify every20 input, raw receipt, reference and paired stimulus before judging."""
    import numpy as np
    import soundfile as sf
    from data.synthetic import recovery_curriculum as producer
    specs=specifications();m=json.loads((data/'inputs.json').read_text())
    if (m['schema']!='dl-native-fresh-inputs-v2' or m['plan_sha256']!=PLAN_SHA256
            or m['generator_sha256']!=sha256(Path(__file__))
            or m['speech_producer_sha256']!=sha256(Path(producer.__file__)) or not m['do_not_train']
            or m['training_approved'] or m['assistant_audio_generated'] or m['gain_changed']
            or m['source_model'] != 'gpt-4o-mini-tts-2025-12-15' or m['user_voice'] != 'coral'
            or not m['raw_samples_preserved'] or m['observation_tail_s'] != TAIL_S
            or [r['scenario_id'] for r in m['rows']] != [c['id'] for c in specs]):
        raise RuntimeError('Incomplete or changed native full20 input manifest')
    by_id={}
    for case,row in zip(specs,m['rows']):
        wav=Path(row['path']);ref=Path(row['reference_path'])
        if (row['split']!='fresh-evaluation' or wav != data/case['id']/'conversation.wav'
                or ref!=wav.with_name('reference.json') or sha256(wav)!=row['wav_sha256']
                or sha256(ref)!=row['reference_sha256']):raise RuntimeError('Changed native input')
        utterances=json.loads(ref.read_text())['utterances'];pcm,rate=sf.read(wav,dtype='float32')
        if (rate!=24000 or pcm.ndim!=1 or not np.isfinite(pcm).all()
                or len(utterances)!=len(case['turns']) or row['utterance_count']!=len(utterances)
                or len(row['generation_receipts'])!=len(utterances)
                or abs(row['duration_s']-len(pcm)/rate)>1e-9):raise RuntimeError('Invalid native source inventory')
        assembled=np.zeros_like(pcm);prior_end=0.
        for i,(utterance,receipt,turn) in enumerate(zip(utterances,row['generation_receipts'],case['turns'])):
            path=wav.with_name('user-%d.wav'%i);source,sr=sf.read(path,dtype='float32')
            original=json.loads(path.with_suffix('.receipt.json').read_text())
            start=math.ceil(max(float(turn[0]),prior_end+.8)*rate);end=start+len(source);prior_end=end/rate
            if (receipt!=dict(file=path.name,**original) or receipt['sha256']!=sha256(path)
                    or receipt['input_text']!=turn[1] or receipt['source_model']!=m['source_model']
                    or receipt['voice']!=m['user_voice'] or sr!=rate or source.ndim!=1
                    or not np.isfinite(source).all() or end>len(pcm)
                    or utterance!=dict(text=turn[1],start_s=start/rate,end_s=end/rate,source_file=path.name)):
                raise RuntimeError('Changed exact words, schedule, raw speech or receipt')
            assembled[start:end]=source
        if len(pcm) != math.ceil((prior_end+TAIL_S)*rate):
            raise RuntimeError('Declared full observation tail changed')
        if not np.array_equal(assembled,pcm):raise RuntimeError('Raw user samples were changed or gain-adjusted')
        paired=case.get('same_user_audio_as')
        if row['same_user_audio_as']!=paired or (paired and (
                row['wav_sha256']!=by_id[paired]['wav_sha256']
                or row['reference_sha256']!=by_id[paired]['reference_sha256']
                or row['generation_receipts']!=by_id[paired]['generation_receipts'])):
            raise RuntimeError('Counterfactual user waveform changed')
        by_id[case['id']]=row
    return m


def judge(data,out):
    import numpy as np
    import soundfile as sf
    import whisper_timestamped as whisper
    import whisper as package
    from scipy.signal import resample_poly
    from data.synthetic.filter import norm_words,wer
    from evals.moshi_speech_audit import asr_window,restore_clock
    m=load_inputs(data);out.mkdir(parents=True,exist_ok=False);model=whisper.load_model('large-v3',device='cpu');results=[]
    if package._MODELS['large-v3'].split('/')[-2] != ASR_WEIGHT:
        raise RuntimeError('Fixed independent recognizer weights changed')
    for row in m['rows']:
        folder=Path(row['path']).parent;reference=json.loads(Path(row['reference_path']).read_text());clips=[]
        for i,utterance in enumerate(reference['utterances']):
            path=folder/utterance['source_file'];pcm,rate=sf.read(path,dtype='float32');wave=waveform(pcm,rate)
            audio=resample_poly(pcm,2,3).astype(np.float32)
            window,head,tail,silent=asr_window(audio,True,True,True)
            t=dict(text='',segments=[]) if silent else whisper.transcribe(model,window,language='en',vad=False,
                beam_size=5,best_of=5,temperature=0.,verbose=None)
            t=restore_clock(t,head/16000);words=[w for s in t['segments'] for w in s.get('words',[])]
            valid=bool(words) and all(math.isfinite(w['start']) and math.isfinite(w['end'])
                and 0<=w['start']<=w['end']<=len(pcm)/rate+.02 for s in t['segments'] for w in [s]+s.get('words',[]))
            scores=agreement(utterance['text'],t['text']);error=scores['spelling_only_wer']
            name=row['scenario_id']+'-%d-asr.json'%i;(out/name).write_text(json.dumps(t,indent=2)+'\n')
            clips.append(dict(ordinal=i,raw_wav_sha256=sha256(path),asr_file=name,asr_sha256=sha256(out/name),
                wer=error,agreement=scores,valid_bounds=valid,waveform=wave,head_window_samples=head,tail_window_samples=tail,
                original_source_unchanged=True,passed=wave['passed'] and valid and math.isfinite(error) and 0<=error<=.05))
        results.append(dict(scenario_id=row['scenario_id'],wav_sha256=row['wav_sha256'],clips=clips,passed=all(c['passed'] for c in clips)))
        print(json.dumps(dict(scenario=row['scenario_id'],passed=results[-1]['passed'],wer=[c['wer'] for c in clips])),flush=True)
    report=dict(schema='dl-native-fresh-quality-v2',passed=all(r['passed'] for r in results),results=results,
                inputs_manifest_sha256=sha256(data/'inputs.json'),plan_sha256=PLAN_SHA256,threshold=.05,
                peak_bar=.995,judge_sha256=sha256(Path(__file__)),asr='whisper large-v3 CPU',expected_text_prompt_used=False,
                asr_weight_sha256=ASR_WEIGHT,training_approved=False,do_not_train=True,
                invalid_recordings_frozen_unscored=True,no_passing_subset=True)
    (out/'quality.json').write_text(json.dumps(report,indent=2)+'\n')
    # Failed input quality is retained as a complete unscored evaluation, never a smaller pass.
    if not report['passed']:raise RuntimeError('Full20 source quality failed; freeze unscored, no reroll or subset')


def approved_inputs(data,quality_path):
    """Recompute all quality decisions from hash-bound raw ASR; never approve a subset."""
    import soundfile as sf
    from data.synthetic.filter import norm_words,wer
    m=load_inputs(data);q=json.loads(quality_path.read_text())
    if (q['schema']!='dl-native-fresh-quality-v2' or not q['passed']
            or q['inputs_manifest_sha256']!=sha256(data/'inputs.json')
            or q['plan_sha256']!=PLAN_SHA256 or q['threshold']!=.05 or q['peak_bar']!=.995
            or q['judge_sha256']!=sha256(Path(__file__)) or q['training_approved']
            or not q['do_not_train'] or not q['invalid_recordings_frozen_unscored']
            or not q['no_passing_subset'] or q['asr']!='whisper large-v3 CPU'
            or q['asr_weight_sha256']!=ASR_WEIGHT or q['expected_text_prompt_used']
            or [r['scenario_id'] for r in q['results']]!=[r['scenario_id'] for r in m['rows']]):
        raise RuntimeError('Complete native full20 quality approval required')
    for row,result in zip(m['rows'],q['results']):
        folder=Path(row['path']).parent;ref=json.loads(Path(row['reference_path']).read_text())
        if (not result['passed'] or result['wav_sha256']!=row['wav_sha256']
                or len(result['clips'])!=len(ref['utterances'])):raise RuntimeError('Incomplete fresh clip approval')
        for i,(clip,utterance) in enumerate(zip(result['clips'],ref['utterances'])):
            path=folder/utterance['source_file'];pcm,rate=sf.read(path,dtype='float32')
            name=row['scenario_id']+'-%d-asr.json'%i;asr=quality_path.parent/name
            if clip['asr_file']!=name or clip['asr_sha256']!=sha256(asr):raise RuntimeError('Changed source ASR')
            t=json.loads(asr.read_text());words=[w for s in t['segments'] for w in s.get('words',[])]
            valid=bool(words) and all(math.isfinite(w['start']) and math.isfinite(w['end'])
                and 0<=w['start']<=w['end']<=len(pcm)/rate+.02 for s in t['segments'] for w in [s]+s.get('words',[]))
            scores=agreement(utterance['text'],t['text']);error=scores['spelling_only_wer'];wave=waveform(pcm,rate)
            if (clip['ordinal']!=i or clip['raw_wav_sha256']!=sha256(path) or not clip['passed']
                    or clip['valid_bounds']!=valid or not valid or clip['waveform']!=wave or not wave['passed']
                    or clip['agreement']!=scores or clip['wer']!=error or not math.isfinite(error) or not 0<=error<=.05
                    or not clip['original_source_unchanged']):raise RuntimeError('Invalid source word/time/peak approval')
    return m


def inference_input(data,quality_path,identity):
    """Only waveform/routing and worker-only world; no question script or scoring targets."""
    m=approved_inputs(data,quality_path)
    case=next(c for c in specifications() if c['id']==identity)
    row=next(r for r in m['rows'] if r['scenario_id']==identity)
    return dict(scenario=identity,path=row['path'],wav_sha256=row['wav_sha256'],kind=case['kind'],
                backend=case['backend'],worker_only_world=case['world'],
                minimum_lookup_delay_s=case.get('minimum_actual_lookup_delay_s',0.),
                manifest_sha256=sha256(data/'inputs.json'),quality_sha256=sha256(quality_path),
                plan_sha256=PLAN_SHA256,source_split='fresh-evaluation',training_approved=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('generate','judge'))
    ap.add_argument('--data',type=Path);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.mode=='generate':generate(args.out)
    elif args.data:judge(args.data,args.out)
    else:ap.error('judge requires --data')


if __name__=='__main__':main()
