"""Fixed, user-only native qualification inputs; never training supervision."""
import argparse
import json
import math
from pathlib import Path
import shutil
from evals.moshi_model_download import sha256

PLAN = Path(__file__).with_name('fixtures')/'native-fulltask-qualification-20261019.json'
PLAN_SHA256 = '162b0e64942dfe1917be9d8b24b579a947d1af54800bf246a1f2d47f3e7d1c21'


def specifications():
    plan = json.loads(PLAN.read_text())
    if (sha256(PLAN) != PLAN_SHA256 or not plan['do_not_train'] or plan['training_approved']
            or len(plan['cases']) != 20
            or [c['id'] for c in plan['cases']] != ['native-fresh-%03d'%i for i in range(20)]):
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
            pcm=np.zeros(math.ceil((prior_end+3.)*24000),dtype=np.float32)
            for first,clip in clips:pcm[first:first+len(clip)]=clip
            sf.write(folder/'conversation.wav',pcm,24000,subtype='FLOAT');duration=len(pcm)/24000
            (folder/'reference.json').write_text(json.dumps(dict(utterances=utterances),indent=2)+'\n')
        row=dict(scenario_id=case['id'],split='fresh-evaluation',path=str(folder/'conversation.wav'),
                 wav_sha256=sha256(folder/'conversation.wav'),reference_path=str(folder/'reference.json'),
                 reference_sha256=sha256(folder/'reference.json'),generation_receipts=receipts,
                 duration_s=duration,utterance_count=len(utterances),same_user_audio_as=paired)
        rows.append(row);by_id[case['id']]=row
    manifest=dict(schema='dl-native-fresh-inputs-v1',rows=rows,plan_sha256=PLAN_SHA256,
                  generator_sha256=sha256(Path(__file__)),do_not_train=True,training_approved=False,
                  assistant_audio_generated=False,source_model=MODEL,user_voice='coral',
                  raw_samples_preserved=True,gain_changed=False,invalid_inputs='freeze unscored; no reroll or subset')
    (out/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(scenarios=len(rows),unique_clips=36,plan_sha256=PLAN_SHA256)),flush=True)


def load_inputs(data):
    """Verify every20 input, raw receipt, reference and paired stimulus before judging."""
    import numpy as np
    import soundfile as sf
    specs=specifications();m=json.loads((data/'inputs.json').read_text())
    if (m['schema']!='dl-native-fresh-inputs-v1' or m['plan_sha256']!=PLAN_SHA256
            or m['generator_sha256']!=sha256(Path(__file__)) or not m['do_not_train']
            or m['training_approved'] or m['assistant_audio_generated'] or m['gain_changed']
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
                    or receipt['voice']!=m['user_voice'] or sr!=rate or end>len(pcm)
                    or utterance!=dict(text=turn[1],start_s=start/rate,end_s=end/rate,source_file=path.name)):
                raise RuntimeError('Changed exact words, schedule, raw speech or receipt')
            assembled[start:end]=source
        if not np.array_equal(assembled,pcm):raise RuntimeError('Raw user samples were changed or gain-adjusted')
        paired=case.get('same_user_audio_as')
        if row['same_user_audio_as']!=paired or (paired and row['wav_sha256']!=by_id[paired]['wav_sha256']):
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
    m=load_inputs(data);out.mkdir(parents=True,exist_ok=False);model=whisper.load_model('turbo',device='cpu');results=[]
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
            error=wer(norm_words(utterance['text']),norm_words(t['text']))
            name=row['scenario_id']+'-%d-asr.json'%i;(out/name).write_text(json.dumps(t,indent=2)+'\n')
            clips.append(dict(ordinal=i,raw_wav_sha256=sha256(path),asr_file=name,asr_sha256=sha256(out/name),
                wer=error,valid_bounds=valid,waveform=wave,head_window_samples=head,tail_window_samples=tail,
                original_source_unchanged=True,passed=wave['passed'] and valid and math.isfinite(error) and 0<=error<=.05))
        results.append(dict(scenario_id=row['scenario_id'],wav_sha256=row['wav_sha256'],clips=clips,passed=all(c['passed'] for c in clips)))
        print(json.dumps(dict(scenario=row['scenario_id'],passed=results[-1]['passed'],wer=[c['wer'] for c in clips])),flush=True)
    report=dict(schema='dl-native-fresh-quality-v1',passed=all(r['passed'] for r in results),results=results,
                inputs_manifest_sha256=sha256(data/'inputs.json'),plan_sha256=PLAN_SHA256,threshold=.05,
                peak_bar=.995,judge_sha256=sha256(Path(__file__)),asr='whisper turbo CPU',
                asr_weight_sha256=package._MODELS['turbo'].split('/')[-2],training_approved=False,do_not_train=True,
                invalid_recordings_frozen_unscored=True,no_passing_subset=True)
    (out/'quality.json').write_text(json.dumps(report,indent=2)+'\n')
    # Failed input quality is retained as a complete unscored evaluation, never a smaller pass.
    if not report['passed']:raise RuntimeError('Full20 source quality failed; freeze unscored, no reroll or subset')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('generate','judge'))
    ap.add_argument('--data',type=Path);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.mode=='generate':generate(args.out)
    elif args.data:judge(args.data,args.out)
    else:ap.error('judge requires --data')


if __name__=='__main__':main()
