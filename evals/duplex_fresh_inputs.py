"""New synthetic evaluation inputs for the hybrid, never training supervision.

Only user clips are generated. Text references stay in separate files, first
read by the independent quality judge and after causal inference for scoring.
No assistant script, record value or expected route is supplied to recognition.
"""
import argparse
import json
import math
from pathlib import Path
from evals.moshi_model_download import sha256


def specifications():
    # These new scenes are an evaluation of the frozen development candidate.
    # Once a result informs a change, it is development evidence for that change.
    return [
        dict(id='hybrid-fresh-000',world={},turns=[(.8,'Why does a window sometimes fog up on the inside?')]),
        dict(id='hybrid-fresh-001',world={'Harbor East':{'delivery window':'Tuesday afternoon'}},turns=[
            (.8,'Please check the delivery window for Project Harbor East.'),
            (9.,'While that is running, how many corners does a rectangle have?')]),
        dict(id='hybrid-fresh-002',world={'Quartz West':{'access code':'falcon nine'}},turns=[
            (.8,'Can you read out the access code?'),(8.,'Project Quartz West.')]),
        dict(id='hybrid-fresh-003',world={'Summit North':{'courier note':'leave with reception'}},turns=[
            (.8,'Look up the courier note for Project Summit North.'),
            (8.,'While you are checking, what is the opposite of clockwise?'),
            (16.,'Cancel the courier lookup, but finish your answer.')]),
        dict(id='hybrid-fresh-004',world={'Summit North':{'courier note':'leave with reception'}},turns=[
            (.8,'Do not look up Project Summit North. Just tell me why a shadow moves during the day.')]),
        dict(id='hybrid-fresh-005',world={},turns=[
            (.8,'Explain in a couple of sentences why the moon seems to change shape.'),(10.,'Mm-hm.')]),
        dict(id='hybrid-fresh-006',world={'Meadow East':{'meeting room':'room seven'},'Copper West':{'meeting room':'room twelve'}},turns=[
            (.8,'Please check the meeting room for Project Meadow East.'),
            (8.,'Actually I meant Project Copper West. Do not read out the Meadow result.')]),
        dict(id='hybrid-fresh-007',world={'Oak Central':{'meeting room':'room seven'},'Pine Central':{'meeting room':'room twelve'}},turns=[
            (.8,'Please check the meeting room for Project Oak Central.'),
            (8.,'Actually use Project Pine Central instead. Do not read out the Oak result.')]),
        dict(id='hybrid-fresh-008',world={},turns=[
            (.8,'I have three boxes with four pencils in each box. If I give away two pencils, how many do I have left?')]),
        dict(id='hybrid-fresh-009',world={'River East':{'delivery day':'Friday morning'}},turns=[
            (.8,'Check the delivery day for Project River East.'),
            (8.,'While you check, explain why ice floats on water.')]),
        dict(id='hybrid-fresh-010',world={'River East':{'storage colour':'orange'},'River West':{'storage colour':'purple'}},turns=[
            (.8,'What is the storage colour?'),(8.,'Project River West.')]),
        dict(id='hybrid-fresh-011',world={'Forest North':{'courier note':'use the side entrance'}},turns=[
            (.8,'Check the courier note for Project Forest North.'),
            (8.,'While you check, explain in two sentences why a mirror reverses an image.'),
            (17.,'Cancel the courier lookup, but finish your explanation.')]),
        dict(id='hybrid-fresh-012',world={'River East':{'delivery day':'Monday morning'},'River West':{'delivery day':'Thursday afternoon'}},turns=[
            (.8,'Check the delivery day for Project River East.'),
            (8.,'Actually use Project River West instead. Do not read out the River East result.')]),
        dict(id='hybrid-fresh-013',world={'Forest North':{'courier note':'use the side entrance'}},turns=[
            (.8,'Do not check Project Forest North. If all robins are birds, does that mean all birds are robins? Explain briefly.')]),
        dict(id='hybrid-fresh-014',world={},turns=[
            (.8,'Explain in three sentences how clouds produce rain.'),(10.,'Okay.')]),
        dict(id='hybrid-fresh-015',world={},turns=[
            (.8,'Explain in four sentences why seasons change during the year.'),
            (11.,'Stop. Give me just the main reason in one short sentence.')]),
        dict(id='hybrid-fresh-016',world={'River East':{'storage label':'green circle'},'River West':{'storage label':'orange star'}},turns=[
            (.8,'What is the storage label?'),(8.,'Project River West.')]),
        dict(id='hybrid-fresh-017',world={},turns=[
            (.8,'Explain in four short sentences how a seed grows into a plant.'),(10.,'Right.')]),
        dict(id='hybrid-fresh-018',world={},turns=[
            (.8,'Explain in four short sentences how a seed grows into a plant.'),
            (12.,'Stop. I only want the first step, in one short sentence.')]),
        dict(id='hybrid-fresh-019',world={'Lake North':{'map label':'silver square'}},turns=[
            (.8,'Check the map label for Project Lake North.'),
            (8.,'While you check, why does a metal spoon feel colder than a wooden spoon in the same room?')]),
        dict(id='hybrid-fresh-020',world={'River East':{'delivery day':'Monday morning'},'River West':{'delivery day':'Thursday afternoon'}},turns=[
            (.8,'Check the delivery day for Project River East.'),
            (8.,'Also check the delivery day for Project River West.'),
            (15.,'Cancel only the River East lookup. Keep checking River West.')]),
        dict(id='hybrid-fresh-021',world={},turns=[
            (.8,'Explain in four sentences why the ocean has tides.'),
            (11.,'Forget the explanation. What is two plus three?')]),
        dict(id='hybrid-fresh-022',world={},turns=[
            (.8,'My favourite animal is a dolphin. Please remember that.'),
            (9.,'What is my favourite animal?')]),
        dict(id='hybrid-fresh-023',world={},turns=[
            (.8,'What is the access code for Project Lake South?')]),
    ]


def generate(out,cases=None,user_voice="verse",backchannel_wording="original"):
    import numpy as np
    import soundfile as sf
    from data.synthetic.recovery_curriculum import synthesize,MODEL
    out.mkdir(parents=True,exist_ok=False);rows=[]
    for spec in specifications():
        if cases and spec['id'] not in cases:continue
        if spec['id']=='hybrid-fresh-005' and backchannel_wording=='uh-huh':spec['turns'][-1]=(10.,'Uh-huh.')
        folder=out/spec['id'];folder.mkdir();clips=[];references=[];receipts=[];prior_end=0.
        for i,(nominal,text) in enumerate(spec['turns']):
            path=folder/('user-'+str(i)+'.wav');receipt=synthesize(text,user_voice,path)
            pcm,rate=sf.read(path,dtype='float32')
            if rate!=24000 or pcm.ndim!=1 or not np.isfinite(pcm).all():raise RuntimeError('Invalid fresh user clip')
            # Preserve every source sample; if TTS exceeds the nominal gap, move
            # the next utterance after the actual end plus800ms, never overlap.
            first=math.ceil(max(nominal,prior_end+.8)*rate);last=first+len(pcm);prior_end=last/rate
            clips.append((first,pcm));receipts.append(dict(file=path.name,**receipt))
            references.append(dict(text=text,start_s=first/rate,end_s=last/rate,source_file=path.name))
        pcm=np.zeros(math.ceil((prior_end+3.)*24000),dtype=np.float32)
        for first,clip in clips:pcm[first:first+len(clip)]=clip
        wav=folder/'conversation.wav';sf.write(wav,pcm,24000,subtype='FLOAT')
        ref=folder/'reference.json';ref.write_text(json.dumps(dict(user_words=[dict(text=r['text']) for r in references],utterances=references),indent=2)+'\n')
        rows.append(dict(scenario_id=spec['id'],split='fresh-evaluation',path=str(wav),reference_path=str(ref),
            wav_sha256=sha256(wav),reference_sha256=sha256(ref),world=spec['world'],
            duration_s=len(pcm)/24000,utterance_count=len(references),user_voice=user_voice,source_model=MODEL,generation_receipts=receipts))
    manifest=dict(schema='dl-hybrid-fresh-inputs-v1',rows=rows,training_approved=False,do_not_train=True,
        generator_sha256=sha256(Path(__file__)),generation_options=dict(cases=cases,user_voice=user_voice,backchannel_wording=backchannel_wording),
        assistant_audio_generated=False,source='new synthetic evaluation user speech')
    (out/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)


def judge(data,out):
    import numpy as np
    import soundfile as sf
    import whisper_timestamped as whisper
    import whisper as whisper_package
    from scipy.signal import resample_poly
    from data.synthetic.filter import norm_words,wer
    manifest=data/'inputs.json';m=json.loads(manifest.read_text());model=whisper.load_model('turbo',device='cuda');results=[]
    if m['training_approved'] or not m['do_not_train'] or m['assistant_audio_generated'] or m['generator_sha256']!=sha256(Path(__file__)):raise RuntimeError('Invalid fresh input scope')
    out.mkdir(parents=True,exist_ok=False)
    for row in m['rows']:
        wav=Path(row['path']);ref=Path(row['reference_path'])
        if sha256(wav)!=row['wav_sha256'] or sha256(ref)!=row['reference_sha256']:raise RuntimeError('Changed fresh source')
        pcm,rate=sf.read(wav,dtype='float32');reference=json.loads(ref.read_text());clips=[]
        for i,utterance in enumerate(reference['utterances']):
            first=round(utterance['start_s']*rate);last=round(utterance['end_s']*rate);clip=pcm[first:last]
            g=math.gcd(rate,16000);audio=resample_poly(clip,16000//g,rate//g).astype(np.float32)
            transcription=whisper.transcribe(model,audio,language='en',vad=False,beam_size=5,best_of=5,temperature=0.,verbose=None)
            error=wer(norm_words(utterance['text']),norm_words(transcription['text']))
            bounds=all(0<=p['start']<=p['end']<=len(audio)/16000+.02 for s in transcription['segments'] for p in [s]+s.get('words',[]))
            (out/(row['scenario_id']+'-'+str(i)+'-whisper.json')).write_text(json.dumps(transcription,indent=2)+'\n')
            clips.append(dict(utterance=i,reference=utterance['text'],transcript=transcription['text'],wer=error,valid_bounds=bounds,passed=error<=.05 and bounds))
        results.append(dict(scenario_id=row['scenario_id'],passed=all(c['passed'] for c in clips),clips=clips,
            wav_sha256=row['wav_sha256'],reference_sha256=row['reference_sha256']))
    report=dict(schema='dl-hybrid-fresh-quality-v1',passed=all(r['passed'] for r in results),results=results,
        inputs_manifest_sha256=sha256(manifest),training_approved=False,do_not_train=True,
        judge_sha256=sha256(Path(__file__)),threshold=.05,asr_model='whisper turbo',
        asr_publisher_weight_sha256=whisper_package._MODELS['turbo'].split('/')[-2])
    (out/'quality.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


def checked_cases(data,quality_path,identities):
    manifest=data/'inputs.json';m=json.loads(manifest.read_text());quality=json.loads(quality_path.read_text())
    if (m['schema']!='dl-hybrid-fresh-inputs-v1' or quality['schema']!='dl-hybrid-fresh-quality-v1'
            or quality['inputs_manifest_sha256']!=sha256(manifest) or m['training_approved']
            or quality['training_approved'] or not m['do_not_train'] or not quality['do_not_train']
            or m['assistant_audio_generated'] or quality['threshold']!=.05 or quality['asr_model']!='whisper turbo'
            or m['generator_sha256']!=sha256(Path(__file__))
            or quality['judge_sha256']!=sha256(Path(__file__))):
        raise RuntimeError('Fresh evaluation lacks an exact independent input quality receipt')
    if (len({r['scenario_id'] for r in m['rows']})!=len(m['rows'])
            or len({r['scenario_id'] for r in quality['results']})!=len(quality['results'])
            or {r['scenario_id'] for r in m['rows']}!={r['scenario_id'] for r in quality['results']}):
        raise RuntimeError('Incomplete or duplicated fresh input quality inventory')
    if len(set(identities))!=len(identities):raise ValueError('Duplicate fresh input selection')
    selected=[]
    for identity in identities:
        row=next(r for r in m['rows'] if r['scenario_id']==identity)
        approved=next(r for r in quality['results'] if r['scenario_id']==identity)
        if (not approved['passed'] or row['split']!='fresh-evaluation'
                or len(approved['clips'])!=row['utterance_count'] or row['utterance_count']<1
                or any(not c['passed'] or not math.isfinite(c['wer']) or not 0<=c['wer']<=.05 or not c['valid_bounds'] for c in approved['clips'])
                or approved['wav_sha256']!=row['wav_sha256'] or approved['reference_sha256']!=row['reference_sha256']
                or sha256(Path(row['path']))!=row['wav_sha256'] or sha256(Path(row['reference_path']))!=row['reference_sha256']):
            raise RuntimeError('Unapproved or changed fresh evaluation input: '+identity)
        selected.append(row)
    return selected


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('generate','judge'))
    ap.add_argument('--data',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--case',action='append',choices=[s['id'] for s in specifications()])
    ap.add_argument('--user-voice',choices=('verse','coral'),default='verse')
    ap.add_argument('--backchannel-wording',choices=('original','uh-huh'),default='original');args=ap.parse_args()
    if args.mode=='generate':generate(args.out,args.case,args.user_voice,args.backchannel_wording)
    else:judge(args.data,args.out)


if __name__=='__main__':main()
