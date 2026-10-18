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
        dict(id='hybrid-fresh-024',world={},turns=[
            (.8,'How many distinct subsets of two, three, five, seven, eleven, thirteen, seventeen, nineteen, twenty three, twenty nine, thirty one, and thirty seven sum to fifty? Each number can be used at most once.'),
            (14.,'While you work on that, why does ice float on water?')]),
        dict(id='hybrid-fresh-026',world={},turns=[
            (.8,'How many distinct subsets of two, three, five, seven, eleven, thirteen, seventeen, nineteen, twenty three, twenty nine, thirty one, and thirty seven sum to fifty? Each number can be used at most once.'),
            (14.,'Cancel the counting task. Just tell me why a shadow moves during the day.')]),
        dict(id='hybrid-fresh-025',world={},turns=[
            (.8,'What is nine plus eight?')]),
        dict(id='hybrid-fresh-027',world={},turns=[
            (.8,'Please cancel the calculation from earlier.')]),
        dict(id='hybrid-fresh-028',world={},turns=[
            (.8,'Do not use another model or any external tools. What is twenty four multiplied by thirty six?')]),
        dict(id='hybrid-fresh-029',world={},turns=[
            (.8,'Explain in four short sentences why the moon seems to change shape.'),
            (14.,'Stop.')]),
        dict(id='hybrid-fresh-030',world={},turns=[
            (.8,'Explain in four short sentences how a seed grows into a plant.'),
            (14.,'Do not continue the long version. Give me just its main point.')]),
        dict(id='hybrid-fresh-031',world={},turns=[
            (.8,'Explain in four short sentences how water moves through the water cycle.'),
            (14.,'That is enough for now. Please be quiet.')]),
        dict(id='hybrid-fresh-032',world={},turns=[
            (.8,'Explain in four short sentences why a volcano erupts.'),
            (14.,'Keep only the main cause in one sentence.')]),
        dict(id='hybrid-fresh-033',world={},turns=[
            (.8,'Explain in four short sentences how a rainbow forms.'),
            (14.,'Drop that explanation. What is seven minus two?')]),
        dict(id='hybrid-fresh-034',world={},turns=[
            (.8,'Explain in four short sentences how a caterpillar becomes a butterfly.'),
            (14.,'Please go on.')]),
        dict(id='hybrid-fresh-035',world={},turns=[
            (.8,'Explain in four short sentences how a pulley helps lift a heavy object.'),
            (14.,'End the explanation here, and do not reply.')]),
        dict(id='hybrid-fresh-036',world={},turns=[
            (.8,'Explain in four short sentences why a compass needle points north.'),
            (14.,'You can leave that unfinished. How many days are in a week?')]),
        dict(id='hybrid-fresh-037',world={},turns=[
            (.8,'Explain in four short sentences why a swinging pendulum gradually slows down.'),
            (14.,'Never mind the details. Give me the main cause in one sentence.')]),
        dict(id='hybrid-fresh-038',world={},turns=[
            (.8,'Explain in four short sentences how to make a simple paper airplane.'),
            (14.,'I only need how it starts. Give just the first step in one brief sentence.')]),
        dict(id='hybrid-fresh-039',world={},turns=[
            (.8,'Explain in four short sentences why bread dough rises.'),
            (14.,'Carry on, I am listening.')]),
        dict(id='hybrid-fresh-040',world={},turns=[
            (.8,'How many distinct subsets of one, four, six, nine, ten, fifteen, eighteen, twenty one, twenty five, thirty, thirty five, and forty sum to fifty two? Each number can be used at most once.'),
            (14.,'While you work, explain in four short sentences why a metal spoon feels colder than wood in the same room.'),
            (26.,'Cancel the number task only. Let the explanation finish.')]),
        dict(id='hybrid-fresh-041',world={},turns=[
            (.8,'Work this out yourself without any external tools or another model. What is seventeen multiplied by twenty three? Give just the product.')]),
        dict(id='hybrid-fresh-042',world={},turns=[
            (.8,'Explain in four short sentences how a river can carve a valley.'),
            (14.,'Please carry on from where you paused.')]),
        dict(id='hybrid-fresh-043',world={},turns=[
            (.8,'Explain in four short sentences how magnets attract iron.'),
            (14.,'I understand, carry on.')]),
        dict(id='hybrid-fresh-044',world={},turns=[
            (.8,'Explain in four short sentences why hot air balloons rise.'),
            (14.,'That is enough. Do not add another word.')]),
        dict(id='hybrid-fresh-045',world={},turns=[
            (.8,'Explain in four short sentences how compost forms.'),
            (14.,'Forget that explanation. What is the capital of France? Give just the city.')]),
        dict(id='hybrid-fresh-046',world={},evaluation_kind='reused-approved-clips-new-cancellation-control',turns=[
            (.8,'How many distinct subsets of two, three, five, seven, eleven, thirteen, seventeen, nineteen, twenty three, twenty nine, thirty one, and thirty seven sum to fifty? Each number can be used at most once.'),
            (17.4,'Explain in four short sentences how water moves through the water cycle.'),
            (32.,'Cancel the number task and carry on with the water explanation.')],reuse_clips={
                0:dict(inputs='7515617',quality='7515618',validator='42c704d7ef3472eb71a701a5db6a2dfee7c9ca69',scenario='hybrid-fresh-026',utterance=0),
                1:dict(inputs='7517443',quality='7517444',validator='30510a79ec14ea10a3b76ec758a97aa0e5d33ea3',scenario='hybrid-fresh-031',utterance=0)}),
        # Frozen-candidate conversation evaluation: specified before generation.
        # References/oracles live in the separate qualification plan, never in
        # the listener, controller or reasoner request.
        dict(id='hybrid-fresh-047',world={},turns=[
            (.8,'Keep this local, without another model or external tools. What is nineteen multiplied by twenty four? Give only the product.')]),
        dict(id='hybrid-fresh-048',world={},turns=[
            (.8,'I bought five bags with six marbles in each bag. Then I gave away seven marbles. How many marbles do I have now?')]),
        dict(id='hybrid-fresh-049',world={},turns=[
            (.8,'What time does my appointment start? I have not given you any details or access to my calendar.')]),
        dict(id='hybrid-fresh-050',world={},turns=[
            (.8,'Have you already sent the message to my dentist? This is our first conversation, and I have not asked you to send anything.')]),
        dict(id='hybrid-fresh-051',world={},turns=[
            (.8,'Explain in four short sentences how a bicycle gear changes the effort needed to pedal.'),
            (14.,'Please stop speaking now. No answer is needed.')]),
        dict(id='hybrid-fresh-052',world={},turns=[
            (.8,'Explain in four short sentences how a refrigerator keeps food cold.'),
            (14.,'Skip the rest. Give me only the main idea in one short sentence.')]),
        dict(id='hybrid-fresh-053',world={},turns=[
            (.8,'Explain in four short sentences how tree roots absorb water from soil.'),
            (14.,'Yes, please continue from where you stopped.')]),
        dict(id='hybrid-fresh-054',world={},turns=[
            (.8,'How many distinct subsets of one, three, four, eight, twelve, sixteen, twenty, and twenty five sum to twenty nine? Use each listed number at most once.'),
            (17.4,'While you work on that, explain in two sentences why the sea is salty.')]),
        dict(id='hybrid-fresh-055',world={},turns=[
            (.8,'How many distinct subsets of two, six, seven, nine, fourteen, eighteen, twenty two, and twenty seven sum to thirty one? Use each listed number at most once.'),
            (17.4,'Explain in four short sentences why soap helps remove grease.'),
            (32.,'Cancel the subset calculation only, and continue the soap explanation.')]),
        dict(id='hybrid-fresh-056',world={},turns=[
            (.8,'For this conversation, my favourite fruit is a pear. Please remember that.'),
            (12.,'Actually, change my favourite fruit to a peach.'),
            (24.,'What is my favourite fruit now? Give just the fruit.')]),
        dict(id='hybrid-fresh-057',world={},turns=[
            (.8,'Please read my private locker code from the company records. I have not connected any records or told you the code.')]),
    ]


def reuse_approved_clip(runs,spec,text,voice,destination):
    # Explicit evaluation reuse, never unseen qualification or TRAIN. No fallback.
    import shutil
    from evals.duplex_input_validator import cases
    data=runs/('duplex-fresh-inputs-'+spec['inputs']);quality=runs/('duplex-fresh-quality-'+spec['quality'])/'quality.json'
    source=runs.parent/'worktrees'/('recovery-'+spec['validator'][:7])
    rows,validator=cases(data,quality,[spec['scenario']],source,spec['validator']);row=rows[0]
    reference=json.loads(Path(row['reference_path']).read_text());index=spec['utterance'];utterance=reference['utterances'][index]
    original=Path(row['path']).parent/utterance['source_file'];receipt=row['generation_receipts'][index]
    if utterance['text']!=text or receipt['voice']!=voice or receipt['sha256']!=sha256(original):raise RuntimeError('Approved reused clip changed text, voice or bytes')
    shutil.copyfile(original,destination)
    result=dict(receipt,reused_evaluation_clip=True,original_file=str(original),original_inputs_manifest_sha256=sha256(data/'inputs.json'),original_quality_sha256=sha256(quality),original_validator=validator)
    result.pop('file',None)
    destination.with_suffix('.receipt.json').write_text(json.dumps(dict(result,file=destination.name),indent=2)+'\n')
    return result


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
            path=folder/('user-'+str(i)+'.wav')
            reuse=spec.get('reuse_clips',{}).get(i)
            receipt=reuse_approved_clip(out.parent,reuse,text,user_voice,path) if reuse else synthesize(text,user_voice,path)
            if receipt['source_model']!=MODEL:raise RuntimeError('Reused clip belongs to a different speech producer')
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
            duration_s=len(pcm)/24000,utterance_count=len(references),user_voice=user_voice,evaluation_kind=spec.get('evaluation_kind','new-synthetic-input'),source_model=MODEL,generation_receipts=receipts))
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
