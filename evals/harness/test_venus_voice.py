"""Integrity and split checks; these are not claims of humanlike speech."""
import hashlib
import numpy as np
import pytest
import soundfile as sf
from data.venus.voice_curriculum import build
from portable.venus.voice import reference

def test_voice_sources_keep_silence_and_holdout_separate():
    a=build();assert a['split_counts']==dict(TRAIN=48,DEV=12,HOLDOUT=12)
    assert sum(not c['assistant_text'] for c in a['cases'])==6
    assert all(not c['training_approved'] for c in a['cases'])
    assert all(not c['assistant_text'] for c in a['cases'] if c['deliberate_silence'])
    assert all(c['family']=='allowed_backchannel' for c in a['cases'] if c['permitted_acknowledgement'])
    sets=[{c['assistant_text'] for c in a['cases'] if c['split']==s and c['assistant_text']} for s in ['TRAIN','DEV','HOLDOUT']]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])

def test_reference_hash_change_or_clipping_is_rejected_without_audio_repair(tmp_path):
    p=tmp_path/'ref.wav';audio=.2*np.sin(np.arange(48000)*.1);sf.write(p,audio,16000,subtype='FLOAT')
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    a=reference(tmp_path,p,digest);assert a['seconds']==3 and not a['hosted_tts'] and not a['qualified']
    with pytest.raises(ValueError):reference(tmp_path,p,None)
    sf.write(p,audio*.8,16000,subtype='FLOAT')
    with pytest.raises(ValueError):reference(tmp_path,p,digest)
    sf.write(p,np.ones(48000),16000,subtype='FLOAT')
    with pytest.raises(ValueError):reference(tmp_path,p,hashlib.sha256(p.read_bytes()).hexdigest())

def test_reference_runs_on_python310_without_hashlib_file_digest(tmp_path,monkeypatch):
    monkeypatch.delattr(hashlib,'file_digest',raising=False)
    p=tmp_path/'ref.wav';sf.write(p,.1*np.sin(np.arange(48000)*.1),16000,subtype='FLOAT')
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    assert reference(tmp_path,p,digest)['sha256']==digest

def test_voice_comparison_is_the_complete_predeclared_ordinary_set():
    import json
    from pathlib import Path
    from training.venus.voice_compare import CONDITIONS
    a=json.loads(Path('docs/NATIVE-VENUS-VOICE-REFERENCE-EXECUTION-2026-10-27.json').read_text())
    assert list(CONDITIONS)==a['conditions'] and a['cases_per_condition']==8 and a['input_seconds_per_condition']==81
    assert a['focused_prior_gpu_seconds_upper']==200 and a['prior_gpu_seconds_upper']==54451

def test_new_eleven_streams_do_not_claim_minicpm_or_relabel_unpinned_cache():
    from types import SimpleNamespace
    from data.synthetic.provenance import record,cache_namespace
    engine=SimpleNamespace(model='eleven_v3')
    p=record('elevenlabs',engine)
    assert p['provenance_class']=='closed-content' and p['tts_engine']=='ElevenLabs/eleven_v3'
    assert 'open models only' not in p['license'] and 'ADR0003 Amendment A' in p['license']
    assert cache_namespace('elevenlabs',engine)!='elevenlabs'

def test_speech_payloads_exclude_context_scores_and_silence():
    from data.venus.render_voice import payloads,RATE_CEILING,VOICE_BUDGET
    plan=build();rows=payloads(plan)
    assert len(rows)==67 and all(r['payload']['text'] for r in rows)
    assert sum(len(r['payload']['text']) for r in rows)*RATE_CEILING<=VOICE_BUDGET
    for row in rows:
        assert set(row['payload'])=={'text','model_id','language_code','seed','voice_settings'}
        assert not {'user_context','split','expected','scoring'}&row['payload'].keys()

def test_local_guard_readiness_is_never_exposed_before_remote_ack(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from portable.venus.guard import publish_guard
    ready=tmp_path/'ready.json';observed=[]
    def remote(*args,**kwargs):
        observed.append(ready.exists())
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('portable.venus.guard.subprocess.run',remote)
    publish_guard(['ssh','strict-host'],'{"live":true}',ready)
    assert observed==[False] and ready.read_text()=='{"live":true}\n'
    ready.unlink()
    monkeypatch.setattr('portable.venus.guard.subprocess.run',lambda *a,**kw:SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError):publish_guard(['ssh','strict-host'],'{}',ready)
    assert not ready.exists()

def test_nonspoken_actions_preserve_every_ordinary_condition_and_original_history():
    from types import SimpleNamespace
    import torch
    from portable.venus.speech_channel import install
    names={'<delegate>':1,'</delegate>':2};calls=[]
    tokenizer=SimpleNamespace(convert_tokens_to_ids=names.__getitem__,decode=lambda ids,**kw:{1:'<delegate>',2:'</delegate>'}[ids[0]],all_special_ids=[1,2,3])
    def convert(rows):return torch.tensor([[[float(r[0]),float(r[0])+1] for r in rows]]) if rows else torch.zeros(1,1,2)
    def generate(**kw):calls.append(kw);return 'original',kw.get('past_key_values')
    tts=SimpleNamespace(generate_chunk=generate);duplex=SimpleNamespace(tokenizer=tokenizer,model=SimpleNamespace(tts=tts),_convert_results_to_tts_input=convert)
    channel=install(duplex);ordinary=[[10,None,False],[11,None,False],[3,None,True]]
    assert torch.equal(duplex._convert_results_to_tts_input(ordinary),convert(ordinary))
    cache=object();kwargs=dict(inputs_embeds=convert(ordinary),past_key_values=cache,temperature=.8)
    assert tts.generate_chunk(**kwargs)==('original',cache) and len(calls)==1
    first=[[1,None,False],[20,None,False]];copy=[list(r) for r in first]
    hidden=duplex._convert_results_to_tts_input(first);assert hidden.shape==(1,0,2) and first==copy
    ids,returned_cache=tts.generate_chunk(inputs_embeds=hidden,past_key_values=cache)
    assert ids.shape==(1,0) and returned_cache is cache and len(calls)==1
    # A tool spanning units is hidden; speech after its closing tag survives.
    second=[[21,None,False],[2,None,False],[12,None,False]]
    assert torch.equal(duplex._convert_results_to_tts_input(second),convert(second[2:]))
    assert not channel.inside and channel.hidden_tokens==4
    channel.reset();assert channel.hidden_tokens==0 and not channel.inside

def test_word_timed_native_targets_allow_speech_to_finish_before_turn_end():
    from data.venus.pacing import units
    from training.venus.spoken_audit import references
    rows=[dict(text='Glass',timestamp=(.1,.5)),dict(text='lets',timestamp=(.6,.9)),dict(text='light',timestamp=(1.1,1.5)),dict(text='in.',timestamp=(1.6,2.3))]
    result=units('Glass lets light in.',rows)
    assert [r['text'] for r in result]==['Glass lets','light in.','']
    assert [r['final'] for r in result]==[False,False,True]
    with pytest.raises(ValueError):units('Glass lets light in.',rows[:-1])
    ref=references([dict(text='',raw_text='<|speak|>The opposite of<|chunk_eos|></unit>'),dict(text='',raw_text='<|speak|>shallow is deep.<|turn_eos|><|chunk_eos|></unit>')])
    assert ' '.join(ref['intended_user_facing'].split())=='The opposite of shallow is deep.'

def test_native_schedule_uses_measured_duration_instead_of_one_sentence_frame():
    from types import SimpleNamespace
    from training.venus.native_cache import schedule
    from data.venus.pacing import units
    text='Glass lets light in.';aligned=[dict(text=w,timestamp=t) for w,t in zip(text.split(),[(.1,.5),(.6,.9),(1.1,1.5),(1.6,2.3)])]
    duplex=SimpleNamespace(speak_token_id=1,turn_eos_token_id=2,chunk_eos_token_id=3,tokenizer=SimpleNamespace(encode=lambda s,**kw:list(range(10,10+len(s.split())))))
    timeline=dict(frames=6,actions=[dict(frame=2,kind='speak',text=text,final=True,native_speech_units=units(text,aligned))])
    result=schedule(duplex,timeline)
    assert set(result)=={2,3,4} and all(2 not in result[f] for f in [2,3]) and 2 in result[4]

def test_measured_pause_resumes_unsaid_words_instead_of_dropping_them():
    from data.venus.paced_timeline import paced_scene
    from data.venus.pacing import words
    scene=dict(id='pause-fixture',fold='TRAIN',family='pause',user_turns=['instructions','Pause now.','Continue.'],teacher_program=dict(kind='pause_resume',first_text='First, rinse the cup.',remaining_text='Then, dry it.'))
    audio={text:dict(seconds=seconds,id=text,sha256='fixture') for text,seconds in zip(scene['user_turns'],[2,.7,.5])}
    def audit(text,times):return dict(passed=True,word_error_rate=0,peak=.5,source_sha256='fixture',words=[dict(text=w,timestamp=t) for w,t in zip(text.split(),times)])
    source={'First, rinse the cup.':audit('First, rinse the cup.',[(.1,.4),(.6,.9),(1.1,1.3),(1.5,2.2)]),'Then, dry it.':audit('Then, dry it.',[(.1,.4),(.6,.9),(1.1,1.7)])}
    result=paced_scene(scene,audio,None,source);first,stop,resume=result['actions']
    assert words(first['text'])==['first','rinse'] and words(resume['text'])[:2]==['the','cup']
    assert words(first['text']+' '+resume['text'])==words('First, rinse the cup. Then, dry it.')
    assert resume['resumes_unspoken_first_step_without_restart']
    assert first['frame']+len(first['native_speech_units'])==stop['frame']

def test_whisper_word_mode_enables_the_boolean_timestamp_processor():
    from types import SimpleNamespace
    from data.venus.asr import enable_complete_word_timestamps
    recorded=[];model=SimpleNamespace(generate=lambda **kwargs:recorded.append(kwargs))
    enable_complete_word_timestamps(model)
    options=dict(return_timestamps='word',return_token_timestamps=True,return_segments=True)
    model.generate(**options)
    assert options['return_timestamps']=='word' and recorded[0]['return_timestamps'] is True
    assert recorded[0]['return_token_timestamps'] and recorded[0]['return_segments']

def test_distinct_native_conversations_reset_audio_cache_before_preparing(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from portable.venus.native_app import NativeEngine
    monkeypatch.setattr('portable.venus.native_app.ensure_guard',lambda:None)
    events=[];model=SimpleNamespace(audio_past_key_values='old-conversation')
    def reset():model.audio_past_key_values=None;events.append('reset')
    def prepare(**kwargs):
        assert model.audio_past_key_values is None;events.append('prepare')
    model.reset_session=reset
    engine=NativeEngine.__new__(NativeEngine);engine.model=model;engine.action_channel=None
    engine.torch=SimpleNamespace(inference_mode=nullcontext);engine.duplex=SimpleNamespace(prepare=prepare)
    engine.profile=dict(system_prompt='Streaming Omni Conversation.');engine.voice_reference=dict(path='public.wav')
    engine.prepare();model.audio_past_key_values='another-conversation';engine.prepare()
    assert events==['reset','prepare','reset','prepare']

def test_source_repair_never_accepts_copied_audio_when_renderer_writes_nothing(tmp_path,monkeypatch):
    import json,wave
    from types import SimpleNamespace
    from data.venus.repair import render
    original=tmp_path/'original';original.mkdir();ident='example'
    with wave.open(str(original/(ident+'.wav')),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(bytes(32000))
    (original/(ident+'.aiff')).write_bytes(b'original-only')
    digest=hashlib.sha256((original/(ident+'.wav')).read_bytes()).hexdigest()
    (original/'render-receipts.json').write_text(json.dumps(dict(complete=True,utterances=[dict(id=ident,text='The lid.',sha256=digest)])))
    observed=tmp_path/'observed.json';observed.write_text(json.dumps(dict(rows=[dict(id=ident,recognized='The lead.',source_sha256=digest)])))
    def fake(args,**kwargs):
        assert args[0]!='afconvert' and args[0]!='/usr/bin/afconvert'
        return SimpleNamespace(stdout='Samantha             en_US    # hello\n' if args==['say','-v','?'] else '')
    monkeypatch.setattr('data.venus.repair.subprocess.run',fake)
    with pytest.raises(ValueError,match='did not create fresh audio'):
        render(SimpleNamespace(original=str(original),observed=str(observed),output=str(tmp_path/'new'),voice='Samantha',rate=150,emphasize_in=False))


def test_original_voice_source_orthography_preserves_values_and_phonemes():
    from data.venus.voice_source_quality import wer
    assert wer('One hundred and twenty four.','124.')==0
    assert wer('The twenty third of April.','The 23rd of April.')==0
    assert wer('Three pounds and fifty pence.','£3.50.')==0
    assert wer('Mm-hm.','Mm-hmm.')==0
    assert wer('Favourite colours and two kilometres.','Favorite colors and two kilometers.')==0
    assert wer('fifteen','50')>0
    assert wer('room sign','rune sign')>0
    assert wer('warm day','worn day')>0
    assert wer('half past eight','eight thirty')>0
