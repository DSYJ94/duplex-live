import inspect,json,time,hashlib
from pathlib import Path
from types import SimpleNamespace,MethodType
import numpy as np
import pytest
import torch
from evals.harness.test_venus_tts_observer import sampler,FIXTURE as SAMPLER
from portable.venus import speech_completion as completion

FIXTURE=Path('evals/fixtures/venus_duplex_generate_4681004.py')


def duplex(eos_at=60):
    ns=dict(torch=torch,time=time)
    exec(compile(FIXTURE.read_text(),str(FIXTURE),'exec'),ns)
    tts,_=sampler(eos_at);state=SimpleNamespace(tokens=iter([]),resets=0,rendered=[],flushes=[],prefills=0)
    d=SimpleNamespace(model=SimpleNamespace(tts=tts),pending_logits=torch.zeros(1),
        _streaming_generate_count=0,force_listen_count=0,ls_mode='explicit',audio_chunk_idx=0,
        current_turn_ended=True,listen_token_id=1,tts_bos_token_id=2,chunk_eos_token_id=4,
        chunk_terminator_token_ids={1,4},chunk_speak_token_ids={2},turn_terminator_token_ids={3},
        total_ids=[],total_hidden=[],res_ids=[],speak_count=0,device='cpu',current_mode='AUDIO',
        generate_audio=True,tts_text_start_pos=0,tts_past_key_values=None,tts_current_turn_start_time=None,
        tts_temperature=torch.ones(1),tts_repetition_penalty=1.,tts_eos_token=torch.tensor(7),tts_logits_processors=[])
    d.is_session_stop_set=lambda:False;d.is_break_set=lambda:False
    d.tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda text:5,decode=lambda ids,**kw:' '.join(str(v) for v in ids if v>=10))
    d.decoder=SimpleNamespace(embed_token=lambda x:x,feed=lambda x,**kw:(torch.zeros(1),torch.zeros(1,1,2)) if kw.get('return_logits') else None,
        decode=lambda **kw:torch.tensor([next(state.tokens)]),register_unit_end=lambda **kw:None,
        _window_config=SimpleNamespace(sliding_window_mode='none'))
    d._convert_results_to_tts_input=lambda rows:torch.zeros(1,len(rows)+1,2)
    def reset():state.resets+=1
    def render(tokens,prompt,is_last_chunk=False,force_flush=False):
        state.rendered.extend(tokens.flatten().tolist());state.flushes.append(is_last_chunk)
        return np.ones(tokens.shape[1]*8,dtype=np.float32)
    d._generate_waveform_from_tokens=render;d._reset_token2wav_for_new_turn=reset
    d._generate_silence_waveform=lambda:np.zeros(8,dtype=np.float32)
    d.prepare=lambda *a,**kw:None
    d.streaming_generate=MethodType(ns['streaming_generate'],d)
    def frame(ids):
        state.tokens=iter(ids);d.pending_logits=torch.zeros(1);d.audio_chunk_idx+=1;state.prefills+=1
        return d.streaming_generate()
    return d,state,frame


def speaking():return [2,10,11,12,13,14,15,16,17,3,4]


def test_released_path_reproduces_terminal_reset_before_actual_eos():
    d,s,frame=duplex();first=frame(speaking())
    assert first['end_of_turn'] and first['n_tts_tokens']==25 and s.resets==1
    assert frame([1])['is_listen'] and len(s.rendered)==25


def test_pending_terminal_finishes_across_native_listen_frames(monkeypatch):
    d,s,frame=duplex();monkeypatch.setattr(completion,'SOURCE_HASHES',{hashlib.sha256(p.read_bytes()).hexdigest() for p in (FIXTURE,SAMPLER)})
    handle=completion.install(d)
    first=frame(speaking());assert first['n_tts_tokens']==25
    frame([1]);frame([1])
    assert len(s.rendered)==60,'Native sampler reached26 without EOS: retain and drain remaining speech instead of clearing it'
    assert s.resets==1 and s.flushes==[False,False,True] and s.prefills==3


@pytest.fixture
def allowed_sources(monkeypatch):
    monkeypatch.setattr(completion,'SOURCE_HASHES',{hashlib.sha256(p.read_bytes()).hexdigest() for p in (FIXTURE,SAMPLER)})


@pytest.mark.parametrize('eos_at',[0,2,25,26,50,60,198])
def test_resumed_exact_native_sampler_matches_unsegmented_tokens_rng_cache_positions(allowed_sources,eos_at):
    def collect_positions(tts):
        original=tts.model;positions=[]
        def model(**kw):
            positions.extend(kw['position_ids'].flatten().tolist());return original(**kw)
        tts.model=model;return positions
    baseline,base_cache=sampler(eos_at);bp=collect_positions(baseline)
    params=dict(inputs_embeds=torch.zeros(1,10,2),temperature=torch.ones(1),repetition_penalty=1.,eos_token=torch.tensor(7),max_new_token=200,min_new_tokens=0,force_no_stop=False,past_key_values=None,text_start_pos=0)
    torch.manual_seed(42);expected,cache=baseline.generate_chunk(**params);rng=torch.get_rng_state()
    actual,actual_cache=sampler(eos_at);ap=collect_positions(actual);torch.manual_seed(42)
    pieces=list(completion._resumable(actual.generate_chunk)(**params))
    assert torch.equal(torch.cat([x[0] for x in pieces],dim=1),expected)
    assert torch.equal(torch.get_rng_state(),rng) and ap==bp
    assert cache is base_cache and all(x[1] is actual_cache for x in pieces)
    assert all(x[0].shape[1]==25 and not x[2] for x in pieces[:-1]) and pieces[-1][2]


@pytest.mark.parametrize('eos_at',[0,2,25])
def test_normal_terminal_eos_retains_released_waveform_state_and_rng(allowed_sources,eos_at):
    b,bs,bframe=duplex(eos_at);torch.manual_seed(42);expected=bframe(speaking());rng=torch.get_rng_state()
    d,s,frame=duplex(eos_at);completion.install(d);torch.manual_seed(42);actual=frame(speaking())
    assert torch.equal(torch.get_rng_state(),rng)
    assert actual['text']==expected['text'] and actual['is_listen']==expected['is_listen'] and actual['end_of_turn']==expected['end_of_turn']
    assert np.array_equal(actual['audio_waveform'],expected['audio_waveform'])
    assert d.total_ids==b.total_ids and s.resets==bs.resets==1 and s.flushes==bs.flushes==[True]
    assert d.tts_text_start_pos==b.tts_text_start_pos==0 and d.tts_past_key_values is b.tts_past_key_values is None
    assert not actual['speech_completion']['pending'] and actual['speech_completion']['acoustic_complete']


def test_explicit_stop_clears_pending_without_resuming_sampler(allowed_sources):
    d,s,frame=duplex();h=completion.install(d);frame(speaking());pending=h.pending
    before=len(s.rendered);d.is_break_set=lambda:True;r=frame([1])
    assert len(s.rendered)==before and pending.gi_frame is None and h.pending is None
    assert not np.any(r['audio_waveform']) and r['speech_completion']['events'][0]['reason']=='native_session_stop_or_break'


def test_native_new_speech_cancels_old_pending_and_starts_from_empty_cache(allowed_sources):
    d,s,frame=duplex();h=completion.install(d);frame(speaking());pending=h.pending
    assert d.tts_text_start_pos>0 and d.tts_past_key_values is not None
    original=h.resumable;calls=[]
    def capture(**kwargs):calls.append(kwargs);return original(**kwargs)
    h.resumable=capture
    result=frame([2,19,3,4])
    assert pending.gi_frame is None and result['speech_completion']['events'][0]['reason']=='new_native_speaking_turn'
    assert calls[-1]['past_key_values'] is None and calls[-1]['text_start_pos']==0 and calls[-1]['inputs_embeds'].shape[1]==3


def test_session_reset_and_missing_input_do_not_release_stale_speech(allowed_sources):
    d,s,frame=duplex();h=completion.install(d);frame(speaking());pending=h.pending
    assert d.pending_logits is None
    before=len(s.rendered);result=d.streaming_generate()
    assert len(s.rendered)==before and result['speech_completion']['pending']
    d.prepare();assert h.pending is None and pending.gi_frame is None
    assert d.tts_past_key_values is None and d.tts_text_start_pos==0
    result=frame([1]);assert len(s.rendered)==before and not result['speech_completion']['pending']


def test_missing_acoustic_eos_is_bounded_explicit_failure_not_success(allowed_sources):
    d,s,frame=duplex(999);h=completion.install(d);frame(speaking())
    with pytest.raises(RuntimeError,match='incomplete speech is not accepted'):
        for _ in range(10):frame([1])
    assert h.pending is None and d.tts_past_key_values is None and len(s.rendered)==175


def test_unmodified_nonterminal_path_keeps_native_tokens_and_rng(allowed_sources):
    b,bs,bframe=duplex();torch.manual_seed(42);expected=bframe([2,10,4]);rng=torch.get_rng_state()
    d,s,frame=duplex();completion.install(d);torch.manual_seed(42);actual=frame([2,10,4])
    assert np.array_equal(actual['audio_waveform'],expected['audio_waveform']) and torch.equal(torch.get_rng_state(),rng)
    assert d.total_ids==b.total_ids and d.tts_text_start_pos==b.tts_text_start_pos and s.resets==bs.resets==0
    assert not actual['speech_completion']['pending']


def test_unknown_native_source_and_duplicate_install_are_rejected(allowed_sources,monkeypatch):
    d,s,frame=duplex();original=d.streaming_generate
    monkeypatch.setattr(completion,'SOURCE_HASHES',set())
    with pytest.raises(ValueError,match='pinned'):completion.install(d)
    assert d.streaming_generate is original and not hasattr(d,'_speech_completion')
    monkeypatch.setattr(completion,'SOURCE_HASHES',{hashlib.sha256(p.read_bytes()).hexdigest() for p in (FIXTURE,SAMPLER)})
    completion.install(d)
    with pytest.raises(ValueError,match='already'):completion.install(d)


def test_actual_native_frame_export_retains_pending_audio_and_linguistic_listen(monkeypatch):
    import base64
    from evals.harness.test_venus_stage_timing import engine
    audio=np.linspace(-.1,.1,24000,dtype=np.float32)
    e=engine(monkeypatch,True,audio);original=e.duplex.streaming_generate
    metadata=dict(version='native-terminal-resume-v1',pending=True,acoustic_complete=False,events=[dict(kind='terminal_acoustic_piece')],qualified=False)
    e.duplex.streaming_generate=lambda **kw:dict(original(**kw),speech_completion=metadata)
    result=e.step(bytes(32000),None)
    assert result['is_listen'] and result['speech_completion']==metadata
    assert base64.b64decode(result['audio'])==audio.astype('<f4').tobytes() and not result['wire_silence_omitted']



def test_native_entrypoint_cannot_enable_changed_runtime_under_consumed_old_lease(tmp_path,monkeypatch):
    from portable.venus import native_app
    monkeypatch.setattr(native_app,'ensure_guard',lambda:None)
    p=tmp_path/'lease.json';p.write_text(json.dumps(dict(resource_key='venus-tts-placement-v56')))
    monkeypatch.setenv('DUPLEX_LIVE_LEASE_FILE',str(p))
    with pytest.raises(ValueError,match='own approved'):
        native_app.NativeEngine('no-model-loaded',speech_completion=True)
    monkeypatch.delenv('DUPLEX_LIVE_LEASE_FILE')
    with pytest.raises(ValueError,match='execution lease'):
        native_app.NativeEngine('no-model-loaded',speech_completion=True)
