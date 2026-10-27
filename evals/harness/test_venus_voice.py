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
