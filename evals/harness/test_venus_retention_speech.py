from collections import Counter
import hashlib,json
from data.venus.retention_speech import prepare,PLAN,MODEL,VOICES


def test_all96_train_sources_and_spoken_targets_have_exact_complete_binding():
    p=prepare();source=json.loads(PLAN.read_text());by={r['id']:r for r in p['requests']}
    assert len(p['requests'])==182 and len(p['scenes'])==96 and p['characters']==12746
    assert p['maximum_additional_cash_charge_usd']==0
    assert p['conservative_cost_upper_usd']==6.373
    for scene,original in zip(p['scenes'],source['scenes']):
        assert scene['id']==original['id'] and scene['fold']=='TRAIN'
        assert [by[x]['body']['text'] for x in scene['user_utterance_ids']]==original['user_turns']
        for target in scene['assistant_spoken_targets']:
            assert by[target['utterance_id']]['body']['text']==original['teacher_program'][target['program_field']]
    assert all(r['body']['model_id']==MODEL for r in by.values())
    assert all(set(r['body'])=={'text','model_id','language_code','seed','voice_settings'} for r in by.values())
    assert all('assistant-british'==by[t['utterance_id']]['profile'] for s in p['scenes'] for t in s['assistant_spoken_targets'])
    assert p['training_approved'] is False


def test_only_existing_british_and_scottish_profiles_no_owner_or_holdout_payloads():
    p=prepare()
    assert {r['voice_id'] for r in p['requests']}==set(VOICES.values())
    assert all(r['fold']=='TRAIN' for r in p['requests'])
    assert not p['owner_conversations'] and not p['human_ratings_sent'] and not p['score_annotations_sent']


def test_interrupted_prefix_is_recovered_only_from_exact_original_audio_and_request_receipt(tmp_path):
    import wave
    import pytest
    from data.venus.retention_speech import inherited_sources,canonical_file_payload
    p={'requests':[dict(id='one',payload_sha256='body',voice_id='voice'),dict(id='two')]}
    folder=tmp_path/'one';folder.mkdir();raw=bytes(480);(folder/'original.pcm').write_bytes(raw)
    with wave.open(str(folder/'original.wav'),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(raw)
    attempt=dict(id='one',payload_sha256='body',voice_id='voice',model_id=MODEL,request_id='actual-fixture-response',status='FAILED_OR_UNKNOWN_NO_RETRY',original_pcm_sha256=hashlib.sha256(raw).hexdigest(),original_wav_sha256=hashlib.sha256((folder/'original.wav').read_bytes()).hexdigest())
    (tmp_path/'ledger.json').write_text(json.dumps(dict(payload_sha256=canonical_file_payload(p),attempts=[attempt])))
    recovered,remaining=inherited_sources(p,tmp_path)
    assert len(recovered)==len(remaining)==1 and remaining[0]['id']=='two'
    assert recovered[0]['original_attempt']['status']=='FAILED_OR_UNKNOWN_NO_RETRY'
    assert recovered[0]['recovered_after_local_bookkeeping_failure']
    (folder/'original.pcm').write_bytes(b'changed')
    with pytest.raises(ValueError):inherited_sources(p,tmp_path)
