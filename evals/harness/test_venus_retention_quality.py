"""Complete-cohort originals, receipt tampering and usable alignment boundaries."""
import hashlib
import json
import wave
import pytest
from data.venus.retention_speech import prepare, inherited_sources, canonical_file_payload, MODEL
from data.venus.retention_quality import inventory, complete_timestamps, critical_keys_match, lookup_targets


def make_sources(tmp_path):
    p=prepare();payload=tmp_path/'payload.json';payload.write_text(json.dumps(p,indent=2)+'\n')
    original=tmp_path/'original';original.mkdir();resumed=tmp_path/'resumed';resumed.mkdir()
    attempts=[];raw=b'\x00\x01'*240
    for index,row in enumerate(p['requests']):
        folder=(original if index<135 else resumed)/row['id'];folder.mkdir()
        pcm=folder/'original.pcm';pcm.write_bytes(raw);wav=folder/'original.wav'
        with wave.open(str(wav),'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(raw)
        attempts.append(dict(id=row['id'],payload_sha256=row['payload_sha256'],voice_id=row['voice_id'],
            model_id=MODEL,request_id='fixture-response-'+str(index),status='COMPLETE',
            original_pcm_sha256=hashlib.sha256(raw).hexdigest(),
            original_wav_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),peak=256/32768,seconds=.01))
    attempts[134]['status']='FAILED_OR_UNKNOWN_NO_RETRY'
    (original/'ledger.json').write_text(json.dumps(dict(payload_sha256=canonical_file_payload(p),attempts=attempts[:135])))
    inherited,_=inherited_sources(p,original)
    for row in inherited:row['original_wav']='/old-machine/path.wav'
    (resumed/'ledger.json').write_text(json.dumps(dict(payload_sha256=canonical_file_payload(p),complete=True,
        attempts=attempts[135:],inherited_verified_sources=inherited)))
    return original,resumed,payload


def test_complete_original182_with_relocated_paths_and_preserved_failed_bookkeeping(tmp_path):
    paths=make_sources(tmp_path);rows=inventory(*paths)
    assert len(rows)==182 and {r['fold'] for r in rows}=={'TRAIN'}
    assert rows[134]['original_attempt_status']=='FAILED_OR_UNKNOWN_NO_RETRY'
    assert all(r['peak_pass'] and r['seconds']==.01 for r in rows)
    original,resumed,payload=paths;ledger=resumed/'ledger.json';data=json.loads(ledger.read_text())
    data['attempts'][0]['payload_sha256']='another-authored-question';ledger.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='exact authored'):inventory(*paths)


def test_cannot_accept_subset_or_tampered_original_audio(tmp_path):
    paths=make_sources(tmp_path);original,resumed,payload=paths
    pcm=next(original.glob('*/original.pcm'));old=pcm.read_bytes();pcm.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Ambiguous'):inventory(*paths)
    pcm.write_bytes(old);ledger=resumed/'ledger.json';data=json.loads(ledger.read_text());data['attempts'].pop()
    ledger.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Complete original135'):inventory(*paths)


def test_complete_word_alignment_rejects_missing_final_end_and_invalid_clocks():
    assert complete_timestamps([dict(text='Hello',timestamp=(0.,.3)),dict(text='world',timestamp=(.3,.9))],1.)
    for chunks in ([],[dict(text='word',timestamp=(.1,None))],[dict(text='word',timestamp=(.2,.1))],
                   [dict(text='word',timestamp=(0.,float('nan')))],[dict(text='word',timestamp=(0.,2.))]):
        assert not complete_timestamps(chunks,1.)


def test_wrong_key_cannot_pass_under_whole_sentence_word_error_allowance():
    from data.venus.context_source_quality import wer
    p=prepare();targets=lookup_targets(p)
    assert len(targets)==48 and all(len(v)==1 for v in targets.values())
    row=next(r for r in p['requests'] if r['id'] in targets)
    key=targets[row['id']][0];reference=row['body']['text']
    wrong=reference.replace(key,'wardrobe')
    assert wer(reference,wrong)<=.05
    assert not critical_keys_match(reference,wrong,[key])
    assert not critical_keys_match(reference,reference+' '+key,[key])
    assert critical_keys_match(reference,reference,[key])
