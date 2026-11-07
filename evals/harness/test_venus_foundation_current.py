"""All32 reference identity and source preservation; no model execution."""
import copy
import json
import pytest
from training.venus.foundation_compare import validate_current_reference,sha


def fixture(root):
    profile=dict(optimizer_updates=646,checkpoint_sha256='checkpoint',nonspoken_action_channel=True,
                 speech_completion=dict(enabled=True))
    cases=[];speech=[];results=[]
    for i in range(32):
        ident='case'+str(i);folder=root/ident;folder.mkdir();(folder/'joined-emitted.wav').write_bytes(b'original');(folder/'chunk.wav').write_bytes(b'chunk')
        cases.append(dict(id=ident,input_sha256=str(i),frames=2,family='ordinary'))
        results.append(dict(id=ident,input_sha256=str(i),input_frames=2,family='ordinary'))
        speech.append(dict(id=ident,joined_wav_sha256=sha(folder/'joined-emitted.wav'),source_chunks=[dict(file='chunk.wav',sha256=sha(folder/'chunk.wav'))]))
    (root/'profile.json').write_text(json.dumps(profile));(root/'results.json').write_text(json.dumps(dict(cases=results)))
    (root/'native-report.json').write_text(json.dumps(dict(complete=True,cases=speech,source='source',word_limit=.05,peak_limit=.995)))
    manifest=dict(schema='dl-current-runtime-foundation32-inputs-v1',required_cases=32,reference_source='source',reference_checkpoint_sha256='checkpoint',
                  reference_collection_sha256=sha(root/'results.json'),reference_speech_report_sha256=sha(root/'native-report.json'),reference_profile_sha256=sha(root/'profile.json'))
    return manifest,cases


def test_all32_reference_required_and_wrong_inputs_or_raw_audio_cannot_be_substituted(tmp_path):
    manifest,cases=fixture(tmp_path)
    assert validate_current_reference(manifest,tmp_path,cases)['optimizer_updates']==646
    with pytest.raises(ValueError,match='All32'):validate_current_reference(manifest,tmp_path,cases[:-1])
    wrong=copy.deepcopy(cases);wrong[0]['input_sha256']='different'
    with pytest.raises(ValueError,match='input differs'):validate_current_reference(manifest,tmp_path,wrong)
    (tmp_path/'case0/chunk.wav').write_bytes(b'changed')
    with pytest.raises(ValueError,match='chunk changed'):validate_current_reference(manifest,tmp_path,cases)


def test_current_reference_is_explicit_and_never_relabelled_legacy_report(tmp_path):
    manifest,cases=fixture(tmp_path)
    with pytest.raises(ValueError,match='Explicit'):validate_current_reference(dict(manifest,schema='legacy'),tmp_path,cases)
    p=tmp_path/'profile.json';profile=json.loads(p.read_text());profile['inference_lora_fold']=dict(enabled=True);p.write_text(json.dumps(profile))
    manifest['reference_profile_sha256']=sha(p)
    with pytest.raises(ValueError,match='unmerged646'):validate_current_reference(manifest,tmp_path,cases)
