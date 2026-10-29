import hashlib,json,wave
from pathlib import Path
import numpy as np,soundfile as sf
import pytest
from training.venus.collection_resume import completed_cases,preserve_partial


def fixture(tmp_path):
    root=tmp_path/'collection';folder=root/'parent244'/'case';folder.mkdir(parents=True)
    source=tmp_path/'source';source.mkdir();pcm=bytes(32000)
    with wave.open(str(source/'user.wav'),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(pcm)
    case=dict(id='case',frames=1,input_sha256='sourcehash',folder=str(source))
    stored=dict(id='case',complete=True,input_sha256='sourcehash',input_frames=1)
    row=dict(**stored,condition='parent244')
    (folder/'result.json').write_text(json.dumps(stored))
    trace=dict(sequence=0,input_sha256=hashlib.sha256(pcm).hexdigest(),chunk_file='chunk-000.wav',audio_samples=24000,peak=0.)
    (folder/'trace.jsonl').write_text(json.dumps(trace)+'\n');sf.write(folder/'chunk-000.wav',np.zeros(24000),24000,subtype='FLOAT')
    (root/'results.json').write_text(json.dumps(dict(required_cases=64,cases=[row])))
    return root,case,folder


def test_complete_case_reused_without_replay_and_partial_preserved(tmp_path):
    root,case,folder=fixture(tmp_path)
    assert list(completed_cases(root,[case],['parent244']))==[('parent244','case')]
    partial=root/'parent244'/'unfinished';partial.mkdir();(partial/'trace.jsonl').write_text('partial')
    preserve_partial(root,'parent244','unfinished','v37')
    assert (root/'interrupted-before-v37/parent244/unfinished/trace.jsonl').read_text()=='partial'
    assert folder.exists()


@pytest.mark.parametrize('damage',['truncated','input','wave','duplicate'])
def test_corrupt_completed_case_cannot_be_skipped(tmp_path,damage):
    root,case,folder=fixture(tmp_path)
    if damage=='truncated':(folder/'trace.jsonl').write_text('')
    elif damage=='input':
        p=folder/'trace.jsonl';j=json.loads(p.read_text());j['input_sha256']='wrong';p.write_text(json.dumps(j)+'\n')
    elif damage=='wave':sf.write(folder/'chunk-000.wav',np.ones(24000),24000,subtype='FLOAT')
    else:
        p=root/'results.json';j=json.loads(p.read_text());j['cases']*=2;p.write_text(json.dumps(j))
    with pytest.raises(ValueError):completed_cases(root,[case],['parent244'])


def test_clone_keeps_original_json_immutable_and_reuses_only_raw_chunk_bytes(tmp_path):
    from training.venus.collection_resume import clone_original_collection
    root,case,folder=fixture(tmp_path);new=tmp_path/'new'
    clone_original_collection(root,new)
    (new/'results.json').write_text('derived')
    assert json.loads((root/'results.json').read_text())['required_cases']==64
    assert (new/'parent244/case/chunk-000.wav').read_bytes()==(folder/'chunk-000.wav').read_bytes()


def test_completed_orphan_is_reused_after_aggregate_write_interruption(tmp_path):
    root,case,folder=fixture(tmp_path);case['evaluation_set']='fresh24'
    (root/'results.json').write_text(json.dumps(dict(required_cases=64,cases=[])))
    assert list(completed_cases(root,[case],['parent244']))==[('parent244','case')]


def test_provider_read_timing_tolerates_slow_fresh_read_without_extending_lease(monkeypatch):
    import subprocess
    from portable.venus import guard
    limits=[]
    def slow_read(*args,timeout):
        limits.append(timeout)
        if timeout<17:raise subprocess.TimeoutExpired('gcloud',timeout)
        return {'fresh':True}
    monkeypatch.setattr(guard,'cloud',slow_read)
    with pytest.raises(subprocess.TimeoutExpired):guard.bounded_cloud_read('instances','list')
    assert guard.bounded_cloud_read('instances','list',limits=(25,20))=={'fresh':True}
    with pytest.raises(ValueError):guard.bounded_cloud_read('instances','list',limits=(60,60))


def test_full_parent_reuse_preserves_failed_cases_and_requires_exact_all_waveforms(tmp_path):
    import shutil
    from training.venus.collection_resume import reuse_parent
    root,template,folder=fixture(tmp_path)
    original_files={p.name:p.read_bytes() for p in folder.iterdir()}
    shutil.rmtree(root/'parent244')
    rows=[];audits=[];cases=[]
    for number in range(32):
        ident='case-'+str(number)
        case=dict(template,id=ident,evaluation_set='fresh24' if number<24 else 'old8ordinary-retention')
        cases.append(case)
        for condition in ('parent244','paced-continuation'):
            dest=root/condition/ident;dest.mkdir(parents=True)
            for name,data in original_files.items():(dest/name).write_bytes(data)
            stored=dict(id=ident,complete=True,input_sha256=case['input_sha256'],input_frames=1,
                        teacher_actions_used=False,oracle_feedback_used=False)
            (dest/'result.json').write_text(json.dumps(stored))
            shutil.copyfile(dest/'chunk-000.wav',dest/'joined-emitted.wav')
            rows.append(dict(stored,condition=condition,evaluation_set=case['evaluation_set']))
            h=hashlib.sha256((dest/'chunk-000.wav').read_bytes()).hexdigest()
            audits.append(dict(id=ident,condition=condition,word_pass=False,
                joined_wav_sha256=h,source_chunks=[dict(file='chunk-000.wav',sha256=h)]))
    (root/'results.json').write_text(json.dumps(dict(cases=rows,required_cases=64)))
    (root/'paired-native-report.json').write_text(json.dumps(dict(complete=True,cases=audits)))
    (root/'paced-continuation/profile.json').write_text(json.dumps(dict(checkpoint_sha256='b'*64,optimizer_updates=298)))
    out=tmp_path/'derived';out.mkdir()
    reuse_parent(root,out,cases,'b'*64,'parent298')
    assert len(completed_cases(out,cases,('parent298','paced-continuation')))==32
    reused=json.loads((out/'speech-audit-progress.json').read_text())
    assert reused['completed']==32 and all(not r['word_pass'] for r in reused['cases'])
    assert len(json.loads((root/'results.json').read_text())['cases'])==64
    # Appending bytes preserves decoded PCM, but must invalidate exact ASR input.
    p=root/'paced-continuation/case-0/joined-emitted.wav'
    p.write_bytes(p.read_bytes()+b'changed-container')
    another=tmp_path/'another';another.mkdir()
    with pytest.raises(ValueError,match='ASR waveform'):
        reuse_parent(root,another,cases,'b'*64,'parent298')
