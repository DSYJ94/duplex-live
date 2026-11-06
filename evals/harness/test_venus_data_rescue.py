import hashlib
import json

import pytest
from training.venus import data_rescue as d


def test_absent_exact_approval_prevents_all_provider_calls(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'AUTH',tmp_path/'no-owner-approval')
    monkeypatch.setattr(d.r,'cloud',lambda *a:pytest.fail('Must not reach provider'))
    with pytest.raises(FileNotFoundError):d.execute()


@pytest.mark.parametrize('path',['/absolute','../escape','native/.private/key',
    'native/owner-conversations/audio.wav','native/session-token','native/API keys.md',
    'native/core.123','native/credentials/key','native/key.pem'])
def test_private_owner_and_escape_paths_excluded(path):
    assert not d.allowed(path)


@pytest.fixture
def inventory_inputs(tmp_path,monkeypatch):
    base=tmp_path/'workspace';base.mkdir()
    monkeypatch.setattr(d,'BASE',base);monkeypatch.setattr(d,'authority',lambda:{})
    q=dict(cache_manifest_sha256=d.MANIFEST_SHA,required_cache=[])
    for i in range(268):
        p=base/'venus-training-v1/paced-prefix-cache-v24'/('scene-%03d.pt'%i)
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'exact-native-cache')
        q['required_cache'].append(dict(path=str(p.relative_to(base)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    for name in d.EXTRAS:
        top=base/name;top.mkdir(parents=True,exist_ok=True);(top/'public-native.json').write_text('{}')
    for name in d.SINGLE:
        p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    monkeypatch.setattr(d,'request',lambda *a:q)
    request=tmp_path/'request.json';request.write_text(json.dumps(q))
    return base,request,tmp_path/'inventory.json'


def test_all268_hashes_plus_named_native_files_only(inventory_inputs):
    base,request,output=inventory_inputs
    private=base/d.EXTRAS[0]/'owner-conversations';private.mkdir();(private/'private.wav').write_bytes(b'DO_NOT_COPY')
    (base/d.EXTRAS[0]/'outside-link').symlink_to('/etc/passwd')
    d.inventory(request,output);a=json.loads(output.read_text())
    cached=[x for x in a['files'] if 'paced-prefix-cache-v24/' in x['path']]
    assert len(cached)==268 and len(a['files'])==275
    assert not any('owner-conversations' in x['path'] for x in a['files'])
    assert not a['links']


def test_one_wrong_cache_hash_rejects_the_whole_selection(inventory_inputs):
    base,request,output=inventory_inputs
    (base/'venus-training-v1/paced-prefix-cache-v24/scene-267.pt').write_bytes(b'wrong')
    with pytest.raises(ValueError,match='cached file hash differs'):d.inventory(request,output)
    assert not output.exists()


def test_oversize_selection_cannot_be_accepted(inventory_inputs,monkeypatch):
    _,request,output=inventory_inputs;monkeypatch.setattr(d,'MAX_BYTES',1)
    with pytest.raises(ValueError,match='transfer cap'):d.inventory(request,output)
    assert not output.exists()


def test_exact_pinned_cache_reference_can_resolve_to_its_same_native_bytes(inventory_inputs):
    base,request,output=inventory_inputs
    original=base/'venus-training-v1/paced-prefix-cache-v24/scene-000.pt'
    target=base/'venus-training-v1/earlier-cache/scene-000.pt';target.parent.mkdir();original.rename(target)
    original.symlink_to(target)
    d.inventory(request,output);a=json.loads(output.read_text());d.admit_inventory(a,json.loads(request.read_text()))
    assert a['cache_sources'][0]['stored_path']=='venus-training-v1/earlier-cache/scene-000.pt'
    assert len(a['cache_sources'])==268
    a['cache_sources'].pop()
    with pytest.raises(ValueError,match='All268'):d.admit_inventory(a,json.loads(request.read_text()))
