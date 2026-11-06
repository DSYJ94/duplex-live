import base64
import hashlib
import json

import pytest
from training.venus import boot_hostkeys as h


@pytest.fixture
def provenance(tmp_path,monkeypatch):
    value=base64.b64encode(b'CPU-authenticated public-key fixture').decode()
    (tmp_path/'known-hosts').write_text('cpu ssh-ed25519 '+value+'\n')
    (tmp_path/'provider-start.json').write_text(json.dumps(dict(id='2296018948858061861',disks=[dict(boot=True,autoDelete=False,source='/disks/duplex-venus-large-data-20261104-eua-v67')])))
    (tmp_path/'original-binding-restored.json').write_text(json.dumps(dict(original_vm_id='7521667153519853425',disk_id='2297341858915206463',status='TERMINATED',auto_delete=False)))
    monkeypatch.setattr(h,'ROOT',tmp_path)
    proofs={name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in h.PROOFS}
    monkeypatch.setattr(h,'PROOFS',proofs)
    fp='SHA256:'+base64.b64encode(hashlib.sha256(base64.b64decode(value)).digest()).decode().rstrip('=')
    monkeypatch.setattr(h,'FINGERPRINT',fp)
    return tmp_path,value


def test_previously_authenticated_same_disk_key_is_accepted(provenance):
    _,value=provenance
    assert h.verified()==[dict(key='ssh-ed25519',value=value)]


def test_network_or_old_gpu_key_cannot_replace_the_authenticated_disk_key(provenance):
    root,_=provenance
    (root/'known-hosts').write_text('cpu ssh-ed25519 YXR0YWNrZXI=\n')
    with pytest.raises(ValueError,match='provenance changed'):h.verified()


def test_different_disk_fails_even_with_matching_file_hash(provenance):
    root,_=provenance
    p=root/'provider-start.json';a=json.loads(p.read_text());a['disks'][0]['source']='/disks/different'
    p.write_text(json.dumps(a));h.PROOFS[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='same retained boot disk'):h.verified()


def test_pinned_boot_key_never_uses_stale_instance_metadata_or_disables_strict_checks(tmp_path,monkeypatch):
    from training.venus import source_controller as s,controller as c,ssh_readiness
    monkeypatch.setattr(c,'R',tmp_path)
    monkeypatch.setattr(c,'cloud',lambda *a:pytest.fail('Stale per-instance keys must not replace boot-disk trust'))
    monkeypatch.setattr(c,'ssh',lambda *a:'')
    monkeypatch.setattr(ssh_readiness,'wait',lambda *a:None)
    monkeypatch.setattr(c,'ssh_argv',None)
    vm=dict(networkInterfaces=[dict(accessConfigs=[dict(natIP='192.0.2.1')])])
    s.prepare_ssh(vm,'europe-west4-a',readiness_timeout=240,trusted_boot_keys=[dict(key='ssh-ed25519',value='YXV0aGVudGljYXRlZA==')])
    assert 'StrictHostKeyChecking=yes' in c.ssh_argv
    assert 'HostKeyAlgorithms=ssh-ed25519' in c.ssh_argv
    assert 'ssh-ed25519 YXV0aGVudGljYXRlZA==' in (tmp_path/'.private/known-hosts').read_text()
