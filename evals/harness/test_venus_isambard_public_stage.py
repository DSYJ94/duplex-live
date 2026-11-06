import json

import pytest
from training.venus import isambard_public_stage as s


def test_refuses_login_node_before_download(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    monkeypatch.setattr(s.subprocess,'check_output',lambda *a,**kw:pytest.fail('No commands on login node'))
    with pytest.raises(ValueError,match='never on a login node'):s.main('source')


@pytest.mark.parametrize('code',[0,1])
def test_terminal_records_actual_worker_status_without_model_execution(tmp_path,monkeypatch,code):
    monkeypatch.setenv('SLURM_JOB_ID','TEST123');monkeypatch.setenv('SLURM_JOB_ACCOUNT','brics.u6ada')
    monkeypatch.setattr(s,'BASE',tmp_path);(tmp_path/'staging-20261106-v1').mkdir()
    manifest=tmp_path/'manifest.json';manifest.write_text('[]');monkeypatch.setattr(s,'MANIFEST',manifest)
    monkeypatch.setattr(s,'MANIFEST_SHA',s.hashlib.sha256(manifest.read_bytes()).hexdigest())
    monkeypatch.setattr(s.subprocess,'check_output',lambda *a,**kw:'EXACT_SOURCE\n')
    monkeypatch.setattr(s.platform,'machine',lambda:'aarch64')
    class Worker:
        pid=12345
        def __init__(self,argv,**kw):
            assert 'portable.venus.fetch_public' in argv
            assert not any('train' in a or 'native_app' in a for a in argv)
        def wait(self,timeout=None):return code
        def poll(self):return code
    monkeypatch.setattr(s.subprocess,'Popen',Worker)
    with pytest.raises(SystemExit) as error:s.main('EXACT_SOURCE')
    assert error.value.code==code
    proof=json.loads((tmp_path/'staging-20261106-v1/slurm-TEST123/actual-terminal.json').read_text())
    assert proof['actual_exit_code']==code and proof['all35_hashes_verified'] is (code==0)
    assert not proof['model_execution'] and proof['new_optimizer_updates']==0
