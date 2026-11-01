import hashlib
import json
from pathlib import Path
import shutil

import pytest
from training.venus import context_controller as control


@pytest.mark.parametrize('corrupt',[False,True])
def test_candidate_preservation_receipt_requires_actual_complete_hashes(monkeypatch,tmp_path,corrupt):
    source=tmp_path/'remote-candidate';source.mkdir()
    (source/'training-terminal.json').write_text('{"completed":true}')
    (source/'step-0060.pt').write_bytes(b'fixture-only-saved-state-not-a-model')
    rows=[dict(path=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(source.iterdir())]
    result=tmp_path/'controller';result.mkdir()
    monkeypatch.setattr(control.c,'R',result)
    monkeypatch.setattr(control.c,'source','fixture-source')
    monkeypatch.setattr(control.c,'ssh_argv',['ssh','fixture'])
    monkeypatch.setattr(control.c,'ssh',lambda *a,**kw:json.dumps(rows))
    monkeypatch.setattr(control.c,'record',lambda *a,**kw:None)
    def copy(*args,**kw):
        for p in source.iterdir():shutil.copy2(p,result/'critical-candidate'/p.name)
        if corrupt:(result/'critical-candidate/step-0060.pt').write_bytes(b'corrupt')
    monkeypatch.setattr(control.c,'run',copy)
    if corrupt:
        with pytest.raises(ValueError):control.preserve_candidate('/fixture/remote-candidate')
        assert not (result/'critical-candidate-verified.json').exists()
    else:
        control.preserve_candidate('/fixture/remote-candidate')
        report=json.loads((result/'critical-candidate-verified.json').read_text())
        assert report['all_hashes_verified'] and len(report['files'])==2
