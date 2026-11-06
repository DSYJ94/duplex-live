"""Files-first recovery preserves original native admission and one GPU clock."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from training.venus import british_files_first as f,british_speech_guard as g,british_controller as b


def test_files_first_requires_explicit_execution_record(monkeypatch):
    monkeypatch.setattr(g,'authority',lambda:{})
    with pytest.raises(ValueError,match='files-first'):f.precheck()


def test_full_backup_is_written_only_after_all_original_hashes(tmp_path,monkeypatch):
    dest=tmp_path/'dest';dest.mkdir();run=tmp_path/'run';run.mkdir()
    (dest/'old.json').write_bytes(b'old');(dest/'new.json').write_bytes(b'new')
    rows=[dict(path=name,bytes=3,sha256=f.r.sha(dest/name)) for name in ['old.json','new.json']]
    monkeypatch.setattr(f.r,'DEST',dest);monkeypatch.setattr(b,'R',run);monkeypatch.setattr(g,'BACKUP',run/'proof.json')
    monkeypatch.setattr(f.r,'manifest',lambda:(dict(files=rows,links=[]),[rows[1]]))
    monkeypatch.setattr(f.time,'time',lambda:0)
    c=SimpleNamespace(ssh_argv=['ssh','TEST_HOST'],run=lambda *a:None)
    watch=SimpleNamespace(poll=lambda:None)
    f.recover(c,1000,watch)
    proof=json.loads(g.BACKUP.read_text());assert proof['all_hashes_verified'] and 'before any native model' in proof['recovered_via']
    g.BACKUP.unlink();(dest/'old.json').write_bytes(b'bad')
    with pytest.raises(ValueError,match='Full original'):f.recover(c,1000,watch)
    assert not g.BACKUP.exists()


def test_dead_independent_guard_prevents_transfer(tmp_path,monkeypatch):
    monkeypatch.setattr(f.r,'manifest',lambda:(dict(files=[],links=[]),[dict(path='x',bytes=1,sha256='x')]))
    c=SimpleNamespace(run=lambda *a:pytest.fail('Transfer after dead guard'))
    with pytest.raises(ValueError,match='guard expired'):f.recover(c,1000,SimpleNamespace(poll=lambda:1))
