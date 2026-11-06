import json
from types import SimpleNamespace

import pytest
from training.venus import ssh_readiness as s


def test_fast_boot_failures_retry_by_elapsed_time_instead_of_twenty_attempts(tmp_path):
    now=[0.0]; calls=[]
    def check(argv, timeout):
        calls.append(timeout)
        if len(calls)<=25:raise RuntimeError("4003: failed to connect to backend")
        return 'DUPLEX_SSH_READY'
    c=SimpleNamespace(R=tmp_path,ssh_argv=['ssh','fixture'])
    s.wait(c,240,clock=lambda:now[0],sleep=lambda n:now.__setitem__(0,now[0]+n),check=check)
    proof=json.loads((tmp_path/'ssh-readiness.json').read_text())
    assert len(calls)==26 and proof['ready'] and len(proof['attempts'])==25
    assert now[0]==75 and calls[-1]<=20


def test_boot_deadline_preserves_error_and_does_not_claim_ready(tmp_path):
    now=[0.0]
    def check(*args):raise RuntimeError('4010: destination read failed')
    with pytest.raises(RuntimeError,match='deadline exhausted.*4010'):
        s.wait(SimpleNamespace(R=tmp_path,ssh_argv=[]),7,clock=lambda:now[0],
               sleep=lambda n:now.__setitem__(0,now[0]+n),check=check)
    assert now[0]==7
    assert not json.loads((tmp_path/'ssh-readiness.json').read_text())['ready']


def test_host_key_failure_stops_immediately(tmp_path):
    def check(*args):raise RuntimeError('Host key verification failed')
    with pytest.raises(RuntimeError,match='Permanent'):
        s.wait(SimpleNamespace(R=tmp_path,ssh_argv=[]),240,clock=lambda:0,
               sleep=lambda n:pytest.fail('Must not retry host key failure'),check=check)
