"""Replacement recovery cannot borrow consumed approval, clocks or guard paths."""
import json

import pytest

from training.venus import british_recovery as b, results_recovery as r
from evals.harness.test_venus_results_recovery import cpu


@pytest.fixture
def configured(monkeypatch):
    for name in ('R','CPU','PROPOSAL','AUTH','RUN_SECONDS','PRIOR_CPU_USD','ENTRYPOINT','require_authorization'):
        monkeypatch.setattr(r,name,getattr(r,name))
    b.configure()


def test_new_guard_and_main_share_shorter_duration_and_namespace(configured):
    assert r.RUN_SECONDS==600 and r.ENTRYPOINT=='training.venus.british_recovery'
    assert r.CPU=='duplex-venus-results-cpu-20261106-v2'
    assert str(r.R).endswith('results-recovery-v83')
    assert r.PRIOR_CPU_USD==b.prior_usage()*.12/3600
    assert b.prior_usage()+r.RUN_SECONDS+100<900


def test_no_borrowing_previous_approved_recovery(configured,monkeypatch,tmp_path):
    monkeypatch.setattr(b,'AUTH',tmp_path/'new-missing-authorization.json')
    monkeypatch.setattr(r,'cloud',lambda *a:pytest.fail('New resource before new approval'))
    with pytest.raises(FileNotFoundError):r.execute()


def test_actual_new_cpu_contract_rejects_601_seconds(configured):
    vm=cpu();vm['name']=r.CPU;vm['scheduling']['maxRunDuration']['seconds']='600'
    r.validate_cpu(vm)
    vm['scheduling']['maxRunDuration']['seconds']='601'
    with pytest.raises(ValueError,match='STOP'):r.validate_cpu(vm)


def test_prior_usage_must_be_actual_original_terminal(monkeypatch,tmp_path):
    original=json.loads((b.PRIOR/'provider-final.json').read_text())
    original['status']='RUNNING'
    (tmp_path/'provider-final.json').write_text(json.dumps(original))
    monkeypatch.setattr(b,'PRIOR',tmp_path)
    with pytest.raises(ValueError,match='terminal'):b.prior_usage()


def test_changed_proposal_cannot_inherit_exact_new_approval(monkeypatch,tmp_path):
    path=tmp_path/'approval.json'
    path.write_text(json.dumps(dict(status='APPROVED',human_reply_verbatim='test fixture only',proposal_path=str(b.PROPOSAL),proposal_sha256='wrong')))
    monkeypatch.setattr(b,'AUTH',path)
    with pytest.raises(ValueError,match='Exact consolidated'):b.require_authorization()
