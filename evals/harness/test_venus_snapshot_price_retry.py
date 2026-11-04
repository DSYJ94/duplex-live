import io,json
from types import SimpleNamespace
from urllib.error import HTTPError
import pytest
from training.venus import tts_placement as p


def test_actual_snapshot_price_read_backs_off_before_any_policy_mutation(tmp_path,monkeypatch):
    calls=[];sleeps=[]
    ids=['817F-F5A3-514E','092B-DDAC-75F4','22CE-2CCB-B087','9B7C-E01E-BA5B']
    payload=dict(skus=[dict(skuId=k,description='fixture snapshot rate',pricingInfo=[dict(pricingExpression=dict(usageUnit='GiBy.mo',tieredRates=[dict(unitPrice=dict(units='0',nanos=50000000))]))]) for k in ids])
    def read(req,timeout):
        calls.append(req.full_url)
        if len(calls)==1:raise HTTPError(req.full_url,429,'fixture throttle',{},None)
        return io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr(p.urllib.request,'urlopen',read);monkeypatch.setattr(p.c,'run',lambda *a:'fixture-only-token')
    monkeypatch.setattr(p.c,'R',tmp_path);monkeypatch.setattr(p,'time',SimpleNamespace(sleep=sleeps.append))
    class ReachedPolicy(Exception):pass
    def policy(*a):raise ReachedPolicy
    monkeypatch.setattr(p,'api',policy)
    with pytest.raises(ReachedPolicy):p.snapshot()
    assert len(calls)==2 and sleeps==[20]
    assert len(json.loads((tmp_path/'snapshot-prices.json').read_text())['rates'])==4


def test_price_retry_is_bounded_and_never_mutates_on_exhaustion(tmp_path,monkeypatch):
    calls=[];sleeps=[]
    def read(req,timeout):
        calls.append(req.full_url);raise HTTPError(req.full_url,429,'fixture throttle',{},None)
    monkeypatch.setattr(p.urllib.request,'urlopen',read);monkeypatch.setattr(p.c,'run',lambda *a:'fixture-only-token')
    monkeypatch.setattr(p.c,'R',tmp_path);monkeypatch.setattr(p,'time',SimpleNamespace(sleep=sleeps.append))
    monkeypatch.setattr(p,'api',lambda *a:pytest.fail('No snapshot mutation after exhausted price reads'))
    with pytest.raises(HTTPError):p.snapshot()
    assert len(calls)==3 and sleeps==[20,40]
    assert not list(tmp_path.iterdir())


def test_resume_exact_read_failure_preserves_claim_bundle_and_deadline(tmp_path,monkeypatch):
    from training.venus import tts_stop_controller as ctl
    import time
    root=tmp_path/'tts-completion-v63';root.mkdir()
    for name,value in [('claim.json',dict(pid=123,prior_focused_gpu_seconds_upper=33700)),('events.json',[dict(event='tts_stop_failed',error_type='HTTPError',error='HTTP Error 429: Too Many Requests')]),('storage-deadline.json',dict(storage_deadline_unix=time.time()+7200))]:
        (root/name).write_text(json.dumps(value))
    (root/'source.bundle').write_bytes(b'original fixture source');(root/'bootstrap.sh').write_text('not started')
    deadline=(root/'storage-deadline.json').read_bytes();claim=(root/'claim.json').read_bytes()
    monkeypatch.setattr(ctl.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stdout=''));monkeypatch.setattr(ctl.c,'events',[])
    (root/'disk.json').write_text('{}')
    with pytest.raises(AssertionError):ctl.resume_price_preflight(root)
    (root/'disk.json').unlink();ctl.resume_price_preflight(root)
    assert (root/'storage-deadline.json').read_bytes()==deadline and (root/'claim.json').read_bytes()==claim
    assert (root/'price-throttled-source.bundle').read_bytes()==b'original fixture source'
    with pytest.raises(AssertionError):ctl.resume_price_preflight(root)
