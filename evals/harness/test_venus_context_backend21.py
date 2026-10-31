import copy
import json
from pathlib import Path
import pytest
from portable.venus.context_backend_plan import payload,plan,location_payload,location_plan
from evals.venus_context_backend21 import preflight,preflight_v2


def test_exact21_admission_includes_failed_native_tasks_and_all_original_audio(monkeypatch):
    from datetime import datetime
    from evals import venus_context_backend21 as runner
    if not runner.PRICES.exists():pytest.skip('Optional preserved local evaluation artifacts')
    source=json.loads(Path('docs/NATIVE-VENUS-CONTEXT-BACKEND21-PROPOSAL-2026-10-31.json').read_text())
    if not all(Path(x['audio_file']).exists() for x in source['requests']):pytest.skip('Optional preserved local audio')
    prices=json.loads(runner.PRICES.read_text())
    monkeypatch.setattr(runner.time,'time',lambda:datetime.fromisoformat(prices['recorded_utc']).timestamp()+1)
    auth,p,price=preflight()
    assert auth['maximum_api_usd']==5 and price['maximum_total_usd']==4.4268
    assert sum(x['seconds'] for x in p['requests'])==154
    assert any(not x['native_selected_task'] for x in p['requests'])
    for row in p['requests']:
        body=payload(row['native_selected_task'],'x'*2048,row['causal_audio'])
        assert body['model']=='gpt-6-astra' and body['service_tier']=='default'
        assert body['max_output_tokens']==1024 and body['store'] is False
        assert json.loads(body['input'])['native_selected_task']==row['native_selected_task']


def test_does_not_truncate_observation_or_accept_future_audio():
    row=json.loads(Path('docs/NATIVE-VENUS-CONTEXT-BACKEND21-PROPOSAL-2026-10-31.json').read_text())['requests'][0]
    with pytest.raises(ValueError):payload('', 'x'*2049,row['causal_audio'])
    e=copy.deepcopy(row['causal_audio']);e['after_opening_audio_included']=True
    with pytest.raises(ValueError):payload('', '',e)


def test_rejects_incomplete_or_ungrounded_lookup_plan():
    def response(d):return dict(status='completed',output=[dict(type='function_call',name='plan_context_task',arguments=json.dumps(d))])
    good=dict(intent='clarify',lookup_key=None,grounding='insufficient',response='Which record should I check?')
    assert plan(response(good))==good
    with pytest.raises(ValueError):plan(response(dict(good,intent='lookup',lookup_key='guessed')))
    with pytest.raises(ValueError):plan(dict(status='incomplete',output=[]))


def test_v2_contract_defines_record_location_and_accepts_declared_cancel_target():
    p=json.loads(Path('docs/NATIVE-VENUS-CONTEXT-BACKEND21-PROPOSAL-2026-10-31.json').read_text())
    for r in p['requests']:
        body=location_payload(r['native_selected_task'],'x'*2048,r['causal_audio'])
        fields=body['tools'][0]['parameters']['properties']
        assert 'lookup_key' not in fields and 'record_location' in fields
        assert 'not the object' in body['instructions']
    d=dict(intent='cancel',record_location='named location',grounding='audio',response='Cancellation requested, not executed.')
    response=dict(status='completed',output=[dict(type='function_call',name='plan_context_task_v2',arguments=json.dumps(d))])
    assert location_plan(response)==d
    response['output'][0]['arguments']=json.dumps(dict(d,intent='lookup',grounding='insufficient'))
    with pytest.raises(ValueError):location_plan(response)


def test_v2_requires_its_own_approval_before_credentials_or_provider_access(monkeypatch):
    from evals import venus_context_backend21 as runner
    def missing(path):raise FileNotFoundError('Fixture: version2 approval absent')
    monkeypatch.setattr(runner,'read',missing)
    with pytest.raises(FileNotFoundError):preflight_v2()
