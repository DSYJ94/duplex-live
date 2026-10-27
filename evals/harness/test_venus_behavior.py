"""Protocol tests only; no native behavioral qualification from these fixtures."""
import base64,hashlib,json,time,unittest,wave
from pathlib import Path
from types import SimpleNamespace
import pytest
from training.venus.behavior import preflight,run_case


class FakeEngine:
    def __init__(self,dispatch):
        self.dispatch=dispatch;self.inputs=[];self.duplex=SimpleNamespace(current_turn_ended=True)
    def prepare(self):pass
    def step(self,pcm,feedback):
        index=len(self.inputs);self.inputs.append((pcm,feedback))
        return dict(raw_text='<delegate>look up amber</delegate>' if self.dispatch and index==0 else '',
            text='',audio=base64.b64encode(b'').decode(),peak=0,audio_samples=0,is_listen=True)


def test_focused_budget_retains_previous_intervals_and_forbids_overlap():
    from training.venus.focused_guard import AUTH,IDENT,NAME,validate
    a=json.loads(AUTH.read_text());r=a['resources'];start=1800000000
    from datetime import datetime,timezone
    lease=dict(resource_key='venus-focused-audit-v15',vm_id=IDENT,disk_id=r['same_existing_disk_id'],zone=r['zone'],disk_size_gib=100,creation_timestamp='immutable',stop_unix=start+1300,prior_new_gpu_seconds_upper=0,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,authorization_sha256=hashlib.sha256(AUTH.read_bytes()).hexdigest(),verified_hourly_price_usd=3.68708363)
    vm=dict(id=IDENT,name=NAME,zone='/'+r['zone'],status='RUNNING',machineType='/a2-highgpu-1g',creationTimestamp='immutable',lastStartTimestamp=datetime.fromtimestamp(start,timezone.utc).isoformat(),tags=dict(items=['duplex-venus-train-bounded']),disks=[dict(boot=True,autoDelete=False,source='/disks/'+r['same_existing_disk_name'])],scheduling=dict(provisioningModel='FLEX_START',maxRunDuration=dict(seconds=1300),instanceTerminationAction='STOP',automaticRestart=False))
    assert validate(vm,[vm],[],lease,start+10)==start+1300
    for bad in ({'prior_gpu_seconds':40351},{'prior_new_gpu_seconds_upper':100},{'stop_unix':start+3000},{'verified_hourly_price_usd':4.01}):
        with pytest.raises(AssertionError):validate(vm,[vm],[],dict(lease,**bad),start+10)
    with pytest.raises(AssertionError):validate(vm,[vm,dict(vm,id='999')],[],lease,start+10)
    with pytest.raises(AssertionError):validate(vm,[vm],[dict(status='RUNNING')],lease,start+10)


class NativeWorkerProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_diagnosed_lookup_variants_and_missing_key_do_not_become_sum_jobs(self):
        import asyncio
        from data.venus.receipts import LookupWorker,lookup_key
        for query in ('look up cabinet','lookup marker cabinet','lookup the marker in cabinet.'):
            assert lookup_key(query)=='cabinet'
            worker=LookupWorker({'cabinet':'violet'},0);worker.submit(query)
            await asyncio.gather(*worker.tasks.values());assert worker.completed[0]['result']=='Lookup cabinet: violet'
            await worker.close()
        worker=LookupWorker({'cabinet':'violet'},0);worker.submit('lookup')
        await asyncio.gather(*worker.tasks.values())
        assert worker.completed[0]['error']=='lookup_requires_explicit_key'
        assert 'violet' not in worker.take_results();await worker.close()
        with pytest.raises(ValueError):lookup_key('lookup cabinet and jacket')

    async def test_only_model_dispatch_can_produce_backend_feedback(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input';source.mkdir()
            with wave.open(str(source/'user.wav'),'wb') as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(bytes(32000*4))
            for dispatch in (False,True):
                out=root/str(dispatch);out.mkdir();engine=FakeEngine(dispatch)
                case=dict(id='fixture',family='delegate',folder=str(source),frames=4,world={'amber':'blue'},worker_delay=0,
                    user_turns=[dict(start_seconds=0,seconds=.5)],input_sha256='fixture-only')
                result=await run_case(engine,case,out,time.time()+3600,pace=False)
                received=[feedback for _,feedback in engine.inputs if feedback]
                assert bool(received)==dispatch
                if dispatch:
                    assert engine.inputs[0][1] is None and 'blue' in received[0]
                    assert any(e['type']=='worker_completed' for e in result['actual_worker_events'])
                else:assert not result['actual_worker_events']
                assert result['complete'] and not result['qualified'] and not result['oracle_feedback_used']
                assert not result['teacher_actions_used']


@pytest.mark.skipif(not Path('runs/venus-training-20261026/data-approval-v7.json').exists(), reason='Integration requires the verified ignored synthetic artifacts')
def test_complete36_preflight_uses_only_user_inputs_and_live_worker_configuration():
    cases=preflight('evals/fixtures/venus-control-corpus-v4-20261026.json',
        'runs/venus-training-20261026/timelines-v7-repaired','runs/venus-training-20261026/data-approval-v7.json')
    assert len(cases)==36 and sum(c['frames'] for c in cases)==753
    assert all(not {'actions','backend_feedback','teacher_program','expected','result_answer'}&c.keys() for c in cases)


def test_native_behavior_guard_carries_training_time_and_rejects_larger_window():
    from training.venus.behavior_guard import AUTH,validate,stamp
    raw=AUTH.read_bytes();a=json.loads(raw);r=a['resources'];start=stamp('2026-10-27T17:00:00Z')
    lease=dict(vm_id='123',disk_id=r['same_disk_id'],zone=r['zone'],disk_size_gib=100,creation_timestamp='2026-10-27T16:59:00Z',stop_unix=start+1300,prior_new_gpu_seconds_upper=13000,prior_gpu_seconds=40351,prior_scope_usd=91.98448055555555,authorization_sha256=hashlib.sha256(raw).hexdigest(),verified_hourly_price_usd=3.68708363)
    vm=dict(id='123',name=r['new_name'],zone='/'+r['zone'],status='RUNNING',machineType='/'+r['machine'],creationTimestamp=lease['creation_timestamp'],lastStartTimestamp='2026-10-27T17:00:00Z',tags=dict(items=['duplex-venus-train-bounded']),disks=[dict(boot=True,autoDelete=False,source='/disks/'+r['same_disk_name'])],scheduling=dict(provisioningModel='FLEX_START',maxRunDuration=dict(seconds=1300),instanceTerminationAction='STOP',automaticRestart=False))
    assert validate(vm,[vm],[],lease,start+10)==start+1300
    for bad in ({'prior_new_gpu_seconds_upper':11300},{'stop_unix':start+3000},{'verified_hourly_price_usd':4.01}):
        with pytest.raises(AssertionError):validate(vm,[vm],[],dict(lease,**bad),start+10)
    with pytest.raises(AssertionError):validate(vm,[vm,dict(vm,id='999')],[],lease,start+10)


def test_spoken_audit_keeps_tool_words_separate_from_intended_speech():
    from training.venus.spoken_audit import references
    traces=[dict(text='',raw_text='<unit><|speak|><delegate>look'),dict(text='',raw_text=' up amber</delegate><|turn_eos|></unit>'),dict(text='',raw_text='<unit><|speak|>It is blue.<|turn_eos|></unit>')]
    refs=references(traces)
    assert refs['literal_generated']=='look up amber It is blue.'
    assert refs['intended_user_facing']=='It is blue.' and refs['complete_tool_calls']==1
    partial=references([dict(text='',raw_text='<unit><|speak|><delegate>look up amber')])
    assert partial['unclosed_tool_call'] and not partial['intended_user_facing']


def test_cue_boundary_is_the_cue_word_not_the_entire_question():
    from training.venus.behavior_review import cue_frame
    turn=dict(text='Pebble. What does a telescope do?',start_seconds=10,seconds=5,sha256='same-audio')
    row=dict(source_sha256='same-audio',words=[dict(text=' Pebble.',timestamp=[0,.4])])
    assert cue_frame(turn,row)==(10,10.4)
    with pytest.raises(ValueError):cue_frame(dict(turn,sha256='other-audio'),row)


def test_paced_continuation_budget_cannot_restart_or_add_gpu(tmp_path,monkeypatch):
    from training.venus import paced_guard as g
    from datetime import datetime,timezone
    auth=json.loads(g.AUTH.read_text());r=auth['resources'];start=1800000000
    plan=dict(status='WITHIN_APPROVED_FOCUSED_REPAIR',parent_sha256=hashlib.sha256(g.AUTH.read_bytes()).hexdigest(),vm_name='fixture-paced-only',maximum_gpu_seconds=10000,focused_gpu_seconds_upper_before_launch=2400)
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));monkeypatch.setattr(g,'PLAN',path)
    lease=dict(resource_key='venus-paced-continuation-v22',vm_id='123',disk_id=r['same_existing_disk_id'],zone=r['zone'],disk_size_gib=100,creation_timestamp='immutable',stop_unix=start+10000,prior_new_gpu_seconds_upper=2400,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,authorization_sha256=hashlib.sha256(g.AUTH.read_bytes()).hexdigest(),execution_plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),verified_hourly_price_usd=3.68708363)
    vm=dict(id='123',name='fixture-paced-only',zone='/'+r['zone'],status='RUNNING',machineType='/a2-highgpu-1g',creationTimestamp='immutable',lastStartTimestamp=datetime.fromtimestamp(start,timezone.utc).isoformat(),tags=dict(items=['duplex-venus-train-bounded']),disks=[dict(boot=True,autoDelete=False,source='/disks/'+r['same_existing_disk_name'])],scheduling=dict(provisioningModel='FLEX_START',maxRunDuration=dict(seconds=10000),instanceTerminationAction='STOP',automaticRestart=False))
    assert g.validate(vm,[vm],[],lease,start+1)==start+10000
    for wrong in ({'prior_new_gpu_seconds_upper':0},{'prior_gpu_seconds':40351},{'stop_unix':start+11000},{'disk_id':'999'},{'execution_plan_sha256':'wrong'}):
        with pytest.raises(AssertionError):g.validate(vm,[vm],[],dict(lease,**wrong),start+1)
    with pytest.raises(AssertionError):g.validate(vm,[vm,dict(vm,id='456')],[],lease,start+1)
    with pytest.raises(AssertionError):g.validate(vm,[vm],[dict(status='RUNNING')],lease,start+1)


def test_warm_recovery_guard_requires_exact_vm_and_carried_failed_start():
    from training.venus import paced_guard as g
    from datetime import datetime,timezone
    a=json.loads(g.AUTH.read_text());r=a['resources'];p=json.loads(g.PLAN_V23.read_text());start=1800000000;ident='2800910655359330463'
    lease=dict(resource_key='venus-paced-continuation-v23',vm_id=ident,disk_id=r['same_existing_disk_id'],zone=r['zone'],disk_size_gib=100,creation_timestamp='immutable',stop_unix=start+10000,prior_new_gpu_seconds_upper=2600,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,authorization_sha256=hashlib.sha256(g.AUTH.read_bytes()).hexdigest(),execution_plan_sha256=hashlib.sha256(g.PLAN_V23.read_bytes()).hexdigest(),verified_hourly_price_usd=3.68708363)
    vm=dict(id=ident,name=p['vm_name'],zone='/'+r['zone'],status='RUNNING',machineType='/a2-highgpu-1g',creationTimestamp='immutable',lastStartTimestamp=datetime.fromtimestamp(start,timezone.utc).isoformat(),tags=dict(items=['duplex-venus-train-bounded']),disks=[dict(boot=True,autoDelete=False,source='/disks/'+r['same_existing_disk_name'])],scheduling=dict(provisioningModel='FLEX_START',maxRunDuration=dict(seconds=10000),instanceTerminationAction='STOP',automaticRestart=False))
    assert g.validate(vm,[vm],[],lease,start+1)==start+10000
    for wrong in ({'vm_id':'123'},{'prior_new_gpu_seconds_upper':2400},{'stop_unix':start+12000}):
        with pytest.raises(AssertionError):g.validate(vm,[vm],[],dict(lease,**wrong),start+1)


def test_transient_runpod_status_retry_never_reuses_permission():
    import urllib.error
    from portable.venus.guard import bounded_runpod_read
    calls=[];pauses=[]
    def reader(key):
        calls.append(key)
        if len(calls)<3:raise urllib.error.HTTPError('https://api.runpod.io/v2/pods',503,'unavailable',{},None)
        return [{'status':'EXITED'}]
    assert bounded_runpod_read('fixture',reader=reader,sleep=pauses.append)==[{'status':'EXITED'}]
    assert len(calls)==3 and pauses==[1,2]
    for code,expected in ((503,3),(401,1),(403,1)):
        count=[]
        def failing(key):
            count.append(1);raise urllib.error.HTTPError('https://api.runpod.io/v2/pods',code,'failed',{},None)
        with pytest.raises(urllib.error.HTTPError):bounded_runpod_read('fixture',reader=failing,sleep=lambda _:None)
        assert len(count)==expected


def test_recovery24_scope_carries_actual_interruption_without_budget_reset():
    import json,hashlib
    from datetime import datetime,timezone
    from training.venus.paced_guard import AUTH,PLAN_V24,validate
    p=json.loads(PLAN_V24.read_text());r=json.loads(AUTH.read_text())['resources'];start=1800000000
    lease=dict(resource_key='venus-paced-recovery-v24',vm_id='321',disk_id=r['same_existing_disk_id'],zone=r['zone'],disk_size_gib=100,creation_timestamp='immutable',stop_unix=start+7200,prior_new_gpu_seconds_upper=6500,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,authorization_sha256=hashlib.sha256(AUTH.read_bytes()).hexdigest(),execution_plan_sha256=hashlib.sha256(PLAN_V24.read_bytes()).hexdigest(),verified_hourly_price_usd=3.68708363)
    vm=dict(id='321',name=p['vm_name'],zone='/'+r['zone'],status='RUNNING',machineType='/a2-highgpu-1g',creationTimestamp='immutable',lastStartTimestamp=datetime.fromtimestamp(start,timezone.utc).isoformat(),tags=dict(items=['duplex-venus-train-bounded']),disks=[dict(boot=True,autoDelete=False,source='/disks/'+r['same_existing_disk_name'])],scheduling=dict(provisioningModel='FLEX_START',maxRunDuration=dict(seconds=7200),instanceTerminationAction='STOP',automaticRestart=False))
    assert validate(vm,[vm],[],lease,start+1)==start+7200
    for wrong in ({'prior_new_gpu_seconds_upper':2600},{'stop_unix':start+10000},{'prior_gpu_seconds':40351}):
        with pytest.raises(AssertionError):validate(vm,[vm],[],dict(lease,**wrong),start+1)
