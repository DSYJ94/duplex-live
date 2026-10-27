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


class NativeWorkerProtocolTests(unittest.IsolatedAsyncioTestCase):
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
