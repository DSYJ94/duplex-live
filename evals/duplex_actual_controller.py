"""Reconstruct an actual saved native controller proposal, without executing it.

The complete real state/context and original native output are verified first.
The dummy response below only captures a payload; it is not a target or answer.
"""
import copy,json
from pathlib import Path
from evals.moshi_model_download import sha256


def capture(root,task):
    from evals.duplex_reasoning_worker import ReasoningWorld,run_reasoning_worker
    suite=json.loads((root/'report.json').read_text());case=suite['results'][0];folder=root/case['scenario']
    meta=json.loads((folder/(task+'-backend.json')).read_text());old=meta['result']['events'][0]['local_inference']
    state=old['compact_reply_action_protocol'];context=copy.deepcopy(case['world_tasks'][task]['context'])
    world=ReasoningWorld({});world.tasks=copy.deepcopy(state['available']['tasks'])
    world.tasks[task]=copy.deepcopy(case['world_tasks'][task]);world.tasks[task].update(closed=False,cancelled=False,phase='pending')
    class Capture:
        provider_name='fixture-only-payload-capture'
        def __call__(self,payload):
            self.payload=payload
            return dict(id='capture-only',status='completed',model='fixture-only',output=[dict(type='function_call',name='respond_to_user',arguments=json.dumps(dict(route='answer',content='Capture fixture only.',consume_pending_speech_ids=[])))])
    cap=Capture();target=meta.get('resume_target');options={}
    if target is not None:options=dict(resumable_speech=target,bound_resume_target=state.get('resume_action_protocol')=='application-bound-single-retained-generator',allow_cancel_resume=True)
    run_reasoning_worker(context,world,task,'Qwen/Qwen3.5-9B',meta['revision'],'state-aware',meta['pending_speech'],cap,**options)
    if cap.payload['local_worker_state']!=state or json.loads(cap.payload['input'])!=context:raise ValueError('Reconstructed actual native state/context differs')
    return cap.payload,dict(source_report_sha256=sha256(root/'report.json'),source_backend_sha256=sha256(folder/(task+'-backend.json')),original_input_tokens=old['input_tokens'],original_native_action=old['native_model_action'],original_raw_text=old['raw_text'],original_generated_tokens=old['generated_tokens'],actual_state_exact=True)
