"""Rebuild an actual failed synthetic controller request, without executing it."""
import copy
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def capture_failure(path):
    from evals.duplex_reasoning_worker import ReasoningWorld,run_reasoning_worker
    failure=json.loads(Path(path).read_text());task=failure['task'];identity=task['task_id']
    if failure['schema']!='dl-hybrid-controller-failure-v1' or failure['hybrid_system_qualified']:raise ValueError('Expected unqualified recorded failure')
    world=ReasoningWorld({});world.tasks=copy.deepcopy(failure['actual_world_tasks'])
    context=copy.deepcopy(world.tasks[identity]['context']);saved=failure['controller_evidence']['local_generation']
    class Capture:
        provider_name='fixture-only-failed-request-capture'
        def __call__(self,p):
            self.payload=copy.deepcopy(p)
            return dict(id='capture-only',model='Qwen/Qwen3.5-9B',status='completed',output=[dict(type='function_call',name='respond_to_user',arguments=json.dumps(dict(route='answer',content='Request capture only.',consume_pending_speech_ids=[])))])
    recorder=Capture();options={}
    if task.get('resume_target') is not None:options=dict(resumable_speech=task['resume_target'],bound_resume_target=True,allow_cancel_resume=True)
    run_reasoning_worker(context,world,identity,'Qwen/Qwen3.5-9B',task['revision'],'state-aware',task['pending_speech'],recorder,**options)
    return recorder.payload,dict(source_failure_sha256=sha256(path),original_input_tokens=saved['input_tokens'],original_raw_text=saved['raw_text'],original_generated_tokens=saved['generated_tokens'],actual_context_rebuilt=True)
