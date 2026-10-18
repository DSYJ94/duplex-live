"""Declared synthetic capability holdouts; no captured conversation or TRAIN inputs."""
import copy
import json
from pathlib import Path


def payload(scene,model):
    from evals.duplex_reasoning_worker import ReasoningWorld,run_reasoning_worker
    def forbidden(question):raise RuntimeError('No external request is permitted in this fixture evaluation')
    world=ReasoningWorld({},reasoner=forbidden)
    for identity in scene.get('completed_calculations',[]):
        world.reserve(identity);world.tasks[identity].update(operation='delegate_reasoning',phase='completed',closed=True)
    context=copy.deepcopy(scene['context']);revision=world.reserve('latest',context)
    class Capture:
        provider_name='fixture-only-affordance-payload-capture'
        def __call__(self,p):
            self.payload=copy.deepcopy(p)
            return dict(id='capture-only',model=model,status='completed',output=[dict(type='function_call',name='respond_to_user',arguments=json.dumps(dict(route='answer',content='Fixture capture only.',consume_pending_speech_ids=[])))])
    recorder=Capture();options={}
    if scene.get('resume_target'):options=dict(resumable_speech=scene['resume_target'],bound_resume_target=True,allow_cancel_resume=True)
    run_reasoning_worker(context,world,'latest',model,revision,'state-aware',scene['pending_speech'],recorder,**options)
    return recorder.payload,world,revision,options


def evaluate(executor,folder):
    from evals.duplex_reasoning_worker import run_reasoning_worker
    scenes=json.loads(Path('evals/fixtures/fresh-capability-controls-20261018.json').read_text())['cases']
    folder.mkdir();rows=[]
    for scene in scenes:
        row=dict(id=scene['id'],scene=scene,completed=False,mechanics_passed=False,semantic_review_required=scene['expected']['route']!='resume')
        try:
            request,world,revision,options=payload(scene,executor.model_id);response=executor(copy.deepcopy(request))
            class Replay:
                provider_name='recorded-actual-local-controller'
                def __call__(self,p):
                    if p!=request:raise ValueError('Fixture worker request changed')
                    return response
            result=run_reasoning_worker(copy.deepcopy(scene['context']),world,'latest',executor.model_id,revision,'state-aware',scene['pending_speech'],Replay(),**options)
            checks=dict(route=result['route']==scene['expected']['route'],no_receipt=result['receipts']==[],no_speech_retired=result['consume_pending_speech_ids']==[])
            if scene.get('resume_target'):checks['same_generator']=result.get('resume_target')==scene['resume_target']
            if scene['expected'].get('exact_content') is not None:checks['content']=result['content'].strip().rstrip('.').lower()==scene['expected']['exact_content'].lower()
            row.update(completed=True,response=response,result=result,checks=checks,mechanics_passed=all(checks.values()))
        except Exception as error:row.update(error_type=type(error).__name__,message=str(error),evidence=getattr(error,'result',None))
        rows.append(row);(folder/(scene['id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    report=dict(cases=len(rows),completed=sum(r['completed'] for r in rows),mechanics_passed=sum(r['mechanics_passed'] for r in rows),semantic_review_pending=True,actual_external_requests=0,results=rows)
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n');return {k:v for k,v in report.items() if k!='results'}
