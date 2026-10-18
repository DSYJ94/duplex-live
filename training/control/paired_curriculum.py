"""Authored contrastive TRAIN/holdout scenes for speech obligations and routing.

No captured voice, actual owner transcript, teacher result or evaluation file is
read. Declared speech and task states are fixtures, never playback evidence.
"""
import copy,json
from training.control.curriculum import xml_action

KINDS=('replace','preserve','fulfill','withdraw','ordinary','resume','cancel-resume','delegate','clarify')
PATTERNS=(
 ('Discard the earlier instructions. What is {math}? Say only the number.', 'Keep the earlier instructions for later. First, what is {math}? Say only the number.'),
 ('I no longer want those instructions. Give just the answer to {math}.', 'I still want those instructions afterwards. Give just the answer to {math} first.'),
 ('Leave the previous answer unfinished and tell me only the result of {math}.', 'Leave the previous answer waiting while you tell me only the result of {math}.'),
 ('Replace the old reply with only the number for {math}.', 'Preserve the old reply; before it, say only the number for {math}.'))


def specifications(preview=False):
    from data.synthetic.filter import _num2words
    rows=[]
    for split,families,count in [('train',(0,) if preview else (0,1),1 if preview else 6),('heldout',(2,) if preview else (2,3),1 if preview else 3)]:
        for family in families:
            for variant in range(count):
                a=(4+variant*2+family if split=='train' else 35+variant*3+family)
                b=2+family;value=a+b;math=_num2words(a)+' plus '+_num2words(b)
                item=(['fan','lamp','basket','cushion','folder','bottle'] if split=='train' else ['compass','lantern','bucket'])[variant]
                place='table' if split=='train' else 'shelf';pair=f'{split}-f{family}-v{variant}'
                sid='owed-'+pair;tid='calculation-'+pair
                question=f'Give two instructions for putting the {item} away.'
                prepared=f'Place the {item} on the {place}. Leave it beside the window.'
                for kind in KINDS:
                    pending=[dict(id=sid,request_text=question,text=prepared)]
                    context=[dict(role='user',text=question)];active=[];resume=None
                    expected=dict(route='answer',consumed=[],cancelled=[],number=_num2words(value))
                    if kind in ('replace','preserve'):user=PATTERNS[family][kind=='preserve'].format(math=math)
                    elif kind=='fulfill':
                        user=['Please give the complete reply you have prepared.','Go ahead and deliver the prepared instructions.','Now say the answer that is waiting.','You can give that ready response now.'][family]
                        expected.update(consumed=[sid],exact_content=prepared);expected.pop('number')
                    elif kind=='withdraw':
                        user=['Say nothing further; I do not want the old answer.','Withdraw those instructions without any spoken reply.','Stop the pending answer and remain silent.','No further speech is wanted.'][family]
                        expected.update(route='silence',consumed=[sid],exact_content='');expected.pop('number')
                    elif kind=='ordinary':
                        user=['What is {math}? Say only the number.','Give just the result of {math}.','Say only the answer to {math}.','How much is {math}? The number alone, please.'][family].format(math=math)
                        pending=[];context=[]
                    elif kind in ('resume','cancel-resume'):
                        context.append(dict(role='assistant',text=prepared,control_source='actual-speech-generation',speech_task_id=sid,first_speech_event_id='fixture-generator',latest_speech_event_id='fixture-generator',playback_segments=1,speech_delivery='generated-not-fully-played',physical_hearing_confirmed=False))
                        resume=dict(speech_id=sid,event_id='fixture-generator');expected.update(route='resume',exact_content='');expected.pop('number')
                        if kind=='resume':user=['Please carry on with the instructions.','Continue from where you paused.','Go on with that same answer.','Finish the explanation you were giving.'][family]
                        else:
                            active=[tid];expected['cancelled']=[tid]
                            user=['Cancel the background calculation and carry on with the instructions.','Stop delivery of that calculation result; resume your instructions.','I no longer need the calculation result. Continue your paused answer.','Withdraw the calculation result and finish that same explanation.'][family]
                    elif kind=='delegate':
                        pending=[];context=[];expected=dict(native_function='delegate_reasoning',zero_arguments=True)
                        nums=[17+family+variant+3*i for i in range(18)]
                        target=sum(nums[::4]);user=f'For the eighteen distinct numbers {nums}, count every subset summing to {target}. Each number can be used at most once; include the empty subset only if it meets the sum. Give only the count.'
                    else:
                        pending=[];context=[];active=[tid,'other-'+pair];expected=dict(route='clarify',consumed=[],cancelled=[],question_required=True)
                        user=['Cancel one calculation, but ask which one before acting.','I want to stop a result, but I have not said which calculation.','Ask me which of the two calculations to cancel.','I have not identified the unwanted calculation. Clarify it before cancelling.'][family]
                    if kind=='replace':expected['consumed']=[sid]
                    context.append(dict(role='user',text=user))
                    rows.append(dict(id=f'paired-{pair}-{kind}',pair_id=pair,split=split,family=f'family-{family}',kind=kind,context=context,pending_speech=pending,active_calculations=active,resume_target=resume,expected=expected,synthetic_fixture=True,simulated_delivery_fields_are_not_real_speech=True,audio_generated=False,provenance=dict(source_model='GPT-6 Codex procedural authoring',date='2026-10-18',class_='closed-content')))
    return rows


def payload(scene,model):
    from evals.duplex_reasoning_worker import ReasoningWorld,run_reasoning_worker
    world=ReasoningWorld({})
    for tid in scene['active_calculations']:world.reserve(tid);world.tasks[tid].update(operation='delegate_reasoning',phase='running')
    context=copy.deepcopy(scene['context']);revision=world.reserve('latest',context)
    class Capture:
        provider_name='fixture-only-paired-payload-capture'
        def __call__(self,p):
            self.value=copy.deepcopy(p)
            return dict(id='fixture-only',model=model,status='completed',output=[dict(type='function_call',name='respond_to_user',arguments=json.dumps(dict(route='answer',content='Payload capture only.',consume_pending_speech_ids=[])))])
    capture=Capture();options={}
    if scene['resume_target'] is not None:options=dict(resumable_speech=scene['resume_target'],bound_resume_target=True,allow_cancel_resume=True)
    run_reasoning_worker(context,world,'latest',model,revision,'state-aware',copy.deepcopy(scene['pending_speech']),capture,**options)
    return capture.value,world,revision,options


def compile_scene(scene,generator):
    from evals.qwen35_controller import Qwen35ResponseExecutor
    model=getattr(generator,'model_id','procedural-authoring');p,_,_,_=payload(scene,model)
    executor=object.__new__(Qwen35ResponseExecutor);executor.compact_replies=True;executor.spoken_output_contract=False
    executor.pending_speech_protocol='explicit-actions';executor.prompt_profile='contract-examples';executor.attention_backend='math'
    executor.model_id=model;executor.generator=generator;executor.history={};executor.counter=0
    return executor._request(p)


def assess(scene,response):
    from data.synthetic.filter import norm_words
    from evals.duplex_reasoning_worker import run_reasoning_worker
    expected=scene['expected'];calls=response.get('output',[])
    if response.get('status')!='completed' or len(calls)!=1:return dict(passed=False,checks=dict(single_call=False))
    call=calls[0];args=json.loads(call['arguments']);checks={}
    if scene['kind']=='delegate':
        checks.update(native_function=call['name']=='delegate_reasoning',zero_arguments=args=={})
        return dict(passed=all(checks.values()),checks=checks,stage='action-selection-only-no-remote-dispatch')
    p,world,revision,options=payload(scene,response['model'])
    class Replay:
        provider_name='recorded-synthetic-label'
        def __call__(self,new):
            if new!=p:raise ValueError('Worker replay changed fixture payload')
            return response
    result=run_reasoning_worker(copy.deepcopy(scene['context']),world,'latest',response['model'],revision,'state-aware',copy.deepcopy(scene['pending_speech']),Replay(),**options)
    checks.update(route=result['route']==expected['route'],speech_retirement=result['consume_pending_speech_ids']==expected['consumed'],cancellation=sorted(t for t in scene['active_calculations'] if world.tasks[t]['cancelled'])==sorted(expected['cancelled']))
    if 'exact_content' in expected:checks['content']=result['content']==expected['exact_content']
    if 'number' in expected:checks['number']=norm_words(result['content'])==norm_words(expected['number'])
    if expected.get('question_required'):checks['question']=bool(result['content'].strip()) and '?' in result['content']
    if scene['resume_target'] is not None and expected['route']=='resume':checks['same_generator_identity']=result.get('resume_target')==scene['resume_target']
    return dict(passed=all(checks.values()),checks=checks,stage='fixture-worker-action-and-state',result={k:result[k] for k in ('route','content','consume_pending_speech_ids','receipts')})
