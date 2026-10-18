"""New TRAIN state contrasts plus recollected original TRAIN rehearsal.

No evaluated output, owner transcript or fresh evaluation file is read. Task
states/delivery fields below are declared fixtures, never actual playback proof.
"""
import copy
import threading
from data.synthetic.filter import norm_words

STATE_KINDS=('resume-active','resume-finished','resume-cancelled','resume-duplicates','resume-keep-work','cancel-new-answer','stop-only','clarify-target','finished-confirmation')
NAMES=('stencil batch','parcel plan','drum sequence','harp arrangement','fountain layout')
ITEMS=('travel mug','wooden ladder','glass jar','wool blanket','metal tray')
CONTINUE=(
    'Cancel delivery of the {target} result and carry on with the {item} instructions without starting over.',
    'Drop the {target} result. Finish the same {item} explanation from the place you paused, and keep the {other} work intact.',
    'Stop the {target} result delivery only; pick up the paused {item} instructions. Leave the {other} alone.',
    'Continue the {item} directions exactly where you left off; I have changed my mind about receiving the {target} result.',
    'Please go on with the unchanged {item} explanation, but withdraw delivery of the {target} answer, preserving the {other}.')


def new_scenes():
    scenes=[]
    for split,families,count in [('train',range(3),3),('heldout',range(3,5),2)]:
        for family in families:
            for variant in range(count):
                target=NAMES[family]+' '+str(variant+7);other=NAMES[(family+1)%5]+' '+str(variant+21)
                item=ITEMS[family];key=f'{split}-f{family}-v{variant}'
                for kind in STATE_KINDS:
                    # Same latest request and opaque IDs across all four resume states.
                    user=CONTINUE[family].format(target=target,other=other,item=item)
                    if kind=='resume-keep-work':user=f'Continue the same {item} directions from where you paused. Keep all calculation result deliveries active; do not cancel any of them.'
                    elif kind=='cancel-new-answer':user=f'Stop delivery of the {target} result. Replace the {item} explanation with the answer to {11+family+variant} plus {3+variant}; answer the new question too.'
                    elif kind=='stop-only':user=f'End the spoken {item} directions without any further words. Keep the calculations running.'
                    elif kind=='clarify-target':user='One calculation result should be stopped, but I have not chosen which. Ask me to identify it before changing anything.'
                    elif kind=='finished-confirmation':user=f'Withdraw the {target} result delivery. Tell me honestly if that task is already finished; leave the {other} alone.'
                    scenes.append(dict(id='state-'+key+'-'+kind,split=split,family=f'state-family-{family}',kind=kind,
                        target=target,other=other,item=item,user=user,number=14+family+2*variant,
                        task_ids=['job-'+str(101+family*17+variant*3),'job-'+str(102+family*17+variant*3),'job-'+str(103+family*17+variant*3)],
                        synthetic_fixture=True,real_playback=False,owner_data_used=False))
    return scenes


def specifications(preview=False):
    from training.control.paired_curriculum import specifications as paired
    from training.control.curriculum import specifications as original
    from training.control.capability_curriculum import specifications as capabilities
    rows=[]
    for corpus,source in [('paired',paired()),('rehearsal',original()),('capability',capabilities()),('state',new_scenes())]:
        for scene in source:
            rows.append(dict(id=scene['id'],split=scene['split'],kind=scene['kind'],family=corpus+'-'+scene['family'],
                domain=scene.get('domain','reasoning'),corpus=corpus,base=scene,synthetic_fixture=True))
    if preview:
        chosen={}
        for row in rows:
            if row['split']=='train':chosen.setdefault((row['corpus'],row['kind']),row)
        rows=list(chosen.values())+[r for r in rows if r['split']=='heldout']
    assert len({r['id'] for r in rows})==len(rows)
    if not preview:assert sum(r['split']=='train' for r in rows)==517 and sum(r['split']=='heldout' for r in rows)==204
    return rows


def state_world(scene):
    from evals.duplex_reasoning_worker import ReasoningWorld
    def forbidden(question):raise RuntimeError('TRAIN fixture cannot call an external reasoner')
    world=ReasoningWorld({},reasoner=forbidden);target,other,duplicate=scene['task_ids'];kind=scene['kind']
    for identity,name in [(target,scene['target']),(other,scene['other'])]:
        world.reserve(identity);world.tasks[identity].update(operation='delegate_reasoning',phase='running',question=f'Count valid arrangements for the {name}.')
    if kind in ('resume-finished','finished-confirmation'):world.tasks[target].update(closed=True,phase='completed')
    elif kind=='resume-cancelled':world.tasks[target]['cancelled']=True
    elif kind=='resume-duplicates':
        world.reserve(duplicate);world.tasks[duplicate].update(operation='delegate_reasoning',phase='running',question=world.tasks[target]['question'])
    question=f'Explain two steps for putting away a {scene["item"]}.'
    prepared=f'Place the {scene["item"]} in its storage space. Close the storage door.'
    context=[dict(role='user',text=f'Count valid arrangements for the {scene["target"]}.'),
        dict(role='user',text=f'Separately count valid arrangements for the {scene["other"]}.'),
        dict(role='user',text=question),dict(role='assistant',text=prepared,control_source='actual-speech-generation',
            speech_task_id='retained-directions',first_speech_event_id='fixture-cursor',latest_speech_event_id='fixture-cursor',
            playback_segments=1,speech_delivery='generated-not-fully-played',physical_hearing_confirmed=False),dict(role='user',text=scene['user'])]
    revision=world.reserve('latest',context)
    pending=[dict(id='retained-directions',request_text=question,text=prepared)]
    options=dict(resumable_speech=dict(speech_id='retained-directions',event_id='fixture-cursor'),bound_resume_target=True,allow_cancel_resume=True)
    return context,world,revision,pending,options


def compiler(generator,model):
    from evals.qwen35_controller import Qwen35ResponseExecutor
    from evals.application_capabilities import ApplicationCapabilityExecutor
    class Compile(Qwen35ResponseExecutor):
        provider_name='actual-api-teacher-native-compiler-no-qwen-inference'
        def __call__(self,payload):return self._request(payload)
    x=object.__new__(Compile);x.model_id=model;x.compact_replies=True;x.spoken_output_contract=False
    x.pending_speech_protocol='explicit-actions';x.prompt_profile='contract-examples';x.attention_backend='math'
    x.generator=generator;x.history={};x.counter=0;x.lock=threading.RLock()
    return ApplicationCapabilityExecutor(x)


def run(scene,generator,model,*,response_executor=None):
    from evals.idempotent_reasoning_worker import run_idempotent_reasoning_worker
    if response_executor is None:x=compiler(generator,model)
    else:
        from evals.application_capabilities import ApplicationCapabilityExecutor
        x=ApplicationCapabilityExecutor(response_executor)
    base=scene['base'];corpus=scene['corpus']
    if corpus in ('paired','rehearsal'):
        from training.control.paired_curriculum import payload
        prepared=dict(base,resume_target=base.get('resume_target'))
        request,world,revision,options=payload(prepared,model)
        world.reasoner=lambda question: (_ for _ in ()).throw(RuntimeError('No fixture reasoning execution'))
        if corpus=='paired' and base['kind']=='delegate':
            response=x(request)
            return dict(route='delegate-selection-only',content='',consume_pending_speech_ids=[],receipts=[],response=response,selection_only=True),world
        context=copy.deepcopy(base['context']);pending=copy.deepcopy(base['pending_speech'])
    elif corpus=='capability':
        from training.control.capability_curriculum import world_and_worker
        context,world,worker=world_and_worker(base,model)
        if base['domain']=='records':return worker(context,world,'latest',response_executor=x),world
        revision=world.tasks['latest']['revision'];pending=copy.deepcopy(base['pending_speech']);options={}
        if base['resume_target']:options=dict(resumable_speech=base['resume_target'],bound_resume_target=True,allow_cancel_resume=True)
    else:context,world,revision,pending,options=state_world(base)
    result=run_idempotent_reasoning_worker(context,world,'latest',model,revision,'state-aware',pending,x,**options)
    return result,world


def assess(scene,result,world):
    from evals.idempotent_reasoning_worker import require_idempotent_delivery
    base=scene['base'];corpus=scene['corpus']
    if corpus=='rehearsal':
        from training.control.curriculum import assess as old
        return old(base,result,world)
    if corpus=='paired':
        from training.control.paired_curriculum import assess as old
        if base['kind']=='delegate':return old(base,result['response'])
        event=result['events'][0];response=dict(id=event['response_id'],model=event['model'],status='completed',output=event['output'])
        return old(base,response)
    if corpus=='capability':
        from training.control.capability_curriculum import assess as old
        quality=old(base,result,world)
        if base['kind']=='finished-resume' and result.get('resume_after_idempotent_cancellation'):
            require_idempotent_delivery(result['receipts'],world.tasks)
            receipt=result['receipts'][0]
            quality['checks'].pop('no_receipt',None)
            quality['checks']['truthful_noop_receipt']=receipt['newly_cancelled_task_ids']==[] and sorted(receipt['task_ids'])==sorted(base['completed_calculations'])
            quality['passed']=all(quality['checks'].values())
        return quality
    kind=base['kind'];target,other,duplicate=base['task_ids']
    expected=[target,duplicate] if kind=='resume-duplicates' else [target] if kind in ('resume-active','cancel-new-answer') else [target] if kind=='resume-cancelled' else []
    checks=dict(actual_cancellations=sorted(k for k,v in world.tasks.items() if v['cancelled'])==sorted(expected),unrelated_preserved=not world.tasks[other]['cancelled'])
    if kind.startswith('resume-'):
        checks.update(route=result['route']=='resume',same_generator=result.get('resume_target')==dict(speech_id='retained-directions',event_id='fixture-cursor'),no_new_words=result['content']=='',preserved=result['consume_pending_speech_ids']==[])
        if result.get('resume_after_idempotent_cancellation'):require_idempotent_delivery(result['receipts'],world.tasks)
        # A cancellation request uses its truthful idempotent operation even when already finished.
        checks['requested_operation']=not result['receipts'] if kind=='resume-keep-work' else bool(result.get('resume_after_idempotent_cancellation'))
    elif kind=='cancel-new-answer':
        checks.update(route=result['route']=='cancel',old_superseded=result['consume_pending_speech_ids']==['retained-directions'],new_answer=' '.join(norm_words(str(base['number']))) in ' '.join(norm_words(result['content'])))
    elif kind=='stop-only':checks.update(route=result['route']=='silence',silent=result['content']=='',retired=result['consume_pending_speech_ids']==['retained-directions'])
    else:checks.update(route=result['route']==('clarify' if kind=='clarify-target' else 'answer'),preserved=result['consume_pending_speech_ids']==[],question='?' in result['content'] if kind=='clarify-target' else bool(result['content'].strip()),no_receipts=result['receipts']==[])
    return dict(passed=all(checks.values()),checks=checks)
