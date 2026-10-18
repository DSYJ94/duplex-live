"""New do-not-train control cases declared before observing their model output."""
import copy
import json


def cases():
    rows=[]
    for status in ('active','finished','already-cancelled','duplicate-active'):
        for variant in range(3):
            rows.append(dict(id=f'idempotent-fresh-{status}-{variant}',status=status,kind='resume',variant=variant))
    rows.extend(dict(id='idempotent-fresh-'+k,status='active',kind=k,variant=0) for k in ('new-answer','stop','ambiguous'))
    return rows


def fixture(scene,model):
    from evals.duplex_reasoning_worker import ReasoningWorld
    def forbidden(question):raise RuntimeError('External requests prohibited in this fixture')
    world=ReasoningWorld({},reasoner=forbidden);target='task-47';other='task-82';duplicate='task-96'
    for identity,question in [(target,'Count schedules for the harbour exercise.'),(other,'Count schedules for the orchard exercise.')]:
        world.reserve(identity);world.tasks[identity].update(operation='delegate_reasoning',phase='running',question=question)
    status=scene['status'];targets=[target]
    if status=='finished':world.tasks[target].update(closed=True,phase='completed')
    if status=='already-cancelled':world.tasks[target]['cancelled']=True
    if status=='duplicate-active':
        world.reserve(duplicate);world.tasks[duplicate].update(operation='delegate_reasoning',phase='running',question=world.tasks[target]['question']);targets.append(duplicate)
    request='Explain two steps for rinsing a porcelain cup.';prepared='Rinse the cup with warm water. Leave it upside down to dry.'
    context=[dict(role='user',text='Count schedules for the harbour exercise.'),dict(role='user',text='Independently count schedules for the orchard exercise.'),
        dict(role='user',text=request),dict(role='assistant',text=prepared,control_source='actual-speech-generation',speech_task_id='cup-instructions',
            first_speech_event_id='fixture-cup-cursor',latest_speech_event_id='fixture-cup-cursor',playback_segments=1,
            speech_delivery='generated-not-fully-played',physical_hearing_confirmed=False)]
    if scene['kind']=='resume':
        user=('Forget delivery of the harbour answer, leave the orchard work alone, and pick up your cup explanation where you left off.',
              'I do not want the harbour result anymore. Please finish the cup explanation you were in the middle of.',
              'End every harbour result delivery, preserve the orchard job, and carry on with the same cup instructions.')[scene['variant']]
    elif scene['kind']=='new-answer':user='Cancel delivery of the harbour result and replace your cup instructions with this new answer: how much is six plus three?'
    elif scene['kind']=='stop':user='Say nothing more. Leave both calculations running; I only want the spoken explanation stopped.'
    else:user='I want one calculation cancelled and the explanation continued, but first ask which calculation I mean.'
    context.append(dict(role='user',text=user));revision=world.reserve('latest',context)
    pending=[dict(id='cup-instructions',request_text=request,text=prepared)]
    options=dict(resumable_speech=dict(speech_id='cup-instructions',event_id='fixture-cup-cursor'),bound_resume_target=True,allow_cancel_resume=True)
    expected_new=targets if status in ('active','duplicate-active') and scene['kind'] in ('resume','new-answer') else []
    return world,context,revision,pending,options,targets,expected_new


def evaluate(executor,worker,folder):
    from data.synthetic.filter import norm_words
    from evals.idempotent_reasoning_worker import require_idempotent_delivery
    folder.mkdir();rows=[]
    for scene in cases():
        world,context,revision,pending,options,targets,expected=fixture(scene,executor.model_id)
        before=copy.deepcopy(world.tasks);row=dict(id=scene['id'],scene=scene,context=context,actual_world_before=before,passed=False)
        try:
            result=worker(context,world,'latest',executor.model_id,revision,'state-aware',pending,executor,**options)
            changed=sorted(t for t,v in world.tasks.items() if v['cancelled'] and not before[t]['cancelled'])
            checks=dict(exact_new_cancellations=changed==sorted(expected),unrelated_work_unchanged=world.tasks['task-82']==before['task-82'])
            kind=scene['kind'];content=result['content']
            if kind=='resume':checks.update(route=result['route']=='resume',same_generator=result.get('resume_target')==options['resumable_speech'],no_new_words=content=='',speech_preserved=result['consume_pending_speech_ids']==[])
            elif kind=='new-answer':checks.update(route=result['route']=='cancel',answer='nine' in norm_words(content),old_answer_superseded=result['consume_pending_speech_ids']==['cup-instructions'])
            elif kind=='stop':checks.update(route=result['route']=='silence',no_new_words=content=='',speech_retired=result['consume_pending_speech_ids']==['cup-instructions'])
            else:checks.update(route=result['route']=='clarify',question='?' in content,speech_preserved=result['consume_pending_speech_ids']==[])
            if result.get('resume_after_idempotent_cancellation'):require_idempotent_delivery(result['receipts'],world.tasks)
            row.update(result=result,checks=checks,passed=all(checks.values()))
        except Exception as error:row.update(error_type=type(error).__name__,message=str(error),evidence=getattr(error,'result',None))
        row['actual_world_after']=world.tasks;rows.append(row);(folder/(scene['id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    report=dict(cases=len(rows),passed=sum(r['passed'] for r in rows),results=rows,do_not_train=True,actual_external_requests=0,physical_playback_tested=False)
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n');return {k:v for k,v in report.items() if k!='results'}
