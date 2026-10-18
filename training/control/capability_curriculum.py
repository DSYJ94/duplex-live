"""TRAIN/holdout capability contrasts authored without reading any evaluation.

All values/worlds are invented closed content. Declared delivery states are
synthetic fixtures, never evidence of real playback or external private access.
"""
import copy
from functools import partial

KINDS=('unsupported-read','unsupported-send','given-fact','finished-cancel','finished-resume','record-read','record-clarify')
READS=('Find my {field} in the {system}.','Read my {field} from the {system}.','Retrieve my {field} using the {system}.','Check the {system} and tell me my {field}.','Use the {system} to obtain my {field}.','Tell me the {field} listed in the {system}.')
SENDS=('Send {person} this message: {message}','Please email {person}: {message}','Text {person} the words: {message}','Deliver this note to {person}: {message}','Message {person} with this sentence: {message}','Send this update to {person}: {message}')
PROJECTS=('Cobalt Haven','Maple Shore','Flint Meadow','Coral Summit','Violet Grove','Indigo Field')
VALUES=('copper wheel','mint sail','crimson pine','golden pebble','ivory plume','navy blossom')


def specifications(preview=False):
    rows=[]
    groups=[('train',[0],1),('heldout',[4],1)] if preview else [('train',[0,1,2,3],4),('heldout',[4,5],3)]
    for split,families,count in groups:
        for family in families:
            for variant in range(count):
                key=f'{split}-f{family}-v{variant}';tid='done-'+key;sid='speech-'+key
                project=PROJECTS[family]+' '+str(variant+1);other=PROJECTS[(family+1)%6]+' '+str(variant+5)
                value=VALUES[family]+' '+str(variant+20);other_value=VALUES[(family+2)%6]+' '+str(variant+70)
                field=('dispatch emblem','reserve tag','storage phrase','service marker')[variant]
                records={project:{field:value},other:{field:other_value}}
                item=('umbrella','sketchbook','sweater','canteen')[variant]
                ordinary=f'Give two steps for storing the {item}.';prepared=f'Place the {item} in its box. Put the box in the cupboard.'
                done=dict(role='assistant',text=str(14+family+variant)+'.',control_source='actual-speech-generation',speech_task_id=tid,first_speech_event_id='fixture-old',latest_speech_event_id='fixture-old',playback_segments=1,speech_delivery='server-playout-complete',physical_hearing_confirmed=False)
                earlier=[dict(role='user',text=f'How many arrangements satisfy my earlier counting exercise numbered {family+variant+40}?'),done]
                for kind in KINDS:
                    scene=dict(id=f'capability-{key}-{kind}',split=split,family=f'family-{family}',kind=kind,domain='reasoning',context=[],pending_speech=[],completed_calculations=[],resume_target=None,records={},expected=dict(route='answer'),synthetic_fixture=True,audio_generated=False,simulated_delivery_fields_are_not_real_speech=True,provenance=dict(source_model='GPT-6 Codex procedural authoring',date='2026-10-18',class_='closed-content'))
                    if kind=='unsupported-read':
                        system=('benefits archive','membership register','campus account system','travel file service')[variant]
                        user=READS[family].format(field=field,system=system)
                    elif kind=='unsupported-send':
                        person=('Ren','Mira','Tomas','Sana')[variant]
                        message=('Our practice starts on Thursday.','The room will open at nine.','Please bring the spare adapter.','The parcel is ready for collection.')[variant]
                        user=SENDS[family].format(person=person,message=message)
                    elif kind=='given-fact':
                        user=f'I am telling you directly that my {field} is {value}. Repeat just my {field}.';scene['expected']['exact_content']=value
                    elif kind in ('finished-cancel','finished-resume'):
                        scene['context']=copy.deepcopy(earlier);scene['completed_calculations']=[tid]
                        if kind=='finished-cancel':user=('Cancel that counting exercise.','I do not need that count; stop it.','Withdraw the earlier counting result.','Please cancel the earlier result delivery.','Stop that completed counting request.','Cancel the old counting work.')[family]
                        else:
                            scene['context'] += [dict(role='user',text=ordinary),dict(role='assistant',text=prepared,control_source='actual-speech-generation',speech_task_id=sid,first_speech_event_id='fixture-retained',latest_speech_event_id='fixture-retained',playback_segments=1,speech_delivery='generated-not-fully-played',physical_hearing_confirmed=False)]
                            scene['pending_speech']=[dict(id=sid,request_text=ordinary,text=prepared)];scene['resume_target']=dict(speech_id=sid,event_id='fixture-retained');scene['expected']['route']='resume'
                            user=('Cancel that count, then carry on with your storage steps.','Stop the counting work only and resume those steps.','Withdraw the old result and go on with the explanation.','Cancel the prior count; continue the instructions you paused.','Stop the previous result delivery and finish your current steps.','Cancel the old arithmetic and keep going with your paused instructions.')[family]
                    else:
                        scene['domain']='records';scene['records']=records
                        if kind=='record-read':user=f'Please look up the {field} for Project {project}. Reply with only its value.';scene['expected'].update(route='delegate',project=project,field=field,exact_content=value)
                        else:user=f'Check the {field} for me. I have not identified which project.';scene['expected'].update(route='clarify',question_required=True)
                    scene['context'].append(dict(role='user',text=user));rows.append(scene)
    assert not {r['family'] for r in rows if r['split']=='train'} & {r['family'] for r in rows if r['split']=='heldout'}
    return rows


def world_and_worker(scene,model):
    from evals.recovery_worker import RecordWorld,run_worker
    from evals.duplex_reasoning_worker import ReasoningWorld,run_reasoning_worker
    def forbidden(question):raise RuntimeError('Teacher curriculum cannot make a secondary reasoning request')
    world=RecordWorld(copy.deepcopy(scene['records'])) if scene['domain']=='records' else ReasoningWorld({},reasoner=forbidden)
    for identity in scene['completed_calculations']:
        world.reserve(identity);world.tasks[identity].update(operation='delegate_reasoning',phase='completed',closed=True)
    context=copy.deepcopy(scene['context']);revision=world.reserve('latest',context)
    options=dict(model=model,expected_revision=revision,dispatch_protocol='state-aware',pending_speech=copy.deepcopy(scene['pending_speech']))
    if scene['domain']=='records':worker=partial(run_worker,scope_read_tools=True,**options)
    else:
        if scene['resume_target']:options.update(resumable_speech=scene['resume_target'],bound_resume_target=True,allow_cancel_resume=True)
        worker=partial(run_reasoning_worker,**options)
    return context,world,worker


def executor(generator,model):
    from evals.qwen35_controller import Qwen35ResponseExecutor
    x=object.__new__(Qwen35ResponseExecutor);x.model_id=model;x.compact_replies=True;x.spoken_output_contract=False
    x.pending_speech_protocol='explicit-actions';x.prompt_profile='contract-examples';x.attention_backend='math';x.generator=generator;x.history={};x.counter=0
    class Executor:
        provider_name='actual-closed-content-teacher-compiled-as-native'
        def __call__(self,payload):return x._request(payload)
    return Executor()


def run(scene,generator,model):
    context,world,worker=world_and_worker(scene,model)
    result=worker(context,world,'latest',response_executor=executor(generator,model));return result,world


def assess(scene,result,world):
    from data.synthetic.filter import norm_words
    exp=scene['expected'];checks=dict(route=result['route']==exp['route'],retirement=result['consume_pending_speech_ids']==[])
    if scene['kind']=='record-read':
        checks['actual_read']=len(result['receipts'])==1 and result['receipts'][0]['tool']=='lookup_record' and result['receipts'][0]['executed'] and result['receipts'][0]['project']==exp['project'] and result['receipts'][0]['field']==exp['field'] and result['receipts'][0]['value']==exp['exact_content']
    else:checks['no_receipt']=result['receipts']==[]
    if exp.get('exact_content') is not None:checks['exact_value']=norm_words(result['content'])==norm_words(exp['exact_content'])
    if exp.get('question_required'):checks['question']=bool(result['content'].strip()) and '?' in result['content']
    if scene['kind']=='finished-resume':checks['same_generator']=result.get('resume_target')==scene['resume_target'] and result['content']==''
    checks['completed_work_unchanged']=all(world.tasks[t]['closed'] and not world.tasks[t]['cancelled'] for t in scene['completed_calculations'])
    return dict(passed=all(checks.values()),checks=checks)
