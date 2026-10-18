"""New authored text-control scenes; no voice evaluation or captured API result.

All conversations and running tasks are DECLARED SYNTHETIC FIXTURES. Runtime
speech-delivery fields below are simulated inputs, never real playback evidence.
Template families and numeric ranges separate TRAIN from held-out text checks.
"""
import copy
import hashlib
import json

TEMPLATES={
 'quiet':[
  'Please stop speaking now; there is no other request.',
  'You can leave that explanation unfinished. Say nothing further.',
  'End your spoken answer and remain silent.',
  'I no longer need that response. Please be silent.',
  'No more talking about that, thank you.',
  'Leave the rest unsaid. I want silence now.'],
 'replace':[
  'Set aside the instructions and answer this instead: what is {math}? Give only the result.',
  'Do not finish the earlier answer. Tell me only the result of {math}.',
  'Withdraw the old response, then say just the answer to {math}.',
  'Leave those instructions unfinished; my new question is {math}. Say the result only.',
  'Skip everything else you were saying. How much is {math}? Just the result.',
  'Forget the remaining instructions and give me the number for {math}, with no working.'],
 'first':[
  'Replace the full answer by repeating its first instruction word for word.',
  'I only need the first sentence of those instructions. Quote it exactly.',
  'Do not finish the long version; repeat only the initial instruction exactly as given.',
  'Give just sentence one from that explanation, without changing its words.',
  'Keep only the opening instruction, using your exact earlier wording.',
  'Your next reply should be precisely the first sentence you gave, and nothing else.'],
 'last':[
  'Replace those instructions with only their final sentence, word for word.',
  'I only need the last sentence of the instructions. Repeat it exactly.',
  'Instead of finishing the explanation, say only its final instruction unchanged.',
  'Repeat just sentence three, using the same words as before.',
  'Skip to your closing instruction and reproduce that sentence exactly.',
  'Only your final sentence matters now. Give it with no alterations.'],
 'cancel-preserve':[
  'Cancel the {selected} calculation result, but leave the explanation to finish.',
  'Keep the instructions queued and stop delivery of the {selected} calculation.',
  'I still want your explanation. Cancel only the {selected} result.',
  'Withdraw the background {selected} calculation, without withdrawing your ordinary answer.',
  'Preserve the unfinished instructions; I want the {selected} result cancelled.',
  'End delivery of the {selected} computation and keep your explanation outstanding.'],
 'cancel-one':[
  'Cancel only the {selected} calculation result. Keep the {other} calculation running.',
  'Do not deliver the {selected} result; preserve the {other} job.',
  'The {selected} computation is no longer needed. Do not cancel {other}.',
  'Keep {other} active and cancel result delivery for {selected} alone.',
  'Withdraw just the {selected} result, leaving {other} untouched.',
  'I want the {other} calculation to continue. Stop only the {selected} result.'],
 'no-active':[
  'Is the {selected} calculation currently active? Answer only yes or no.',
  'Does the {selected} calculation remain active right now? Give only yes or no.',
  'Is the {selected} computation still running? Please say just yes or no.',
  'Do I have an active {selected} calculation at present? Answer with yes or no only.',
  'Is there an ongoing {selected} task you could cancel? Just yes or no.',
  'Is the {selected} computation awaiting completion right now? Respond only yes or no.'],
 'ordinary':[
  'What is {math}? Say only the result.',
  'Tell me the result of {math}, without any working.',
  'Please calculate {math} and give only the number.',
  'Answer with just the result: {math}.',
  'How much is {math}? Give the answer alone.',
  'I need only the number obtained from {math}.'],
 'clarify':[
  'Cancel one of the calculations, but I have not decided which. Ask me which one.',
  'I want to stop a calculation result. I have not identified it yet.',
  'One of those two calculations should be cancelled. Ask which before acting.',
  'Cancel a background calculation, but please get its identity from me first.',
  'There is one calculation I no longer need; you must ask me which one.',
  'I need to withdraw one result, but have not told you whose. Clarify that first.'],
}


def specifications(preview=False):
    from data.synthetic.filter import _num2words
    rows=[]
    objects=['disk','card','ribbon','button','tile','coin','label','token','tag','ring','peg','clip']
    for split,families,count in [('train',range(4),2 if preview else 6),('heldout',range(4,6),1 if preview else 4)]:
        for kind,templates in TEMPLATES.items():
            for family in families:
                # Preview preserves whole families; fewer scenes, no row filtering.
                if preview and split=='train' and family>0:continue
                for variant in range(count):
                    a=(3+variant*2+family if split=='train' else 24+variant*2+family)
                    b=(2+family if split=='train' else 13+family)
                    op='plus' if variant%2==0 else 'minus';number=a+b if op=='plus' else a-b
                    math=f'{_num2words(a)} {op} {_num2words(b)}';answer=_num2words(number).capitalize()+'.'
                    serial=len(rows)+1;speech=f'ordinary-{serial:04d}';names=['inventory','postage','payroll','freight','stock','shipping','billing','packing'] if split=='train' else ['savings','expenses','tax','budget','rent','profit','insurance','sales']
                    selected=names[(variant+family)%len(names)];other=names[(variant+family+1)%len(names)]
                    choices=objects if split=='train' else ['marble','spool','cube','brush','pouch','stick','pin','seal','cap','ruler','key','rubber']
                    item=choices[(variant+family*2)%len(choices)];surface='tray' if split=='train' else 'board';container='box' if split=='train' else 'drawer'
                    prior=[f'First, place the {item} on the {surface}.',f'Next, arrange a marker beside the {item}.',f'Last, close the {container}.']
                    request=f'Describe three instructions for arranging the {item} display.';text=' '.join(prior)
                    # This explicitly declared simulated state mirrors runtime shape.
                    context=[dict(role='user',text=request),dict(role='assistant',text=text,control_source='actual-speech-generation',speech_task_id=speech,
                        first_speech_event_id='fixture-generation',latest_speech_event_id='fixture-generation',playback_segments=1,
                        speech_delivery='generated-not-fully-played',physical_hearing_confirmed=False)]
                    pending=[dict(id=speech,request_text=request,text=text)];tasks=[]
                    if kind in ('cancel-preserve','cancel-one','clarify'):tasks=[selected]+([other] if kind in ('cancel-one','clarify') else [])
                    if kind in ('ordinary','no-active','cancel-one','clarify'):context=[];pending=[]
                    user=templates[family].format(math=math,selected=selected,other=other);context.append(dict(role='user',text=user))
                    expected=dict(route='answer',consumed=[],cancelled=[])
                    if kind=='quiet':expected.update(route='silence',consumed=[speech],exact_content='')
                    elif kind=='replace':expected.update(consumed=[speech],normalized_content=_num2words(number))
                    elif kind=='first':expected.update(consumed=[speech],normalized_content=prior[0])
                    elif kind=='last':expected.update(consumed=[speech],normalized_content=prior[-1])
                    elif kind in ('cancel-preserve','cancel-one'):expected.update(route='cancel',cancelled=[selected])
                    elif kind=='no-active':expected.update(normalized_content='No')
                    elif kind=='ordinary':expected.update(normalized_content=_num2words(number))
                    elif kind=='clarify':expected.update(route='clarify')
                    row=dict(id=f'control-{split}-{kind}-family{family}-{variant}',split=split,kind=kind,family=f'{kind}-{family}',
                        context=context,pending_speech=pending,active_calculations=tasks,expected=expected,
                        synthetic_fixture=True,simulated_delivery_fields_are_not_real_speech=True,audio_generated=False,
                        provenance=dict(content='agent-authored procedural scene',source_model='GPT-6 Codex procedural code authoring',created_date='2026-10-18',
                            provenance_class='closed-content',authorization='ADR-0003 Amendment A, owner enterprise agreement'),
                        arithmetic=dict(a=a,b=b,operation=op,result=number))
                    rows.append(row)
    return rows


def xml_action(action):
    parts=[]
    for name,value in action['arguments'].items():
        text=value if isinstance(value,str) else json.dumps(value)
        if '</parameter>' in text or '</tool_call>' in text:raise ValueError('Unserializable native argument')
        parts.append(f'<parameter={name}>{text}</parameter>')
    return '<tool_call>\n<function='+action['name']+'>\n'+'\n'.join(parts)+'\n</function>\n</tool_call>'


def assess(scene,result,world):
    from data.synthetic.filter import norm_words
    e=scene['expected'];checks=dict(route=result['route']==e['route'],speech_retirement=result['consume_pending_speech_ids']==e['consumed'],
        cancellation=sorted(t for t in scene['active_calculations'] if world.tasks[t]['cancelled'])==sorted(e['cancelled']))
    if 'exact_content'in e:checks['content']=result['content']==e['exact_content']
    elif 'normalized_content'in e:checks['content']=norm_words(result['content'])==norm_words(e['normalized_content'])
    elif scene['kind']=='clarify':checks['content']=bool(result['content'].strip()) and '?'in result['content']
    else:
        # Actual cancellation receipt already proves target and preservation.
        # Never approve a claim that remote execution itself was aborted.
        words=' '.join(norm_words(result['content']));checks['content']=bool(words) and not any(p in words for p in ('computation was aborted','execution was aborted','stopped computing','terminated the computation'))
    return dict(passed=all(checks.values()),checks=checks)


def label_scene(scene,generator):
    from evals.qwen35_controller import Qwen35ResponseExecutor
    from evals.duplex_reasoning_worker import ReasoningWorld,run_reasoning_worker
    class CompileExecutor(Qwen35ResponseExecutor):
        # Prompt/schema compilation only; this object never loads a Qwen model.
        provider_name='synthetic-control-label-compiler'
        def __call__(self,payload):return self._request(payload)
    executor=object.__new__(CompileExecutor);executor.compact_replies=True;executor.spoken_output_contract=False
    executor.pending_speech_protocol='explicit-actions';executor.prompt_profile='contract-examples';executor.attention_backend='math'
    executor.model_id=getattr(generator,'model_id','procedural-authoring');executor.generator=generator;executor.history={};executor.counter=0
    def blocked(question):raise RuntimeError('Control labels cannot execute a remote reasoning request')
    world=ReasoningWorld({},blocked)
    for task in scene['active_calculations']:world.reserve(task);world.tasks[task].update(operation='delegate_reasoning',phase='running')
    context=copy.deepcopy(scene['context']);revision=world.reserve('latest',context)
    result=run_reasoning_worker(context,world,'latest',executor.model_id,revision,'state-aware',copy.deepcopy(scene['pending_speech']),executor)
    return result,assess(scene,result,world)
