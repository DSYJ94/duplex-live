"""Recollect complete TRAIN supervision with matched capabilities and state actions."""
import argparse
import concurrent.futures
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
from training.control.state_curriculum import specifications,run,assess
from training.control.curriculum import xml_action
from training.control.teacher_data import request,sha,MODEL

FILES=('training/control/state_curriculum.py', 'training/control/state_data.py', 'training/control/paired_curriculum.py', 'training/control/capability_curriculum.py', 'training/control/curriculum.py', 'training/control/teacher_data.py', 'evals/application_capabilities.py', 'evals/idempotent_reasoning_worker.py', 'evals/qwen_response_executor.py', 'evals/qwen35_controller.py', 'evals/qwen_controller_probe.py', 'evals/duplex_reasoning_worker.py', 'evals/recovery_worker.py', 'evals/duplex_speech_context.py', 'evals/duplex_resume.py')


def api_history(messages):
    """Translate native student chat history to stateless Responses tool items."""
    result=[]
    for message in copy.deepcopy(messages):
        role=message['role']
        if role=='tool':result.append(dict(type='function_call_output',call_id=message['tool_call_id'],output=message['content']))
        elif role=='assistant' and message.get('tool_calls'):
            if message.get('content'):result.append(dict(role='assistant',content=message['content']))
            for call in message['tool_calls']:
                function=call['function'];arguments=function['arguments']
                if not isinstance(arguments,dict):raise ValueError('Expected native object-valued function history')
                result.append(dict(type='function_call',call_id=call['id'],name=function['name'],arguments=json.dumps(arguments)))
        elif role in ('system','user','assistant') and isinstance(message.get('content'),str):result.append(dict(role=role,content=message['content']))
        else:raise ValueError('Unrecognized native teacher history')
    return result


def teacher_input(messages,attempt,domain):
    if domain not in ('reasoning','records'):raise ValueError('Unknown actual application profile')
    result=api_history(messages)
    result[0]['content']+='\nTEACHER-ONLY LABEL QUALITY CONVENTION: Fulfil the complete latest request using the actual capabilities and current native tools shown in the SAME student prompt. For result-delivery cancellation combined with unchanged continuation, select the idempotent cancel_reasoning_and_resume_speech operation and identify all matching result IDs, even for a truthful already-finished or already-cancelled no-op. Plain resume_speech is for continuation without a result-cancellation request. Never cancel unrelated work. If cancellation is combined with a NEW question or replacement answer, speak that new answer through the speaking cancellation action and retire only superseded speech; cancellation confirmation alone does not answer the new question. Never claim an unavailable private read or messaging action ran. Asking for credentials or for the requested private value cannot enable an unsupported operation. A supported reader awaiting project identification needs that project, not a denial of the reader. Honor requested answer form and actual receipt values.'
    if attempt:result[0]['content']+='\nTeacher-only repair: Re-read every clause, actual task status, actual supported operations and speech obligations. Return a valid complete native action; do not omit a requested new answer or fabricate a cancellation.'
    return result


class Teacher:
    model_id=MODEL
    def __init__(self,folder,attempt,domain):self.folder=folder;self.attempt=attempt;self.domain=domain;self.calls=[]
    def request(self,messages,tools,assistant_prefill=''):
        if assistant_prefill!='<tool_call>\n':raise ValueError('Changed native prefix')
        index=len(self.calls);path=self.folder/f'attempt-{self.attempt}-stage-{index}-teacher.json'
        sent=teacher_input(messages,self.attempt,self.domain);api_tools=[dict(type='function',**t['function'],strict=True) for t in tools]
        data,call,seconds=request(sent,api_tools,path);action=dict(name=call['name'],arguments=json.loads(call['arguments']));native=xml_action(action)
        row=dict(stage=index,student_messages=copy.deepcopy(messages),student_tools=copy.deepcopy(tools),student_assistant_prefill=assistant_prefill,target_native_text=native,target_native_action=action,teacher_provenance=dict(actual_model=data['model'],source_model=MODEL,source_date_utc=datetime.now(timezone.utc).isoformat(),response_id=data['id'],response_file=path.name,response_sha256=sha(path),api_input_sha256=__import__('hashlib').sha256(json.dumps(sent,sort_keys=True).encode()).hexdigest(),usage=data.get('usage'),seconds=seconds,provenance_class='closed-content',teacher_repair_note_present=bool(self.attempt),teacher_only_label_policy=True,actual_application_profile=self.domain))
        self.calls.append(row)
        return dict(raw_text=native,cleaned_text=native,assistant_prefill=assistant_prefill,input_tokens=0,generated_tokens=0,qwen_neural_inference_executed=False)


def review_input(scene,result):
    return [dict(role='system',content='Independently judge a synthetic TRAIN label against its declared application, task states and actual worker receipts. No owner or real private data is involved. Require completion of the WHOLE latest request. A cancellation confirmation alone cannot answer an additional new question. Never approve invented external execution, unavailable private access, promises of connectors that do not exist, or cancellation of unrelated work. Already-finished/idempotent status receipts do not newly cancel work. An ordinary cancellation confirmation must not claim remote computation was aborted. A supported record reader must ask for missing project identification rather than deny its existence. Judge only; never rewrite the label.'),dict(role='user',content=json.dumps(dict(scene=scene,actual_result={k:result[k] for k in ('route','content','consume_pending_speech_ids','receipts')})))]


def needs_review(scene):
    return scene['kind'] in ('unsupported-read','unsupported-send','finished-cancel','record-clarify','cancel-new-answer','finished-confirmation','clarify-target','cancel-preserve','cancel-one','clarify')


def review(scene,result,folder,attempt):
    schema=dict(type='function',name='judge_state_label',description='Judge the complete actual capability and speech contract.',strict=True,parameters=dict(type='object',properties=dict(verdict=dict(type='string',enum=['pass','fail']),reason=dict(type='string')),required=['verdict','reason'],additionalProperties=False))
    path=folder/f'attempt-{attempt}-review.json';data,call,seconds=request(review_input(scene,result),[schema],path)
    args=json.loads(call['arguments'])
    if call['name']!='judge_state_label' or set(args)!={'verdict','reason'} or args['verdict']not in ('pass','fail'):raise ValueError('Invalid semantic review')
    return dict(passed=args['verdict']=='pass',reason=args['reason'],response_id=data['id'],actual_model=data['model'],source_date_utc=datetime.now(timezone.utc).isoformat(),response_file=path.name,response_sha256=sha(path),seconds=seconds,api_input_sha256=__import__('hashlib').sha256(json.dumps(review_input(scene,result),sort_keys=True).encode()).hexdigest(),independent_request_same_model=True)


def collect(scene,out):
    folder=out/'rows'/scene['id'];attempts=[];accepted=None
    for attempt in range(3):
        teacher=Teacher(folder,attempt,scene['domain']);entry=dict(attempt=attempt,passed=False)
        try:
            result,world=run(scene,teacher,MODEL);quality=assess(scene,result,world)
            entry.update(mechanical=quality,result=result,teacher_calls=teacher.calls)
            checked=dict(passed=True,method='exact fixture state/value/silent same-generator action')
            if quality['passed'] and needs_review(scene):checked=review(scene,result,folder,attempt)
            entry['semantic_review']=checked;entry['passed']=quality['passed'] and checked['passed']
            if entry['passed']:
                accepted=[dict(row,id=scene['id']+'-stage-'+str(row['stage']),scene_id=scene['id'],split='train',kind=scene['kind'],family=scene['family'],domain=scene['domain'],source_scene_sha256=sha(folder/'scene.json'),quality_attempt=attempt,quality_passed=True,train_corpus=scene['corpus']) for row in teacher.calls]
                if len(accepted)!=(2 if scene['kind']=='record-read' else 1):raise ValueError('Incomplete actual worker stage inventory')
        except Exception as error:entry.update(error_type=type(error).__name__,message=str(error),teacher_calls=teacher.calls);accepted=None
        attempts.append(entry);(folder/'quality.json').write_text(json.dumps(dict(id=scene['id'],passed=bool(accepted),attempts=attempts),indent=2)+'\n')
        if accepted:break
    return accepted,dict(id=scene['id'],passed=bool(accepted),attempts=attempts)


def generate(out,preview):
    out.mkdir(parents=True,exist_ok=False);scenes=specifications(preview);train=[s for s in scenes if s['split']=='train'];held=[s for s in scenes if s['split']=='heldout']
    (out/'scenes.json').write_text(json.dumps(scenes,indent=2)+'\n')
    for scene in train:
        p=out/'rows'/scene['id'];p.mkdir(parents=True);(p/'scene.json').write_text(json.dumps(scene,indent=2)+'\n')
    rows=[];checks=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        tasks=[pool.submit(collect,scene,out) for scene in train]
        for future in concurrent.futures.as_completed(tasks):
            accepted,quality=future.result();checks.append(quality)
            if accepted:rows.extend(accepted)
            print(json.dumps(dict(id=quality['id'],passed=quality['passed'])),flush=True)
    rows.sort(key=lambda r:r['id']);checks.sort(key=lambda r:r['id']);expected=sum(2 if s['kind']=='record-read' else 1 for s in train)
    passed=len(rows)==expected and len(checks)==len(train) and all(x['passed'] for x in checks)
    (out/'quality.json').write_text(json.dumps(dict(all_passed=passed,requested=len(train),passed=sum(x['passed'] for x in checks),rows=checks,expected_label_stages=expected),indent=2)+'\n')
    if passed:(out/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest=dict(schema='dl-state-teacher-data-v1',preview=preview,training_approved=passed and not preview,all_quality_passed=passed,teacher_model=MODEL,provenance_class='closed-content',created_utc=datetime.now(timezone.utc).isoformat(),producer_commit=os.environ.get('DL_TASK_COMMIT'),authorization_sha256=sha('decisions/0003-provenance-classes.md'),synthetic_text_only=True,audio_generated=False,owner_data_used=False,prior_evaluation_read=False,training_scenes=len(train),heldout_scenes=len(held),training_rows=len(rows),expected_training_rows=expected,source_files={f:sha(f) for f in FILES},scenes_sha256=sha(out/'scenes.json'),quality_sha256=sha(out/'quality.json'),train_sha256=sha(out/'train.jsonl') if passed else None)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if not passed:raise ValueError('Incomplete teacher quality; no TRAIN approval')
    verify(out,audit_preview=preview);print(json.dumps(dict(rows=len(rows),preview=preview,training_approved=not preview)))


def verify(folder,*,audit_preview=False):
    folder=Path(folder);m=json.loads((folder/'manifest.json').read_text())
    if (m['schema']!='dl-state-teacher-data-v1' or not m['all_quality_passed'] or m['teacher_model']!=MODEL or m['provenance_class']!='closed-content' or m['prior_evaluation_read'] or m['owner_data_used'] or m['audio_generated'] or not m['synthetic_text_only']):raise ValueError('Missing full quality/provenance')
    if m['preview']:
        if not audit_preview or m['training_approved']:raise ValueError('Preview is not TRAIN approved')
    elif not m['training_approved']:raise ValueError('No TRAIN approval')
    if not m['producer_commit'] or len(m['producer_commit'])!=40 or m['source_files']!={f:sha(f) for f in FILES} or m['authorization_sha256']!=sha('decisions/0003-provenance-classes.md'):raise ValueError('Compiler/authorization lineage differs')
    for name,key in [('scenes.json','scenes_sha256'),('quality.json','quality_sha256'),('train.jsonl','train_sha256')]:
        if sha(folder/name)!=m[key]:raise ValueError('Changed approved artifact')
    scenes=json.loads((folder/'scenes.json').read_text());quality=json.loads((folder/'quality.json').read_text());rows=[json.loads(x) for x in (folder/'train.jsonl').read_text().splitlines()]
    if scenes!=specifications(m['preview']) or not quality['all_passed'] or len(rows)!=m['expected_training_rows'] or len({r['id'] for r in rows})!=len(rows):raise ValueError('Changed/incomplete label inventory')
    train=[s for s in scenes if s['split']=='train'];held=[s for s in scenes if s['split']=='heldout']
    if len(train)!=m['training_scenes'] or len(held)!=m['heldout_scenes'] or {r['scene_id'] for r in rows}!={s['id'] for s in train} or {r['id'] for r in quality['rows']}!={s['id'] for s in train}:raise ValueError('Lost source scenes')
    for scene in train:
        src=folder/'rows'/scene['id'];q=json.loads((src/'quality.json').read_text())
        if json.loads((src/'scene.json').read_text())!=scene or q!=next(x for x in quality['rows'] if x['id']==scene['id']) or not q['passed']:raise ValueError('Changed source quality')
        selected=sorted([r for r in rows if r['scene_id']==scene['id']],key=lambda r:r['stage'])
        if len(selected)!=(2 if scene['kind']=='record-read' else 1):raise ValueError('Missing worker stage')
        class Replay:
            def __init__(self):self.index=0
            def request(self,messages,tools,assistant_prefill=''):
                row=selected[self.index];self.index+=1;p=row['teacher_provenance'];attempt=row['quality_attempt']
                if (row['student_messages']!=messages or row['student_tools']!=tools or row['student_assistant_prefill']!=assistant_prefill or row['source_scene_sha256']!=sha(src/'scene.json') or row['split']!='train' or row['kind']!=scene['kind'] or row['family']!=scene['family']):raise ValueError('Changed exact student prompt/scene')
                path=src/p['response_file'];raw=json.loads(path.read_text());calls=[c for c in raw['output'] if c['type']=='function_call']
                if (raw['status']!='completed' or raw['model']!=p['actual_model'] or raw['id']!=p['response_id'] or sha(path)!=p['response_sha256'] or not p['source_date_utc'] or len(calls)!=1 or p['provenance_class']!='closed-content' or p['actual_application_profile']!=scene['domain'] or not p['teacher_only_label_policy']):raise ValueError('Missing actual teacher response lineage')
                action=dict(name=calls[0]['name'],arguments=json.loads(calls[0]['arguments']))
                if action!=row['target_native_action'] or xml_action(action)!=row['target_native_text'] or __import__('hashlib').sha256(json.dumps(teacher_input(messages,attempt,scene['domain']),sort_keys=True).encode()).hexdigest()!=p['api_input_sha256']:raise ValueError('Changed native target/API input')
                return dict(raw_text=row['target_native_text'],cleaned_text=row['target_native_text'],assistant_prefill=assistant_prefill,input_tokens=0,generated_tokens=0,qwen_neural_inference_executed=False)
        replay=Replay();result,world=run(scene,replay,MODEL)
        if not assess(scene,result,world)['passed'] or replay.index!=len(selected):raise ValueError('Actual worker replay failed')
        entry=q['attempts'][selected[0]['quality_attempt']]
        if not entry['passed'] or {k:v for k,v in entry['result'].items() if k!='events'}!={k:v for k,v in result.items() if k!='events'} or entry['teacher_calls']!=[{k:v for k,v in row.items() if k in ('stage','student_messages','student_tools','student_assistant_prefill','target_native_text','target_native_action','teacher_provenance')} for row in selected]:raise ValueError('Changed accepted attempt')
        review=entry['semantic_review']
        if not review['passed']:raise ValueError('Failed semantic approval')
        if 'response_file' in review:
            path=src/review['response_file'];raw=json.loads(path.read_text());calls=[x for x in raw['output'] if x['type']=='function_call']
            if review['api_input_sha256']!=__import__('hashlib').sha256(json.dumps(review_input(scene,result),sort_keys=True).encode()).hexdigest() or sha(path)!=review['response_sha256'] or raw['id']!=review['response_id'] or raw['model']!=review['actual_model'] or len(calls)!=1 or calls[0]['name']!='judge_state_label' or json.loads(calls[0]['arguments'])!={'verdict':'pass','reason':review['reason']}:raise ValueError('Changed actual semantic review')
    return rows,held,m


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--preview',action='store_true');a=p.parse_args();generate(a.out,a.preview)


if __name__=='__main__':main()
