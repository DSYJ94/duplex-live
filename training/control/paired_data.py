"""Fail-closed teacher collection and replay approval for paired control scenes."""
import argparse,concurrent.futures,json,os
from datetime import datetime,timezone
from pathlib import Path
from training.control.paired_curriculum import specifications,compile_scene,assess,xml_action
from training.control.teacher_data import TeacherGenerator,semantic_review,sha,MODEL

FILES=('training/control/paired_curriculum.py','training/control/paired_data.py','training/control/curriculum.py','training/control/teacher_data.py','evals/qwen_response_executor.py','evals/qwen35_controller.py','evals/duplex_reasoning_worker.py','evals/duplex_speech_context.py','evals/duplex_resume.py')


def collect(scene,out):
    folder=out/'rows'/scene['id'];attempts=[];accepted=None
    for attempt in range(3):
        teacher=TeacherGenerator(folder,attempt);entry=dict(attempt=attempt,passed=False)
        try:
            response=compile_scene(scene,teacher);quality=assess(scene,response)
            entry.update(teacher=teacher.last,mechanical=quality)
            review=dict(passed=True,method='exact answer/identity/state or zero-argument routing checks')
            if quality['passed'] and scene['kind']=='clarify':review=semantic_review(scene,quality['result'],folder,attempt)
            entry['review']=review;entry['passed']=quality['passed'] and review['passed']
            if entry['passed']:
                accepted=dict(id=scene['id'],pair_id=scene['pair_id'],split='train',kind=scene['kind'],family=scene['family'],student_messages=teacher.messages,student_tools=teacher.tools,student_assistant_prefill=teacher.prefill,target_native_text=teacher.last['native_text'],target_native_action=teacher.last['native_action'],compiled_response=response,quality_attempt=attempt,teacher_provenance=teacher.last,source_scene_sha256=sha(folder/'scene.json'),quality_passed=True)
        except Exception as error:entry.update(error_type=type(error).__name__,message=str(error),teacher=teacher.last)
        attempts.append(entry);(folder/'quality.json').write_text(json.dumps(dict(id=scene['id'],passed=bool(accepted),attempts=attempts),indent=2)+'\n')
        if accepted:break
    return accepted,dict(id=scene['id'],passed=bool(accepted),attempts=attempts)


def generate(out,preview=False):
    out.mkdir(parents=True,exist_ok=False);scenes=specifications(preview);train=[s for s in scenes if s['split']=='train'];held=[s for s in scenes if s['split']=='heldout']
    if {s['family'] for s in train}&{s['family'] for s in held} or {s['pair_id'] for s in train}&{s['pair_id'] for s in held}:raise ValueError('A paired family crosses the split')
    (out/'scenes.json').write_text(json.dumps(scenes,indent=2)+'\n')
    for scene in train:
        p=out/'rows'/scene['id'];p.mkdir(parents=True);(p/'scene.json').write_text(json.dumps(scene,indent=2)+'\n')
    accepted=[];quality=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in concurrent.futures.as_completed([pool.submit(collect,s,out) for s in train]):
            row,q=result.result();quality.append(q)
            if row:accepted.append(row)
            print(json.dumps(dict(id=q['id'],passed=q['passed'],attempts=len(q['attempts']))),flush=True)
    quality.sort(key=lambda x:x['id']);accepted.sort(key=lambda x:x['id']);passed=len(accepted)==len(train) and all(q['passed'] for q in quality)
    (out/'quality.json').write_text(json.dumps(dict(all_passed=passed,requested=len(train),passed=len(accepted),rows=quality),indent=2)+'\n')
    # Rejected and preview batches never receive a TRAIN file.
    if passed and not preview:(out/'train.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in accepted))
    (out/'accepted.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in accepted))
    manifest=dict(schema='dl-paired-controller-data-v1',preview=preview,training_approved=passed and not preview,all_label_quality_passed=passed,train_rows=len(train),heldout_rows=len(held),teacher_model=MODEL,provenance_class='closed-content',created_utc=datetime.now(timezone.utc).isoformat(),producer_commit=os.environ.get('DL_TASK_COMMIT'),authorization_sha256=sha('decisions/0003-provenance-classes.md'),source_files={f:sha(f) for f in FILES},scenes_sha256=sha(out/'scenes.json'),quality_sha256=sha(out/'quality.json'),accepted_sha256=sha(out/'accepted.jsonl'),train_sha256=sha(out/'train.jsonl') if passed and not preview else None,prior_evaluation_read=False,captured_results_used_as_labels=False,native_voice_training=False,audio_generated=False,delegation_labels_are_selection_only=True,actual_reasoning_requests=0)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if not passed:raise ValueError('Incomplete label quality: entire batch rejected, no TRAIN file')
    verify(out,audit_preview=preview);print(json.dumps(dict(complete=True,preview=preview,training_approved=manifest['training_approved'],train_rows=len(train),heldout_rows=len(held))),flush=True)


def verify(folder,audit_preview=False):
    folder=Path(folder);m=json.loads((folder/'manifest.json').read_text())
    if m['schema']!='dl-paired-controller-data-v1' or not m['all_label_quality_passed'] or m['teacher_model']!=MODEL or m['prior_evaluation_read'] or m['captured_results_used_as_labels'] or m['audio_generated'] or m['actual_reasoning_requests']!=0:raise ValueError('Missing data provenance/quality approval')
    if m['preview']:
        if not audit_preview or m['training_approved'] or (folder/'train.jsonl').exists():raise ValueError('Preview cannot train')
    elif not m['training_approved'] or sha(folder/'train.jsonl')!=m['train_sha256'] or (folder/'train.jsonl').read_bytes()!=(folder/'accepted.jsonl').read_bytes():raise ValueError('Missing exact full TRAIN file')
    if not m['producer_commit'] or len(m['producer_commit'])!=40 or m['source_files']!={f:sha(f) for f in FILES} or sha('decisions/0003-provenance-classes.md')!=m['authorization_sha256']:raise ValueError('Changed source/compiler/authorization')
    for f,k in [('scenes.json','scenes_sha256'),('quality.json','quality_sha256'),('accepted.jsonl','accepted_sha256')]:
        if sha(folder/f)!=m[k]:raise ValueError('Changed paired corpus artifact')
    scenes=json.loads((folder/'scenes.json').read_text());assert scenes==specifications(m['preview'])
    train={s['id']:s for s in scenes if s['split']=='train'};held=[s for s in scenes if s['split']=='heldout'];rows=[json.loads(x) for x in (folder/'accepted.jsonl').read_text().splitlines()];quality=json.loads((folder/'quality.json').read_text())
    if len(rows)!=len(train) or len({r['id'] for r in rows})!=len(rows) or {r['id'] for r in rows}!=set(train) or not quality['all_passed'] or len(quality['rows'])!=len(train):raise ValueError('Incomplete corpus inventory')
    q_by_id={q['id']:q for q in quality['rows']}
    if set(q_by_id)!=set(train) or len(q_by_id)!=len(quality['rows']):raise ValueError('Invalid quality inventory')
    for row in rows:
        scene=train[row['id']];p=folder/'rows'/row['id'];q=q_by_id[row['id']]
        if json.loads((p/'scene.json').read_text())!=scene or sha(p/'scene.json')!=row['source_scene_sha256'] or json.loads((p/'quality.json').read_text())!=q:raise ValueError('Changed paired scene/quality')
        entry=q['attempts'][row['quality_attempt']];provenance=row['teacher_provenance'];raw=p/(f"attempt-{row['quality_attempt']}-teacher.json")
        if not q['passed'] or not entry['passed'] or entry['teacher']!=provenance or sha(raw)!=provenance['response_sha256']:raise ValueError('Missing actual teacher response')
        response=json.loads(raw.read_text());calls=[x for x in response.get('output',[]) if x['type']=='function_call']
        if response['status']!='completed' or response['id']!=provenance['teacher_response_id'] or response['model']!=provenance['actual_model'] or len(calls)!=1 or not provenance['source_date_utc']:raise ValueError('Teacher provenance changed')
        action=dict(name=calls[0]['name'],arguments=json.loads(calls[0]['arguments']))
        if action!=row['target_native_action'] or xml_action(action)!=row['target_native_text']:raise ValueError('Rewritten teacher action')
        class Recorded:
            model_id=MODEL
            def request(self,messages,tools,assistant_prefill=''):
                if messages!=row['student_messages'] or tools!=row['student_tools'] or assistant_prefill!=row['student_assistant_prefill']:raise ValueError('Changed student prompt/tools/prefix')
                return dict(raw_text=row['target_native_text'],cleaned_text=row['target_native_text'],assistant_prefill=assistant_prefill,input_tokens=0,generated_tokens=0)
        replay=compile_scene(scene,Recorded());checked=assess(scene,replay)
        if not checked['passed'] or checked!=entry['mechanical']:raise ValueError('Actual worker action/state replay failed')
        if scene['kind']=='clarify':
            review=entry['review'];rp=p/(f"attempt-{row['quality_attempt']}-review.json");v=json.loads(rp.read_text());votes=[x for x in v['output'] if x['type']=='function_call']
            if not review['passed'] or sha(rp)!=review['response_sha256'] or v['id']!=review['response_id'] or v['status']!='completed' or len(votes)!=1 or votes[0]['name']!='judge_control_utterance' or json.loads(votes[0]['arguments'])!=dict(verdict='pass',reason=review['reason']):raise ValueError('Clarification semantic review missing')
    return rows,held,m


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--preview',action='store_true');a=p.parse_args();generate(a.out,a.preview)
if __name__=='__main__':main()
