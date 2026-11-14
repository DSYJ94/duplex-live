"""Reapprove exact changed targets; reuse only byte-identical passing records."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
from .reasoning_audio1072_semantic import body,verdict
from evals.native_control_teacher import local_key


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for n in ('parent-plan','plan','parent-review','cost-ledger','env-file','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-changed',type=int,choices=range(1,17),default=7)
    a=p.parse_args();old=json.loads(a.parent_plan.read_text());new=json.loads(a.plan.read_text());parent=json.loads((a.parent_review/'report.json').read_text());cost=json.loads(a.cost_ledger.read_text())
    if not new['owner_approved'] or new['maximum_total_API_USD']!=100 or not parent['complete1072'] or parent['source_plan_sha256']!=sha(a.parent_plan):raise ValueError('Complete real parent review and approved current source required')
    previous={r['id']:r for r in old['exact_semantic_records']};current={r['id']:r for r in new['exact_semantic_records']};judged={r['id']:r for r in parent['rows']}
    if len(current)!=1072 or set(previous)!=set(current) or set(judged)!=set(current):raise ValueError('Every original scene retained')
    failed={k for k,v in judged.items() if not v['correct'] or not v['explanation_consistent']};changed={k for k in current if current[k]!=previous[k]}
    if failed!=changed or len(changed)!=a.expected_changed:raise ValueError('Only the exact diagnosed records may change')
    if any(current[k]['fold']!=previous[k]['fold'] or current[k]['question']!=previous[k]['question'] for k in current):raise ValueError('Question/fold changed')
    payload=body([current[k] for k in sorted(changed)]);payload['max_output_tokens']=4096
    reservation=(len(json.dumps(payload).encode())+4096)*10/1e6+4096*50/1e6
    if cost['combined_usage_upper_USD']+reservation>100:raise ValueError('Carried actual/unknown costs exceed100USD')
    a.out.mkdir(parents=True,exist_ok=False)
    def write(n,v):
        with (a.out/n).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
    write('admission.json',dict(changed_ids=sorted(changed),reused_records=1072-len(changed),carried_usage_upper_USD=cost['combined_usage_upper_USD'],reservation_USD=reservation,combined_reservation_USD=cost['combined_usage_upper_USD']+reservation,maximum_calls=1,cap_USD=100,automatic_retries=0,parent_review_sha256=sha(a.parent_review/'report.json')))
    write('request.json',payload);write('inflight.json',dict(reserved=True))
    key=local_key(a.env_file);req=urllib.request.Request('https://api.openai.com/v1/responses',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=600) as response:result=json.load(response)
    write('response.json',result);rows=[v for k,v in judged.items() if k not in changed]+verdict(result,changed)
    report=dict(source_plan_sha256=sha(a.plan),complete1072=len(rows)==1072,reviewed=len(rows),all_passed=all(r['correct'] and r['explanation_consistent'] for r in rows),rows=rows,reused_verified_unchanged_records=1072-len(changed),new_reviewed_records=len(changed),parent_review_sha256=sha(a.parent_review/'report.json'),native_training_approved=False)
    write('report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
