"""Requalify all changed literal-wording records; preserve exact passing parents."""
import argparse,concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
from .reasoning_audio1072_semantic import body,verdict
from evals.native_control_teacher import local_key


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for n in ('parent-plan','plan','parent-review','cost-ledger','env-file','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();old=json.loads(a.parent_plan.read_text());new=json.loads(a.plan.read_text());parent=json.loads((a.parent_review/'report.json').read_text());cost=json.loads(a.cost_ledger.read_text())
    if not new['owner_approved'] or new['maximum_total_API_USD']!=100 or not parent['complete1072'] or not parent['all_passed'] or parent['source_plan_sha256']!=sha(a.parent_plan):raise ValueError('Actual complete passing parent and approved new source required')
    old_rows={r['id']:r for r in old['exact_semantic_records']};current={r['id']:r for r in new['exact_semantic_records']};judged={r['id']:r for r in parent['rows']}
    if len(current)!=1072 or set(current)!=set(old_rows) or set(current)!=set(judged):raise ValueError('Full original1072 required')
    if any(current[k]['fold']!=old_rows[k]['fold'] for k in current):raise ValueError('Folds changed')
    changed={k for k in current if current[k]!=old_rows[k]};reused=[v for k,v in judged.items() if k not in changed]
    if any(not r['correct'] or not r['explanation_consistent'] for r in reused):raise ValueError('A failed parent cannot be reused')
    records=[current[k] for k in sorted(changed)];bodies=[body(records[i:i+8])for i in range(0,len(records),8)]
    for b in bodies:b['max_output_tokens']=4096
    reserve=sum((len(json.dumps(b).encode())+4096)*10/1e6+4096*50/1e6 for b in bodies)
    if cost['combined_usage_upper_USD']+reserve>100:raise ValueError('Carried unknowns plus complete changed set exceed100USD')
    a.out.mkdir(parents=True,exist_ok=False)
    def write(n,v):
        with (a.out/n).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
    write('admission.json',dict(changed_records=len(changed),reused_records=len(reused),maximum_calls=len(bodies),reservation_USD=reserve,carried_usage_upper_USD=cost['combined_usage_upper_USD'],combined_reservation_USD=cost['combined_usage_upper_USD']+reserve,parent_review_sha256=sha(a.parent_review/'report.json'),automatic_retries=0,cap_USD=100))
    key=local_key(a.env_file)
    def one(pair):
        i,b=pair;write('%03d-request.json'%i,b);write('%03d-inflight.json'%i,dict(reserved=True))
        req=urllib.request.Request('https://api.openai.com/v1/responses',data=json.dumps(b).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=600) as response:result=json.load(response)
            write('%03d-response.json'%i,result);rows=verdict(result,[x['id']for x in json.loads(b['input'])]);write('%03d-verdict.json'%i,rows);print(json.dumps(dict(batch=i,reviewed=len(rows))),flush=True);return rows
        except Exception as exc:write('%03d-error.json'%i,dict(error_type=type(exc).__name__,retry=False));return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(one,enumerate(bodies)))
    rows=reused+[r for batch in results if batch for r in batch];complete=all(x is not None for x in results) and len(rows)==1072
    write('report.json',dict(source_plan_sha256=sha(a.plan),complete1072=complete,reviewed=len(rows),all_passed=complete and all(x['correct'] and x['explanation_consistent']for x in rows),rows=rows,reused_verified_unchanged_records=len(reused),new_reviewed_records=len(records),parent_review_sha256=sha(a.parent_review/'report.json'),native_training_approved=False))
    if not complete:raise RuntimeError('Incomplete review preserved; no subset acceptance or retry')


if __name__=='__main__':main()
