"""One complete36 independent semantic review. Never retries or approves audio."""
import argparse,hashlib,json,urllib.request
from pathlib import Path

BODY=Path('docs/NATIVE-VENUS-CONTEXT-SEMANTIC-PAYLOAD-2026-11-01.json')
ADMISSION=Path('docs/NATIVE-VENUS-CONTEXT-SEMANTIC-ADMISSION-2026-11-01.json')


def admission():
    raw=BODY.read_bytes();body=json.loads(raw);a=json.loads(ADMISSION.read_text())
    if a['status']!='ADMITTED_WITHIN_EXISTING_TEACHER_QUALIFICATION_INSTRUCTION' or a['payload_sha256']!=hashlib.sha256(raw).hexdigest():raise ValueError('Exact complete36 admission required')
    if a['maximum_paid_calls']!=1 or a['maximum_usd']!=3 or body['model']!='gpt-6-astra':raise ValueError('One bounded exact-model review required')
    reservation=(len(json.dumps(body).encode())+4096)*25/1000000+8192*50/1000000
    if reservation!=a['conservative_reservation_usd'] or reservation>3 or 65.41666666666667+reservation>70:raise ValueError('Whole cohort exceeds carried cap')
    return body,a


def validate(response,body):
    calls=[c for c in response.get('output',[]) if c.get('type')=='function_call']
    if response.get('model')!=body['model'] or response.get('status')!='completed' or len(calls)!=1 or calls[0]['name']!='review_teacher_cohort' or not response.get('id'):raise ValueError('Complete identified semantic response required')
    results=json.loads(calls[0]['arguments'])
    ids=set(body['tools'][0]['parameters']['properties'])
    if not isinstance(results,dict) or set(results)!=ids:raise ValueError('All exact36 verdicts required')
    for value in results.values():
        if set(value)!={'semantic_correct','planned_causality_consistent','concern'} or type(value['semantic_correct']) is not bool or type(value['planned_causality_consistent']) is not bool or not isinstance(value['concern'],str) or len(value['concern'])>240:raise ValueError('Strict verdict schema mismatch')
    return results


def execute(output,env_file):
    body,a=admission()
    if output.exists():raise ValueError('Observe original attempt; no retry')
    from evals.native_control_teacher import local_key
    key=local_key(env_file)
    def request(path,body=None):
        headers={'Authorization':'Bearer '+key}
        if body is not None:headers['Content-Type']='application/json'
        req=urllib.request.Request('https://api.openai.com/v1/'+path,headers=headers,data=None if body is None else json.dumps(body).encode())
        with urllib.request.urlopen(req,timeout=180) as r:return json.loads(r.read())
    output.mkdir(parents=True,exist_ok=False)
    ledger=dict(payload_sha256=a['payload_sha256'],paid_attempts=0,reservation_usd=a['conservative_reservation_usd'],complete=False,automatic_retries=0)
    def save():(output/'ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    save()
    metadata=request('models/gpt-6-astra');(output/'model-access.json').write_text(json.dumps(metadata,indent=2)+'\n')
    claims=Path('runs/venus-training-20261101/context-semantic-claims');claims.mkdir(parents=True,exist_ok=True)
    with (claims/(a['payload_sha256']+'.json')).open('x') as claim:json.dump(dict(output=str(output.resolve())),claim)
    ledger['paid_attempts']=1;ledger['status']='reserved_before_post';save()
    try:
        response=request('responses',body);(output/'response.json').write_text(json.dumps(response,indent=2)+'\n');results=validate(response,body)
        usage=response.get('usage',{});cost=usage.get('input_tokens',0)*25/1000000+usage.get('output_tokens',0)*50/1000000
        report=dict(schema='dl-context-semantic-review-v1',complete36=True,all36_semantic_pass=all(x['semantic_correct'] and x['planned_causality_consistent'] for x in results.values()),results=results,response_id=response['id'],usage=usage,conservative_usage_upper_usd=cost,invoice_verified=False,audio_or_native_timing_reviewed=False,training_approved=False,actual_provider_call=True)
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');ledger.update(complete=True,status='completed',conservative_usage_upper_usd=cost);save();print(json.dumps({k:v for k,v in report.items() if k not in ('results','usage')}))
    except Exception as error:
        ledger.update(status='failed_or_unknown',error_type=type(error).__name__);save();raise RuntimeError('Semantic attempt preserved; no retry or subset approval') from None

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--env-file',type=Path,required=True);a=p.parse_args();execute(a.output,a.env_file)
