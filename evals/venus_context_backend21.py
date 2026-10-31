"""One authorized complete21 synthetic backend replay; no GPU or training.

Exclusive output directory and durable pre-request reservations prevent automatic
retries after ambiguous network outcomes. Provider credentials stay on the Mac.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import wave
from portable.venus.context_backend_plan import payload,plan,location_payload,location_plan,MODEL,INPUT_TOKEN_CEILING,OUTPUT_TOKEN_CEILING

AUTH=Path('docs/NATIVE-VENUS-CONTEXT-BACKEND21-AUTHORIZATION-2026-10-31.json')
PRICES=Path('runs/venus-training-20261031/context-backend-price-admission-v1.json')
API='https://api.openai.com/v1/'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
    tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)


def preflight():
    auth=read(AUTH);p=read(auth['proposal_path']);prices=read(PRICES)
    assert auth['status']=='APPROVED' and auth['human_approval']=='Approve complete21 within $5/$70'
    assert auth['proposal_sha256']==sha(auth['proposal_path']) and auth['maximum_api_usd']==p['maximum_api_usd']==5
    assert len(p['requests'])==21 and {x['id'] for x in p['requests']}==set(range(21))
    assert p['maximum_external_calls']==42 and p['maximum_gpu_seconds']==p['maximum_new_optimizer_updates']==0
    assert time.time()-datetime.fromisoformat(prices['recorded_utc']).timestamp()<86400
    upper=21*(INPUT_TOKEN_CEILING*25+OUTPUT_TOKEN_CEILING*50)/1e6+21*.006
    assert upper==prices['maximum_total_usd']<5 and 28000*4/3600+24+5<=70
    for row in p['requests']:
        path=Path(row['audio_file']);assert sha(path)==row['audio_sha256']
        with wave.open(str(path)) as w:
            assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(16000,1,2)
            assert w.getnframes()/16000==row['seconds']<=30
            assert hashlib.sha256(w.readframes(w.getnframes())).hexdigest()==row['pcm_sha256']
        payload(row['native_selected_task'],'x'*2048,row['causal_audio'])
    return auth,p,prices


def preflight_v2():
    path=Path('docs/NATIVE-VENUS-CONTEXT-BACKEND21-V2-AUTHORIZATION-2026-10-31.json')
    auth=read(path)  # No file means no authorization, key read or provider call.
    assert auth['status']=='APPROVED' and auth['human_approval']=='Approve corrected21 within same $5'
    p=read(auth['proposal_path']);assert sha(auth['proposal_path'])==auth['proposal_sha256']
    assert p['maximum_additional_reasoning_calls']==21 and p['maximum_additional_transcription_calls']==0
    assert p['maximum_total_post_attempts_including_v1']==63 and p['unchanged_total_api_usd_cap']==5
    first=read('runs/venus-training-20261031/context-backend21-v1/report.json')
    assert first['actual_post_attempts']==42 and not first['cost_unknown_if_failed_attempt']
    original=read(read(AUTH)['proposal_path']);prices=read(PRICES)
    assert time.time()-datetime.fromisoformat(prices['recorded_utc']).timestamp()<86400
    reservation=21*(INPUT_TOKEN_CEILING*25+OUTPUT_TOKEN_CEILING*50)/1e6
    assert first['conservative_usage_cost_upper_usd']+reservation<=5
    rows=[]
    for row in p['requests']:
        old=original['requests'][row['id']];f=Path(row['request_file'])
        transcript=Path('runs/venus-training-20261031/context-backend21-v1')/f"transcription-{row['id']:02d}.json"
        assert sha(f)==row['sha256'] and sha(transcript)==row['original_transcription_sha256']
        assert read(f)==location_payload(old['native_selected_task'],read(transcript)['text'],old['causal_audio'])
        rows.append(dict(old,prepared_request_file=str(f)))
    assert len(rows)==21 and {x['id'] for x in rows}==set(range(21))
    return auth,dict(p,requests=rows),dict(prices,maximum_total_usd=reservation)


def main(a):
    import requests
    from evals.native_control_teacher import local_key
    revision=getattr(a,'revision',1)
    auth,proposal,prices=preflight_v2() if revision==2 else preflight()
    maximum_calls=21 if revision==2 else 42
    auth_path=Path('docs/NATIVE-VENUS-CONTEXT-BACKEND21-V2-AUTHORIZATION-2026-10-31.json') if revision==2 else AUTH
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    lock=threading.RLock();stop=threading.Event();ledger=[]
    def record():save(out/'ledger.json',dict(calls=ledger,maximum_calls=maximum_calls,full_cohort_reservation_usd=prices['maximum_total_usd'],maximum_usd=5,new_gpu_jobs=0,new_optimizer_updates=0))
    record()
    key=local_key(Path(a.env_file))
    headers={'Authorization':'Bearer '+key}
    def session():
        s=requests.Session();s.trust_env=False;return s
    # Model eligibility reads contain no task payload and incur no inference.
    with session() as s:
        access=[]
        for model in ((MODEL,) if revision==2 else ('whisper-1',MODEL)):
            r=s.get(API+'models/'+model,headers=headers,timeout=(10,25),allow_redirects=False)
            access.append(dict(model=model,http_status=r.status_code))
            save(out/'model-access.json',access)
            if r.status_code!=200:raise RuntimeError('Requested model unavailable; zero paid calls submitted')
    def post(s,kind,row,**kwargs):
        with lock:
            if stop.is_set():raise RuntimeError('Execution stopped after a material provider failure')
            assert len(ledger)<maximum_calls and not any(x['kind']==kind and x['id']==row['id'] for x in ledger)
            assert revision!=2 or kind=='reasoning', 'Revision2 may not upload audio again'
            item=dict(id=row['id'],kind=kind,status='reserved_before_request',started_utc=datetime.now(timezone.utc).isoformat())
            if kind=='reasoning':
                body=kwargs['json'];item['payload_sha256']=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
                assert key not in json.dumps(body)
            else:item['audio_sha256']=row['audio_sha256']
            ledger.append(item);record()
        start=time.monotonic()
        try:
            r=s.post(API+('audio/transcriptions' if kind=='transcription' else 'responses'),headers=headers,timeout=(10,120),allow_redirects=False,**kwargs)
            raw=r.content;data=r.json()
            save(out/f"{kind}-{row['id']:02d}.json",data)
            with lock:
                item.update(http_status=r.status_code,seconds=time.monotonic()-start,response_sha256=hashlib.sha256(raw).hexdigest(),request_id=r.headers.get('x-request-id'),status='returned',usage=data.get('usage'),response_id=data.get('id'))
                if kind=='reasoning' and r.status_code==200:
                    usage=data.get('usage',{});n=usage.get('input_tokens');o=usage.get('output_tokens')
                    if (type(n) is not int or type(o) is not int or n>INPUT_TOKEN_CEILING or o>OUTPUT_TOKEN_CEILING or data.get('service_tier') not in ('default',None)):
                        stop.set();record();raise ValueError('Actual usage/tier exceeds admitted bound or is unavailable')
                    item['conservative_cost_upper_usd']=(n*25+o*50)/1e6
                    item['standard_token_estimate_usd']=(n*10+o*50)/1e6
                elif kind=='transcription':item['conservative_cost_upper_usd']=.006
                if r.status_code in (401,403,404,429):stop.set()
                record()
            if r.status_code!=200:raise RuntimeError('Provider returned non-success status; no automatic retry')
            return data
        except Exception as error:
            with lock:item.update(status='failed_or_ambiguous_no_retry',error_type=type(error).__name__);record()
            raise
    def one(row):
        result=dict(id=row['id'],scene_id=row['scene_id'],audio_sha256=row['audio_sha256'],qualified=False)
        try:
            with session() as s:
                if revision==2:
                    recognized=read(Path('runs/venus-training-20261031/context-backend21-v1')/f"transcription-{row['id']:02d}.json")
                    body=read(row['prepared_request_file']);text=recognized['text']
                else:
                    with Path(row['audio_file']).open('rb') as f:
                        recognized=post(s,'transcription',row,data=dict(model='whisper-1',language='en',response_format='json',temperature='0'),files={'file':('context.wav',f,'audio/wav')})
                    text=recognized.get('text');body=payload(row['native_selected_task'],text,row['causal_audio'])
                save(out/f"reasoning-input-{row['id']:02d}.json",body)
                response=post(s,'reasoning',row,json=body)
                assert response.get('model','').startswith(MODEL)
                result.update(status='complete',transcript=text,native_selected_task=row['native_selected_task'],plan=(location_plan(response) if revision==2 else plan(response)),response_id=response['id'],usage=response.get('usage'))
        except Exception as error:result.update(status='failed',error_type=type(error).__name__)
        save(out/f"case-{row['id']:02d}.json",result)
        print(json.dumps(dict(event='backend_case_terminal',id=row['id'],status=result['status'])),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(one,proposal['requests']))
    assert len(results)==21
    complete=sum(x['status']=='complete' for x in results)
    report=dict(revision=revision,required_cases=21,terminal_cases=21,completed_cases=complete,all21_complete=complete==21,cases=results,actual_post_attempts=len(ledger),maximum_calls=maximum_calls,full_cohort_cost_reservation_usd=prices['maximum_total_usd'],conservative_usage_cost_upper_usd=sum(x.get('conservative_cost_upper_usd',0) for x in ledger),cost_unknown_if_failed_attempt=any(x['status']!='returned' for x in ledger),actual_billed_cost_not_verified=True,authorization_sha256=sha(auth_path),proposal_sha256=auth['proposal_sha256'],qualified=False,native_voice_evaluated=False,training_use_prohibited=True,source=a.source)
    save(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--source',required=True)
    p.add_argument('--env-file',default='runs/local-migration-20261021/.private/openai-backend.env')
    p.add_argument('--revision',type=int,choices=[1,2],default=1)
    main(p.parse_args())
