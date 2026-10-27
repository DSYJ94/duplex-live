"""Bounded matched accent sources using only verified, already available voices."""
import argparse,array,hashlib,json,math,os,urllib.request,urllib.error,wave,time
from pathlib import Path
from datetime import datetime,timezone
MODEL='eleven_v3';RATE=.0005;PRIOR_API_UPPER=5.6301125;ADDITIONAL_CAP=3.5
MATCHED=['voice-v1-warm_conversation-01','voice-v1-explanation_and_emphasis-01','voice-v1-calm_pacing-05','voice-v1-numbers_and_pronunciation-05','voice-v1-curious_questions-06','voice-v1-allowed_backchannel-06']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(a):
    base=Path('runs/venus-training-20261027');plan_path=Path(a.source_plan);plan=json.loads(plan_path.read_text());source={r['id']:r for r in plan['cases']}
    access={}
    for name in ('eleven-british-voice-access-v1.json','eleven-existing-regional-access-v1.json','eleven-glasgow-access-v1.json'):
        for v in json.loads((base/name).read_text())['voices']:access[v['voice_id']]=v
    chosen=[('british-alice','Xb7hH8MSUJpSbSDYk0k2','British English · Alice',True),('british-george','JBFqnCBsd6RMkjVDRZzb','British English · George',False),('scottish-edinburgh','eh9GHWIaYsZB3Alk8wrY','Scottish · Edinburgh',False),('scottish-glasgow','iePvrB4HtMcAormXZou7','Scottish · Glasgow',False),('irish-rory','hmMWXCj9K7N5mCPcRkfC','Irish English · Rory',False)]
    rows=[];profiles=[]
    for profile,ident,label,full in chosen:
        v=access[ident];assert v['accessible'] and v['category'] in ('premade','professional')
        shared=v.get('sharing') or {};assert shared.get('rate',1)==1 and (not shared.get('disable_at_unix') or shared['disable_at_unix']>time.time()+86400)
        profiles.append(dict(id=profile,voice_id=ident,label=label,category=v['category'],labels=v['labels'],existing_account_voice=True,full_curriculum=full,access_evidence=v))
        cases=[c for c in plan['cases'] if c['assistant_text']] if full else [source[i] for i in MATCHED]
        cases=cases+[dict(id='voice-v1-reference',split='REFERENCE_ONLY',family='neutral_reference',user_context='Voice-conditioning reference',assistant_text=plan['reference_text'])]
        for c in cases:
            body=dict(text=c['assistant_text'],model_id=MODEL,language_code='en',seed=20261030,voice_settings=dict(stability=.5,similarity_boost=.75))
            rows.append(dict(id=profile+'--'+c['id'],profile=profile,voice_id=ident,source_case_id=c['id'],split=c['split'],family=c['family'],context=c.get('user_context',''),payload=body,request_sha256=hashlib.sha256(json.dumps(dict(voice_id=ident,payload=body),sort_keys=True).encode()).hexdigest()))
    chars=sum(len(r['payload']['text']) for r in rows);upper=chars*RATE
    assert len(rows)==95 and chars==6488 and upper<=ADDITIONAL_CAP and PRIOR_API_UPPER+upper<10
    result=dict(schema='dl-british-accent-source-v1',status='AUTHORIZED_SYNTHETIC_SOURCES_UNQUALIFIED',source_plan_sha256=sha(plan_path),profiles=profiles,requests=rows,matched_case_ids=MATCHED,primary_british_profile='british-alice',requests_required=95,characters=chars,maximum_additional_usage_usd=upper,additional_cap_usd=ADDITIONAL_CAP,prior_external_usage_upper_usd=PRIOR_API_UPPER,maximum_combined_external_usage_usd=PRIOR_API_UPPER+upper,combined_cap_usd=10,authority='Focused40USDgrant/API10USD and owner request for moreBritish/nonAmericanaccents; existingkey explicitlyapproved',owner_conversations=False,new_voice_cloning=False,account_voice_collection_modified=False,words_and_folds_unchanged=True,one_voice_per_conversation=True,model_accent_transfer_qualified=False,training_approved=False,silence_targets=[r for r in plan['cases'] if not r['assistant_text']])
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('requests_required','characters','maximum_additional_usage_usd','maximum_combined_external_usage_usd')}))

def execute(a):
    path=Path(a.prepared);p=json.loads(path.read_text());expected=p['requests_required'];assert len(p['requests'])==expected and expected in (12,95) and p['maximum_additional_usage_usd']<=ADDITIONAL_CAP and p['maximum_combined_external_usage_usd']<10
    key=os.environ.get('ELEVENLABS_API_KEY');assert key;out=Path(a.output);out.mkdir(exist_ok=False)
    def request(endpoint,body=None):
        req=urllib.request.Request('https://api.elevenlabs.io/v1/'+endpoint,headers={'xi-api-key':key,**({'Content-Type':'application/json'} if body else {})},data=json.dumps(body).encode() if body else None)
        return urllib.request.urlopen(req,timeout=90)
    with request('user/subscription') as response:account=json.load(response)
    remaining=account['character_limit']-account['character_count'];assert remaining>=p['characters']
    (out/'account-preflight.json').write_text(json.dumps(dict(at_utc=datetime.now(timezone.utc).isoformat(),remaining_characters=remaining,required_characters=p['characters'],no_purchase=True))+'\n')
    receipts=[]
    for r in p['requests']:
        folder=out/r['id'];folder.mkdir();(folder/'attempt.json').write_text(json.dumps(dict(r,outcome='SUBMITTED_PENDING_NO_RETRY',at_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
        try:
            with request('text-to-speech/'+r['voice_id']+'?output_format=pcm_24000',r['payload']) as response:raw=response.read();request_id=response.headers.get('request-id')
        except urllib.error.HTTPError as e:raise RuntimeError('Accent synthesis HTTP'+str(e.code)+'; no retry') from None
        assert raw and len(raw)%2==0;(folder/'source.pcm').write_bytes(raw)
        with wave.open(str(folder/'source.wav'),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(raw)
        values=array.array('h',raw);peak=max(abs(x) for x in values)/32768
        row=dict(id=r['id'],profile=r['profile'],voice_id=r['voice_id'],source_case_id=r['source_case_id'],split=r['split'],family=r['family'],text=r['payload']['text'],context=r['context'],provider='ElevenLabs',model_id=MODEL,request_id=request_id,request_sha256=r['request_sha256'],raw_pcm_sha256=hashlib.sha256(raw).hexdigest(),wav_sha256=sha(folder/'source.wav'),sample_rate=24000,seconds=len(values)/24000,peak=peak,peak_pass=peak<=.995,provenance_class='closed-content',provenance_authority='ADR0003 Amendment A',training_approved=False)
        (folder/'receipt.json').write_text(json.dumps(row,indent=2)+'\n');receipts.append(row)
        result=dict(complete=len(receipts)==expected,required=expected,utterances=receipts,prepared_sha256=sha(path),maximum_usage_usd=p['maximum_additional_usage_usd'],all_peak_checks=all(x['peak_pass'] for x in receipts),independent_words='PENDING_GPU_AUDIT',naturalness='PENDING_LISTENING',training_approved=False,subset_acceptance=False)
        (out/'render-receipts.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(event='accent_source_rendered',completed=len(receipts),required=expected,profile=r['profile'],id=r['id'],peak_pass=peak<=.995)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','execute']);p.add_argument('--source-plan');p.add_argument('--prepared');p.add_argument('--output',required=True);a=p.parse_args();dict(prepare=prepare,execute=execute)[a.stage](a)
