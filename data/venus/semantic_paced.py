"""Independent complete292 teacher review inside the approved10USD API reserve."""
import argparse,concurrent.futures,hashlib,json,shlex,urllib.request,urllib.error
from pathlib import Path
from .semantic import MODEL,INSTRUCTIONS,SCHEMA
from .pacing import words
NATIVE_PROTOCOL='Native protocol: a speak action at frame f emits one-second OUTPUT units over [f+offset,f+offset+1). final=True places TURN_EOS AFTER that unit text; it does not mute the text or close at unit onset. Released source generates TTS for that unit, then flushes token2wav before reset. Independently aligned source word ends determine the final unit, with empty continuation when the final word spans another unit. Do not invent an extra silence interval after every final spoken word. Evaluate teacher wording, source-bound pacing and causal task timing; actual acoustic completion still requires independent native audio evaluation.'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(a):
    root=Path(a.timelines);audit=json.loads(Path(a.audit).read_text());quality=json.loads(Path(a.word_quality).read_text());manifest=json.loads((root/'manifest.json').read_text())
    assert audit['passed'] and audit['scenes']==292 and audit['unique_audio']==218 and audit['manifest_sha256']==sha(root/'manifest.json')
    assert quality['all_passed'] and quality['requested']==quality['passed']==218 and quality['audio_receipts_sha256']==audit['audio_receipts_sha256']
    recognized={r['source_sha256']:r for r in quality['rows']};rows=[]
    for record in manifest['scenes']:
        t=json.loads((root/record['id']/'timeline.json').read_text());assert sha(root/record['id']/'timeline.json')==record['timeline_sha256']
        users=[]
        for u in t['user_turns']:
            r=recognized[u['sha256']];assert r['passed'] and r['authored']==u['text']
            item=dict(text=u['text'],start=u['start_seconds'],end=u['start_seconds']+u['seconds'])
            if words(r['recognized'])==words(u['text']):item['independent_asr_exact_orthographic_words']=True
            else:item['recognized']=r['recognized']
            users.append(item)
        actions=[{k:v for k,v in x.items() if k in ('frame','kind','text','final','native_speech_units')} for x in t['actions']]
        for action in actions:
            if action['kind']=='speak':action['output_interval_seconds']=[action['frame'],action['frame']+len(action['native_speech_units'])]
        rows.append(dict(id=record['id'],user=users,actions=actions,actual_feedback=t['backend_feedback']))
    batches=[]
    for start in range(0,len(rows),28):
        selected=rows[start:start+28];payload=dict(model=MODEL,service_tier='default',store=False,reasoning=dict(effort='low'),max_output_tokens=2200,
            input=[dict(role='system',content=INSTRUCTIONS+' '+NATIVE_PROTOCOL+' After interruption, retain unsaid words rather than skip them.'),dict(role='user',content=json.dumps(selected,separators=(',',':')))],tools=[SCHEMA],tool_choice='required',parallel_tool_calls=False)
        maximum=((len(json.dumps(payload,ensure_ascii=True).encode())+2048)*12.5+2200*50)/1e6
        batches.append(dict(ids=[r['id'] for r in selected],payload=payload,maximum_cost_usd=maximum))
    prior_path=Path(a.prior_review) if getattr(a,'prior_review',None) else None
    prior=json.loads(prior_path.read_text()) if prior_path else None
    prior_cost=sum(u['input_tokens']*12.5/1e6+u['output_tokens']*50/1e6 for u in prior['usage']) if prior else 0
    maximum=sum(b['maximum_cost_usd'] for b in batches);assert len(rows)==292 and len(batches)==11 and maximum+prior_cost<=7.5, (maximum,prior_cost)
    result=dict(schema='dl-paced-native-teacher-semantic-request-v1',model=MODEL,manifest_sha256=sha(root/'manifest.json'),audit_sha256=sha(a.audit),word_quality_sha256=sha(a.word_quality),assistant_word_quality_sha256=audit['assistant_word_quality_sha256'],batches=batches,maximum_total_cost_usd=maximum,maximum_requests=11,concurrency=2,
        maximum_semantic_usage_usd=7.5,maximum_voice_usage_usd=2.5,combined_external_cap_usd=10,pricing_source='https://developers.openai.com/api/docs/models/gpt-6-astra',input_cost_ceiling_per_million=12.5,output_cost_per_million=50,price_basis='Standard short-context; input includes cache-write ceiling and byte/token bound; no Fast or regional endpoint',owner_data=False,credentials=False,only_authored_synthetic_data=True)
    result.update(prior_review_sha256=sha(prior_path) if prior_path else None,prior_review_cost_upper_usd=prior_cost,all_reviews_maximum_cost_usd=prior_cost+maximum,native_protocol=NATIVE_PROTOCOL)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(requests=11,scenes=292,maximum_usd=maximum,all_reviews_maximum_usd=prior_cost+maximum,prepared_sha256=sha(a.output))))

def execute(a):
    prepared_path=Path(a.prepared);prepared=json.loads(prepared_path.read_text());assert prepared['maximum_total_cost_usd']<=7.5 and prepared.get('all_reviews_maximum_cost_usd',prepared['maximum_total_cost_usd'])<=7.5 and len(prepared['batches'])==11 and prepared['concurrency']==2
    keyfile=Path('runs/local-migration-20261021/.private/openai-backend.env');assert keyfile.stat().st_mode&0o077==0
    key=None
    for line in keyfile.read_text().splitlines():
        if line.startswith('OPENAI_API_KEY='):
            value=shlex.split(line.split('=',1)[1],comments=True);key=value[0] if len(value)==1 else None
    assert key,'Existing explicitly configured local reviewer key required'
    out=Path(a.output);out.mkdir(exist_ok=False);rows=[];usage=[]
    def one(index,batch):
        (out/f'batch-{index:02d}-submitted.json').write_text(json.dumps(dict(prepared_sha256=sha(prepared_path),ids=batch['ids'],maximum_cost_usd=batch['maximum_cost_usd'],automatic_retry=False))+'\n')
        request=urllib.request.Request('https://api.openai.com/v1/responses',data=json.dumps(batch['payload']).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
        try:
            with urllib.request.urlopen(request,timeout=180) as response:data=json.load(response)
        except urllib.error.HTTPError as error:
            (out/f'batch-{index:02d}-http.json').write_text(json.dumps(dict(status=error.code))+'\n');raise RuntimeError('Reviewer HTTP'+str(error.code)) from None
        (out/f'batch-{index:02d}-raw.json').write_text(json.dumps(data,indent=2)+'\n')
        assert data['status']=='completed' and data.get('service_tier','default')=='default'
        calls=[x for x in data['output'] if x['type']=='function_call'];assert len(calls)==1 and calls[0]['name']==SCHEMA['name']
        reviewed=json.loads(calls[0]['arguments'])['rows'];assert len(reviewed)==len(batch['ids']) and {r['id'] for r in reviewed}==set(batch['ids'])
        return reviewed,data.get('usage')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(one,i,b) for i,b in enumerate(prepared['batches'])]
        try:
            for future in concurrent.futures.as_completed(futures):
                reviewed,bill=future.result();rows.extend(reviewed);usage.append(bill)
                print(json.dumps(dict(reviewed=len(rows),required=292,passed=sum(r['passed'] for r in rows))),flush=True)
        except BaseException:
            for future in futures:future.cancel()
            raise
    result=dict(all_passed=len(rows)==292 and all(r['passed'] for r in rows),scenes=len(rows),rows=rows,usage=usage,prepared_sha256=sha(prepared_path),manifest_sha256=prepared['manifest_sha256'],independent_model=MODEL,model_behavior_qualified=False,subset_acceptance=False,maximum_total_cost_usd=prepared['maximum_total_cost_usd'])
    (out/'semantic-quality.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['all_passed']:raise RuntimeError('Complete292semantic review failed; no optimizer approval')

def approve(a):
    root=Path(a.timelines);word=json.loads(Path(a.word_quality).read_text());audit=json.loads(Path(a.audit).read_text());semantic=json.loads(Path(a.semantic_quality).read_text());prep=json.loads(Path(a.prepared).read_text())
    assert audit['passed'] and word['all_passed'] and word['passed']==word['requested']==218 and semantic['all_passed'] and semantic['scenes']==292
    assert audit['manifest_sha256']==semantic['manifest_sha256']==prep['manifest_sha256']==sha(root/'manifest.json') and semantic['prepared_sha256']==sha(a.prepared)
    assert prep['word_quality_sha256']==sha(a.word_quality) and prep['audit_sha256']==sha(a.audit) and prep['assistant_word_quality_sha256']==audit['assistant_word_quality_sha256']
    result=dict(approved=True,manifest_sha256=sha(root/'manifest.json'),source_audit_sha256=sha(a.audit),word_quality_sha256=sha(a.word_quality),assistant_word_quality_sha256=audit['assistant_word_quality_sha256'],semantic_quality_sha256=sha(a.semantic_quality),train=216,dev=52,holdout=24,native_speech_timing_schema='word-aligned-native-v1',behavior_qualified=False)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','execute','approve']);[p.add_argument('--'+x) for x in ('timelines','word-quality','audit','prepared','semantic-quality','prior-review')];p.add_argument('--output',required=True);a=p.parse_args();dict(prepare=prepare,execute=execute,approve=approve)[a.stage](a)
