"""Bounded ElevenLabs assistant-source rendering; no ambiguous automatic retries.

Preparation is offline. Execution reads only ELEVENLABS_API_KEY from the calling
process and checks account quota first. Credentials never enter artifacts.
"""
import argparse,array,hashlib,json,math,os,time,urllib.request,urllib.error,wave
from pathlib import Path
from datetime import datetime,timezone

RATE_CEILING=.0005  # USD/character, conservative versus published0.10/1K.
VOICE_BUDGET=3.0    # Inside the approved10USD API /40USD total repair allowance.
MODEL='eleven_v3'
VOICE='21m00Tcm4TlvDq8ikWAM'  # Existing stock voice; no voice cloning.

def payloads(plan):
    rows=[dict(id=c['id'],split=c['split'],text=c['assistant_text']) for c in plan['cases'] if c['assistant_text']]
    rows.append(dict(id='voice-v1-reference',split='REFERENCE_ONLY',text=plan['reference_text']))
    result=[]
    for row in rows:
        body=dict(text=row['text'],model_id=MODEL,language_code='en',seed=20261027,
            voice_settings=dict(stability=.5,similarity_boost=.75))
        result.append(dict(id=row['id'],split=row['split'],payload=body,
            payload_sha256=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()))
    assert len(result)==67 and sum(len(r['payload']['text']) for r in result)<=12000
    return result

def main(a):
    src=Path(a.plan);plan=json.loads(src.read_text());rows=payloads(plan)
    characters=sum(len(r['payload']['text']) for r in rows);upper=characters*RATE_CEILING
    assert upper<=VOICE_BUDGET and VOICE_BUDGET<=10
    out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
    frozen=dict(plan_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),requests=rows,
        silence_targets=[dict(id=c['id'],split=c['split'],exact_silent_target=True,tts_request=False) for c in plan['cases'] if not c['assistant_text']],
        voice_id=VOICE,model_id=MODEL,characters=characters,conservative_usage_upper_usd=upper,
        voice_usage_cap_usd=VOICE_BUDGET,combined_external_usage_cap_usd=10,
        pricing_source='https://elevenlabs.io/pricing/api',published_v3_usd_per_1000_characters=.10,
        no_automatic_retry=True,provenance_class='closed-content',authority='ADR0003 Amendment A',qualified=False)
    (out/'prepared-requests.json').write_text(json.dumps(frozen,indent=2)+'\n')
    if not a.execute:print(json.dumps(dict(prepared_requests=len(rows),characters=characters,conservative_upper_usd=upper,external_api_calls=0)));return
    key=os.environ.get('ELEVENLABS_API_KEY');assert key,'Configure existing authorized key locally, never in artifacts'
    def request(path,body=None):
        headers={'xi-api-key':key}
        if body is not None:headers['Content-Type']='application/json'
        req=urllib.request.Request('https://api.elevenlabs.io/v1/'+path,headers=headers,data=json.dumps(body).encode() if body is not None else None)
        return urllib.request.urlopen(req,timeout=90)
    with request('user/subscription') as response:account=json.load(response)
    with request('models') as response:models=json.load(response)
    selected=next(m for m in models if m['model_id']==MODEL and m['can_do_text_to_speech'])
    cost_factor=selected.get('token_cost_factor',1) or 1
    remaining=account['character_limit']-account['character_count'];assert remaining>=math.ceil(characters*cost_factor),'Insufficient prepaid speech allowance; never buy or change overages'
    (out/'account-preflight.json').write_text(json.dumps(dict(at_utc=datetime.now(timezone.utc).isoformat(),tier=account.get('tier'),characters_remaining=remaining,required_characters_upper=math.ceil(characters*cost_factor),model=MODEL),indent=2)+'\n')
    receipts=[]
    for index,row in enumerate(rows):
        folder=out/row['id'];folder.mkdir()
        pending=dict(row,status='SUBMITTED_OUTCOME_PENDING',submitted_utc=datetime.now(timezone.utc).isoformat(),reserved_usage_upper_usd=len(row['payload']['text'])*RATE_CEILING)
        (folder/'attempt.json').write_text(json.dumps(pending,indent=2)+'\n')
        # On HTTP/transport failure stop. The pending record prevents a blind
        # retry after a possibly charged generation or observation timeout.
        try:
            with request('text-to-speech/'+VOICE+'?output_format=pcm_24000',row['payload']) as response:
                raw=response.read();request_id=response.headers.get('request-id')
        except urllib.error.HTTPError as error:raise RuntimeError('Speech API failed with HTTP'+str(error.code)+'; original attempt preserved; no retry') from None
        assert raw and len(raw)%2==0
        (folder/'source.pcm').write_bytes(raw)
        wav=folder/'source.wav'
        with wave.open(str(wav),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(raw)
        samples=array.array('h',raw);peak=max(abs(s) for s in samples)/32768
        receipt=dict(id=row['id'],split=row['split'],provider='ElevenLabs',model_id=MODEL,voice_id=VOICE,
            request_id=request_id,payload_sha256=row['payload_sha256'],text=row['payload']['text'],
            raw_pcm_sha256=hashlib.sha256(raw).hexdigest(),wav_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),
            sample_rate=24000,seconds=len(samples)/24000,peak=peak,peak_pass=peak<=.995,
            source_created_utc=datetime.now(timezone.utc).isoformat(),provenance_class='closed-content',
            provenance_authority='decisions/0003-provenance-classes.md Amendment A',independent_words='PENDING',naturalness='PENDING_LISTENING',training_approved=False)
        (folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');receipts.append(receipt)
        (out/'render-receipts.json').write_text(json.dumps(dict(complete=len(receipts)==67,utterances=receipts,plan_sha256=frozen['plan_sha256'],qualified=False,conservative_total_usage_upper_usd=sum(len(x['text'])*RATE_CEILING for x in receipts)),indent=2)+'\n')
        print(json.dumps(dict(event='voice_source_rendered',completed=len(receipts),required=67,id=row['id'],peak_pass=peak<=.995)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true');main(p.parse_args())
