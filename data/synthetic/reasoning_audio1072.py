"""Approved complete1072 audio renderer. Credentials stay local; no retries on ambiguity."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import wave

MODEL='eleven_flash_v2_5'
VOICES={'assistant':'yf0u5Jay5VVCpmAhFyo4','user':'Xb7hH8MSUJpSbSDYk0k2'}


def digest(value):return hashlib.sha256(value).hexdigest()
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','key-file','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',choices=['eleven_flash_v2_5','eleven_v4_turbo'],default=MODEL);p.add_argument('--speed',type=float,choices=[.9,1.],default=1.);p.add_argument('--stability',type=float,choices=[.5,.75],default=.5)
    p.add_argument('--user-rp-voice',action='store_true')
    a=p.parse_args();voices=dict(VOICES,user=VOICES['assistant']) if a.user_rp_voice else dict(VOICES);model_id=a.model;settings=dict(stability=a.stability,similarity_boost=.75,speed=a.speed);raw=a.plan.read_bytes();plan=json.loads(raw)
    if not plan.get('owner_approved') or plan['maximum_total_API_USD']!=100 or len(plan['exact_speech_payloads'])!=2144:
        raise ValueError('Exact complete owner-approved100USD cohort required')
    secret=a.key_file.read_text();match=re.search(r'eleven\s*labs',secret,re.I)
    if not match:raise ValueError('Existing authorized ElevenLabs credential section missing')
    keys=re.findall(r'(?<![A-Za-z0-9])(sk_[A-Za-z0-9]{20,}|[a-f0-9]{32,64})(?![A-Za-z0-9])',secret[match.start():match.start()+800])
    if len(set(keys))!=1:raise ValueError('Ambiguous credential selection; nothing sent')
    headers={'xi-api-key':keys[0],'Content-Type':'application/json'}
    def get(endpoint):
        with urllib.request.urlopen(urllib.request.Request('https://api.elevenlabs.io/v1/'+endpoint,headers=headers),timeout=30) as r:return json.load(r)
    sub=get('user/subscription');models=get('models')
    model=next(m for m in models if m['model_id']==model_id)
    rate=model['model_rates']['character_cost_multiplier']*model['model_rates']['cost_discount_multiplier']
    if rate!=.5 or sub.get('max_credit_limit_extension')!=0:raise ValueError('Exact discount and disabled overage required')
    for voice in voices.values():
        info=get('voices/'+voice)
        if info.get('labels',{}).get('accent')!='british':raise ValueError('Approved British voice required')
        sharing=info.get('sharing') or {}
        if sharing.get('financial_rewards_enabled') or sharing.get('rate',1)>1:raise ValueError('Unexpected voice fee')
    a.out.mkdir(parents=True,exist_ok=True)
    lock=a.out/'writer.lock';lock.mkdir(exist_ok=False)
    try:
        ledger=a.out/'admission.json'
        identity=dict(plan_sha256=digest(raw),model=model_id,voice_settings=settings,voices=voices,rate=rate,required=2144,credits_only=True,USD_cap=100)
        if ledger.exists():
            if json.loads(ledger.read_text())['identity']!=identity:raise ValueError('Existing collection differs')
        else:write(ledger,dict(identity=identity,initial_used=sub['character_count'],initial_limit=sub['character_limit']))
        payloads=plan['exact_speech_payloads'];remaining=[]
        for row in payloads:
            done=a.out/(row['id']+'.json');wav=a.out/(row['id']+'.wav');pending=a.out/(row['id']+'.inflight.json')
            if done.exists():
                receipt=json.loads(done.read_text())
                if receipt['audio_sha256']!=digest(wav.read_bytes()) or receipt['text_sha256']!=digest(row['text'].encode()):raise ValueError('Completed artifact changed')
            elif pending.exists():raise ValueError('Ambiguous previous request: preserve and reconcile, never blindly retry')
            else:remaining.append(row)
        reserve=sum(len(r['text']) for r in remaining)*rate*1.2
        if sub['character_limit']-sub['character_count']<reserve:raise ValueError('Whole remaining cohort plus20percent does not fit included credits')
        count=len(payloads)-len(remaining)
        for row in remaining:
            if count%25==0:
                current=get('user/subscription')
                if current.get('max_credit_limit_extension')!=0:raise ValueError('Overage protection changed')
                need=sum(len(r['text']) for r in payloads if not (a.out/(r['id']+'.json')).exists())*rate*1.2
                if current['character_limit']-current['character_count']<need:raise ValueError('Credits no longer admit whole remainder')
            role=row['id'].rsplit('-',1)[1];voice=voices[role]
            request=dict(text=row['text'],model_id=model_id,voice_settings=settings,language_code='en')
            pending=a.out/(row['id']+'.inflight.json');write(pending,dict(id=row['id'],text_sha256=digest(row['text'].encode()),voice_id=voice,model=model_id))
            req=urllib.request.Request('https://api.elevenlabs.io/v1/text-to-speech/'+voice+'?output_format=pcm_16000',data=json.dumps(request).encode(),headers=headers,method='POST')
            with urllib.request.urlopen(req,timeout=120) as response:
                pcm=response.read();request_id=response.headers.get('request-id');cost=response.headers.get('character-cost')
            if not pcm or len(pcm)%2:raise ValueError('Invalid PCM; request retained for reconciliation')
            path=a.out/(row['id']+'.wav')
            with wave.open(str(path),'wb') as wav:wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(pcm)
            write(a.out/(row['id']+'.json'),dict(id=row['id'],fold=row['fold'],voice_id=voice,model=model_id,text_sha256=digest(row['text'].encode()),audio_sha256=digest(path.read_bytes()),characters=len(row['text']),request_id=request_id,reported_character_cost=cost,qualified=False))
            count+=1
            print(json.dumps(dict(completed=count,required=2144,id=row['id'],qualified=False)),flush=True)
        write(a.out/'collection-complete.json',dict(identity=identity,completed=count,qualified=False,semantic_word_timing_quality_pending=True))
    finally:lock.rmdir()


if __name__=='__main__':main()
