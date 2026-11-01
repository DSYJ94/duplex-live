"""Explicitly admitted source generation. No retries or source overwrites."""
import argparse
import array
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import wave

PAYLOAD=Path('docs/NATIVE-VENUS-CONTEXT-REPAIR-SPEECH-PAYLOAD-2026-11-01.json')
AUTH=Path('docs/NATIVE-VENUS-CONTEXT-REPAIR-SPEECH-AUTHORIZATION-2026-11-01.json')


def admission():
    raw=PAYLOAD.read_bytes(); plan=json.loads(raw)
    if not AUTH.exists():raise ValueError('Exact new source-payload authorization required before credential access')
    auth=json.loads(AUTH.read_text())
    if auth.get('status')!='APPROVED' or auth.get('payload_sha256')!=hashlib.sha256(raw).hexdigest():
        raise ValueError('Authorization must bind exact payload')
    if auth.get('maximum_paid_calls')!=72 or auth.get('maximum_additional_api_usd')!=5:
        raise ValueError('Exact call and cost grant required')
    rows=plan['requests']
    if len(rows)!=72 or sum(len(r['body']['text']) for r in rows)!=9473:raise ValueError('Whole cohort changed')
    for r in rows:
        if r['payload_sha256']!=hashlib.sha256(json.dumps(r['body'],sort_keys=True).encode()).hexdigest():raise ValueError('Body changed')
        if r['voice_id']!='JBFqnCBsd6RMkjVDRZzb' or r['body']['model_id']!='eleven_v3':raise ValueError('No voice/model substitution')
    return plan


def execute(output):
    plan=admission()  # Must precede any credential access or network operation.
    if output.exists():raise ValueError('Never restart or overwrite an existing attempt directory')
    key=os.environ.get('ELEVENLABS_API_KEY')
    if not key:raise ValueError('Existing key must be configured locally; never put credentials in artifacts')
    def request(path,body=None):
        headers={'xi-api-key':key}
        if body is not None:headers['Content-Type']='application/json'
        req=urllib.request.Request('https://api.elevenlabs.io'+path,headers=headers,data=None if body is None else json.dumps(body).encode())
        with urllib.request.urlopen(req,timeout=100) as response:return response.read(),response.headers.get('request-id')
    sub=json.loads(request('/v1/user/subscription')[0])
    if sub['character_limit']-sub['character_count']<plan['characters']:raise ValueError('Whole cohort must fit existing quota; no subscription purchase')
    models=json.loads(request('/v1/models')[0])
    if not any(m['model_id']=='eleven_v3' for m in models):raise ValueError('Exact model unavailable')
    request('/v1/voices/JBFqnCBsd6RMkjVDRZzb')
    output.mkdir(parents=True,exist_ok=False)
    ledger=dict(payload_sha256=hashlib.sha256(PAYLOAD.read_bytes()).hexdigest(),attempts=[],complete=False,qualified=False,automatic_retries=0)
    def save():
        temp=output/'ledger.tmp';temp.write_text(json.dumps(ledger,indent=2)+'\n');temp.replace(output/'ledger.json')
    save()
    for row in plan['requests']:
        entry=dict(id=row['id'],status='attempt_reserved',payload_sha256=row['payload_sha256'])
        ledger['attempts'].append(entry);save()
        try:
            raw,request_id=request('/v1/text-to-speech/'+row['voice_id']+'?output_format=pcm_24000',row['body'])
            if not raw or len(raw)%2:raise ValueError('Invalid PCM output')
            folder=output/row['id'];folder.mkdir();(folder/'original.pcm').write_bytes(raw)
            wav=folder/'original.wav'
            with wave.open(str(wav),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(raw)
            values=array.array('h',raw);peak=max(abs(v) for v in values)/32768
            entry.update(status='complete',request_id=request_id,fold=row['fold'],pcm_sha256=hashlib.sha256(raw).hexdigest(),wav_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),seconds=len(values)/24000,peak=peak,peak_pass=peak<=.995,independent_words='PENDING',training_approved=False,provenance_class='closed-content',model_id='eleven_v3',voice_id=row['voice_id'])
            save()
        except Exception as error:
            entry.update(status='failed_or_unknown',error_type=type(error).__name__);save()
            raise RuntimeError('Attempt preserved; no retry or subset acceptance') from None
    ledger.update(complete=True,conservative_cost_upper_usd=plan['conservative_cost_upper_usd']);save()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    execute(parser.parse_args().output)
