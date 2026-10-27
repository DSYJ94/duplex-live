"""One bounded repair of the three measured accent-source waveform failures."""
import argparse,array,hashlib,json,os,shutil,urllib.request,urllib.error,wave
from pathlib import Path
from datetime import datetime,timezone

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(a):
    plan=json.loads(Path(a.plan).read_text());assert plan['requests']==3 and plan['characters']==209 and plan['maximum_total_api_usage_usd']<=10
    assert plan['prior_api_usage_upper_usd']==9.5431125 and abs(plan['maximum_additional_usage_usd']-209*.0005)<1e-9
    key=os.environ.get('ELEVENLABS_API_KEY');assert key;requested=0
    for group in plan['groups']:
        root,out=Path(group['source']),Path(group['output']);prepared=json.loads(Path(group['prepared']).read_text());source=json.loads((root/'render-receipts.json').read_text());diagnosis=json.loads(Path(group['diagnosis']).read_text());assert source['complete'] and len(source['utterances'])==source['required'] in (12,95)
        assert sha(root/'render-receipts.json')==group['source_receipts_sha256'] and sha(group['diagnosis'])==group['diagnosis_sha256']
        failed=[]
        for row in source['utterances']:
            raw=(root/row['id']/'source.pcm').read_bytes();assert hashlib.sha256(raw).hexdigest()==row['raw_pcm_sha256'] and sha(root/row['id']/'source.wav')==row['wav_sha256']
            peak=max(abs(x) for x in array.array('h',raw))/32768;assert peak==row['peak']
            if peak>.995:failed.append(row['id'])
        assert set(failed)=={r['id'] for r in diagnosis}==set(group['failed_ids'])
        out.mkdir(exist_ok=False);rows=[];requests={r['id']:r for r in prepared['requests']}
        (out/'repair-plan.json').write_text(json.dumps(dict(plan_sha256=sha(a.plan),group=group,seed=20261101,stability=1.0,gain_changed=False,automatic_retry=False),indent=2)+'\n')
        for old in source['utterances']:
            ident=old['id'];folder=out/ident
            if ident not in failed:shutil.copytree(root/ident,folder);rows.append(old);continue
            folder.mkdir();request=requests[ident];body=dict(request['payload'],seed=20261101,voice_settings=dict(request['payload']['voice_settings'],stability=1.0));assert body['text']==old['text'] and request['voice_id']==old['voice_id']
            frozen=dict(voice_id=old['voice_id'],payload=body);pending=dict(id=ident,request=frozen,request_sha256=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest(),at_utc=datetime.now(timezone.utc).isoformat(),status='SUBMITTED_PENDING_NO_RETRY')
            (folder/'attempt.json').write_text(json.dumps(pending,indent=2)+'\n');requested+=1;assert requested<=3
            req=urllib.request.Request('https://api.elevenlabs.io/v1/text-to-speech/'+old['voice_id']+'?output_format=pcm_24000',data=json.dumps(body).encode(),headers={'xi-api-key':key,'Content-Type':'application/json'},method='POST')
            try:
                with urllib.request.urlopen(req,timeout=90) as response:raw=response.read();request_id=response.headers.get('request-id')
            except urllib.error.HTTPError as e:raise RuntimeError('Accent repair HTTP'+str(e.code)+'; no retry') from None
            assert raw and len(raw)%2==0;(folder/'source.pcm').write_bytes(raw)
            with wave.open(str(folder/'source.wav'),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(raw)
            samples=array.array('h',raw);peak=max(abs(x) for x in samples)/32768
            row=dict(old,request_id=request_id,request_sha256=pending['request_sha256'],raw_pcm_sha256=hashlib.sha256(raw).hexdigest(),wav_sha256=sha(folder/'source.wav'),seconds=len(samples)/24000,peak=peak,peak_pass=peak<=.995,parent_failed_wav_sha256=old['wav_sha256'],regeneration_seed=20261101,regeneration_stability=1.0)
            (folder/'receipt.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row);print(json.dumps(dict(id=ident,peak=peak,peak_pass=row['peak_pass'])),flush=True)
        result=dict(source,utterances=rows,parent_receipts_sha256=sha(root/'render-receipts.json'),repaired_clips=len(failed),all_peak_checks=all(r['peak_pass'] for r in rows),repair_plan_sha256=sha(a.plan),training_approved=False)
        (out/'render-receipts.json').write_text(json.dumps(result,indent=2)+'\n')
    assert requested==3

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);main(p.parse_args())
