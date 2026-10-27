"""Regenerate only two waveform-diagnosed failed clips, preserving complete67."""
import argparse,array,hashlib,json,os,shutil,urllib.request,urllib.error,wave
from pathlib import Path
from datetime import datetime,timezone
from .render_voice import payloads,RATE_CEILING,VOICE_BUDGET,VOICE,MODEL

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(a):
    root,out=Path(a.source),Path(a.output);plan_path=Path(a.plan);plan=json.loads(plan_path.read_text());receipt=json.loads((root/'render-receipts.json').read_text())
    assert receipt['complete'] and len(receipt['utterances'])==67 and receipt['plan_sha256']==sha(plan_path)
    failed=[]
    for row in receipt['utterances']:
        raw=(root/row['id']/'source.pcm').read_bytes();assert hashlib.sha256(raw).hexdigest()==row['raw_pcm_sha256'] and sha(root/row['id']/'source.wav')==row['wav_sha256']
        samples=array.array('h',raw);peak=max(abs(x) for x in samples)/32768;assert peak==row['peak']
        if peak>.995:failed.append(row)
    final=getattr(a,'final_one',False);expected={'voice-v1-honest_uncertainty-06'} if final else {'voice-v1-honest_uncertainty-06','voice-v1-resume_after_interruption-01'}
    assert {r['id'] for r in failed}==expected
    seed=20261029 if final else 20261028
    prior=receipt['conservative_total_usage_upper_usd'];extra=sum(len(r['text']) for r in failed)*RATE_CEILING;assert prior+extra<=VOICE_BUDGET==2.5 and 3.4041125+prior+extra<10
    frozen={r['id']:r for r in payloads(plan)};key=os.environ.get('ELEVENLABS_API_KEY');assert key
    out.mkdir(exist_ok=False)
    (out/'repair-plan.json').write_text(json.dumps(dict(source_receipts_sha256=sha(root/'render-receipts.json'),ids=[r['id'] for r in failed],exact_words_preserved=True,seed=seed,final_bounded_attempt=final,stability=1.0 if final else .5,reason='Actual raw PCM rail sample at32767, over unchangedpeak.995; regeneration only, no gain adjustment',prior_usage_upper_usd=prior,extra_usage_upper_usd=extra,combined_voice_usage_upper_usd=prior+extra,automatic_retry=False),indent=2)+'\n')
    rows=[]
    for old in receipt['utterances']:
        ident=old['id'];folder=out/ident
        if ident not in {r['id'] for r in failed}:shutil.copytree(root/ident,folder);rows.append(old);continue
        folder.mkdir();entry=frozen[ident];body=dict(entry['payload'],seed=seed)
        if final:body['voice_settings']=dict(body['voice_settings'],stability=1.0)
        assert body['text']==old['text']
        pending=dict(id=ident,payload=body,payload_sha256=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),submitted_utc=datetime.now(timezone.utc).isoformat(),outcome='PENDING_NO_RETRY',maximum_usage_usd=len(old['text'])*RATE_CEILING)
        (folder/'attempt.json').write_text(json.dumps(pending,indent=2)+'\n')
        req=urllib.request.Request('https://api.elevenlabs.io/v1/text-to-speech/'+VOICE+'?output_format=pcm_24000',data=json.dumps(body).encode(),headers={'xi-api-key':key,'Content-Type':'application/json'},method='POST')
        try:
            with urllib.request.urlopen(req,timeout=90) as response:raw=response.read();request_id=response.headers.get('request-id')
        except urllib.error.HTTPError as error:raise RuntimeError('Speech repair HTTP'+str(error.code)+'; no retry; original retained') from None
        assert raw and len(raw)%2==0;(folder/'source.pcm').write_bytes(raw)
        with wave.open(str(folder/'source.wav'),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(raw)
        samples=array.array('h',raw);peak=max(abs(x) for x in samples)/32768
        row=dict(old,request_id=request_id,payload_sha256=pending['payload_sha256'],raw_pcm_sha256=hashlib.sha256(raw).hexdigest(),wav_sha256=sha(folder/'source.wav'),seconds=len(samples)/24000,peak=peak,peak_pass=peak<=.995,source_created_utc=datetime.now(timezone.utc).isoformat(),parent_failed_wav_sha256=old['wav_sha256'],regeneration_seed=seed,regeneration_stability=1.0 if final else .5,training_approved=False)
        (folder/'receipt.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
        print(json.dumps(dict(id=ident,regenerated=True,peak=peak,peak_pass=peak<=.995)),flush=True)
    result=dict(receipt,utterances=rows,parent_receipts_sha256=sha(root/'render-receipts.json'),repaired_clips=len(failed),unchanged_clips=67-len(failed),all_peak_checks=all(r['peak_pass'] for r in rows),conservative_total_usage_upper_usd=prior+extra,training_approved=False,independent_words='PENDING_GPU_AUDIT',naturalness='PENDING_LISTENING')
    (out/'render-receipts.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['all_peak_checks'],'Entire67 remains failed; no passing subset'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final-one',action='store_true');[p.add_argument('--'+x,required=True) for x in ('plan','source','output')];main(p.parse_args())
