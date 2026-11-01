"""Whole72 original-source inventory and independent GPU-only word audit.

Numeric/spelling normalization is fixed before recognition; numbers and cue
words remain distinct. Source waveforms are never attenuated or rewritten.
"""
import argparse
import array
import hashlib
import json
from pathlib import Path
import re
import wave
from .asr import words as base_words

SMALL='zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
TENS=['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']


def number(n):
    if not 0<=n<1000000:raise ValueError('Unadmitted numeric range')
    if n<20:return SMALL[n]
    if n<100:return TENS[n//10]+(' '+number(n%10) if n%10 else '')
    if n<1000:return number(n//100)+' hundred'+(' and '+number(n%100) if n%100 else '')
    return number(n//1000)+' thousand'+(' '+number(n%1000) if n%1000 else '')


def words(text):
    s=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',text.lower())
    s=re.sub(r'£(\d+)\.(\d{2})\b',lambda m:number(int(m[1]))+' pounds'+(' and '+number(int(m[2]))+' pence' if int(m[2]) else ''),s)
    s=re.sub(r'£(\d+)\b',r'\1 pounds',s)
    s=re.sub(r'\b(\d+)\.(\d+)\b',lambda m:number(int(m[1]))+' point '+' '.join(number(int(c)) for c in m[2]),s)
    s=re.sub(r'\b\d+\b',lambda m:number(int(m[0])),s).replace('%',' percent')
    s=s.replace('metres','meters').replace('litres','liters').replace('litre','liter')
    s=re.sub(r'\bper cent\b','percent',s)
    return base_words(s)


def wer(reference,hypothesis):
    a,b=words(reference),words(hypothesis);row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(row[j]+1,nxt[j-1]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]/max(1,len(a))


def inventory(root,payload,required=72):
    ledger=json.loads((root/'ledger.json').read_text())
    if required not in (34,72) or not ledger['complete'] or len(ledger['attempts'])!=required:raise ValueError('Actual complete admitted cohort required')
    raw=payload.read_bytes();plan=json.loads(raw)
    if hashlib.sha256(raw).hexdigest()!=ledger['payload_sha256']:raise ValueError('Original payload changed')
    attempts={x['id']:x for x in ledger['attempts']}
    if len(attempts)!=required or len(plan['requests'])!=required or set(attempts)!={x['id'] for x in plan['requests']}:raise ValueError('Exact full cohort required')
    rows=[]
    for request in plan['requests']:
        row=attempts[request['id']];wav=root/request['id']/'original.wav';pcm=root/request['id']/'original.pcm'
        if row['status']!='complete' or row['fold']!=request['fold'] or row['payload_sha256']!=request['payload_sha256']:raise ValueError('Receipt mismatch')
        for path,key in ((wav,'wav_sha256'),(pcm,'pcm_sha256')):
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row[key]:raise ValueError('Source hash changed')
        with wave.open(str(wav)) as w:
            if (w.getnchannels(),w.getsampwidth(),w.getframerate())!=(1,2,24000):raise ValueError('Original24k mono PCM required')
            raw=w.readframes(w.getnframes())
        if raw!=pcm.read_bytes() or not raw:raise ValueError('Original PCM mismatch/empty')
        peak=max(abs(v) for v in array.array('h',raw))/32768
        rows.append(dict(id=request['id'],fold=request['fold'],text=request['body']['text'],wav=str(wav),sha256=row['wav_sha256'],seconds=len(raw)/48000,peak=peak,peak_pass=peak<=.995))
    return rows


def main(a):
    rows=inventory(a.root,a.payload,a.required)
    if a.output.exists():raise ValueError('Do not overwrite audit evidence')
    report=dict(schema='dl-context-source-quality-v1',required=a.required,source_inventory_complete=True,all_peaks_pass=all(x['peak_pass'] for x in rows),rows=rows,training_approved=False,word_audit_complete=False,wer_threshold=.05,peak_threshold=.995,subset_acceptance=False)
    if not a.inventory_only:
        import numpy as np
        import torch
        from scipy.signal import resample_poly
        from transformers import AutoModelForSpeechSeq2Seq,AutoProcessor,pipeline
        from portable.venus.native_app import ensure_guard
        from .asr import enable_complete_word_timestamps
        ensure_guard()
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('One guarded GPU; never run ASR on this Mac CPU')
        name='openai/whisper-large-v3-turbo';revision='41f01f3fe87f28c78e2fbf8b568835947dd65ed9'
        model=AutoModelForSpeechSeq2Seq.from_pretrained(name,revision=revision,local_files_only=True,torch_dtype=torch.float16,attn_implementation='eager').cuda().eval();enable_complete_word_timestamps(model)
        p=AutoProcessor.from_pretrained(name,revision=revision,local_files_only=True)
        recognize=pipeline('automatic-speech-recognition',model=model,tokenizer=p.tokenizer,feature_extractor=p.feature_extractor,device=0)
        a.output.mkdir(parents=True,exist_ok=False)
        for row in rows:
            ensure_guard()
            with wave.open(row['wav']) as w:raw=w.readframes(w.getnframes())
            data=np.frombuffer(raw,dtype='<i2').astype(np.float32)/32768
            result=recognize(dict(array=resample_poly(data,2,3),sampling_rate=16000),return_timestamps='word',generate_kwargs=dict(language='en',task='transcribe',do_sample=False))
            row.update(recognized=result['text'],word_error_rate=wer(row['text'],result['text']),words=result.get('chunks'))
            row['passed']=row['peak_pass'] and row['word_error_rate']<=.05
            (a.output/(row['id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
        report.update(word_audit_complete=True,recognizer=name,revision=revision,all_sources_pass=all(x['passed'] for x in rows))
        report['all'+str(a.required)+'_pass']=report['all_sources_pass']
    else:a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
    if not a.inventory_only and not report['all_sources_pass']:raise RuntimeError('Complete source cohort failed; preserve originals and no subset acceptance')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--payload',type=Path,default=Path('docs/NATIVE-VENUS-CONTEXT-REPAIR-SPEECH-PAYLOAD-2026-11-01.json'));p.add_argument('--output',type=Path,required=True);p.add_argument('--inventory-only',action='store_true');p.add_argument('--required',type=int,choices=[34,72],default=72);main(p.parse_args())
