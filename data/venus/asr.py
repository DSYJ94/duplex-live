"""Independent GPU word audit of EVERY unique unedited source recording."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time


def words(s):
    # Fixed ordinary orthography only. Cue words, values and homophones stay distinct.
    s=s.lower().replace('colour','color').replace('vapour','vapor').replace('organise','organize')
    s=re.sub(r'\blookup\b','look up',s)
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?",s)


def wer(reference,hypothesis):
    a,b=words(reference),words(hypothesis);row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(row[j]+1,nxt[j-1]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]/max(1,len(a))


def main(a):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq,AutoProcessor,pipeline
    import soundfile as sf
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    root=Path(a.audio);receipt=root/'render-receipts.json';source=json.loads(receipt.read_text());assert source['complete']
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    revision='41f01f3fe87f28c78e2fbf8b568835947dd65ed9'
    model=AutoModelForSpeechSeq2Seq.from_pretrained(a.model,revision=revision,torch_dtype=torch.float16,low_cpu_mem_usage=True,attn_implementation='eager').cuda().eval()
    processor=AutoProcessor.from_pretrained(a.model,revision=revision)
    pipe=pipeline('automatic-speech-recognition',model=model,tokenizer=processor.tokenizer,feature_extractor=processor.feature_extractor,torch_dtype=torch.float16,device=0)
    results=[];start=time.monotonic()
    for row in source['utterances']:
        wav=root/(row['id']+'.wav');assert hashlib.sha256(wav.read_bytes()).hexdigest()==row['sha256']
        data,sr=sf.read(wav,dtype='float32');assert sr==16000 and data.size>0
        result=pipe(dict(array=data,sampling_rate=sr),return_timestamps='word',generate_kwargs=dict(language='en',task='transcribe',do_sample=False))
        error=wer(row['text'],result['text']);passed=error<=.05 and float(abs(data).max())<=.995
        value=dict(id=row['id'],source_sha256=row['sha256'],authored=row['text'],recognized=result['text'],word_error_rate=error,peak=float(abs(data).max()),words=result.get('chunks'),passed=passed)
        (out/(row['id']+'.json')).write_text(json.dumps(value,indent=2)+'\n');results.append(value)
        print(json.dumps(dict(id=row['id'],completed=len(results),total=len(source['utterances']),passed=passed,wer=error,elapsed=time.monotonic()-start)),flush=True)
    report=dict(all_passed=len(results)==len(source['utterances']) and all(x['passed'] for x in results),requested=len(source['utterances']),passed=sum(x['passed'] for x in results),rows=results,recognizer=a.model,recognizer_revision=getattr(model.config,'_commit_hash',None),audio_receipts_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),wer_threshold=.05,peak_threshold=.995,subset_acceptance=False)
    (out/'quality.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_passed']:raise RuntimeError('Complete source word audit failed; no TRAIN approval')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audio',required=True);p.add_argument('--model',default='openai/whisper-large-v3-turbo');p.add_argument('--output',required=True);main(p.parse_args())
