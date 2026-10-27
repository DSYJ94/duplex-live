"""Independent full174 ORIGINAL24k source audit; no model on the Mac.

Numeric/currency/ordinal and English spelling rules are fixed before recognition.
No transcript hint is sent. Raw original peaks govern source clipping; floating
recognizer resampling is retained separately and is never normalized or clipped.
"""
import argparse,hashlib,json,re,time
from pathlib import Path
from .asr import words as base_words,enable_complete_word_timestamps

SMALL='zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
TENS=['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']
ORD={1:'first',2:'second',3:'third',4:'fourth',5:'fifth',6:'sixth',7:'seventh',8:'eighth',9:'ninth',10:'tenth',11:'eleventh',12:'twelfth',13:'thirteenth',14:'fourteenth',15:'fifteenth',16:'sixteenth',17:'seventeenth',18:'eighteenth',19:'nineteenth',20:'twentieth',30:'thirtieth',40:'fortieth',50:'fiftieth',60:'sixtieth',70:'seventieth',80:'eightieth',90:'ninetieth'}

def cardinal(n):
    assert 0<=n<1000
    if n<20:return SMALL[n]
    if n<100:return TENS[n//10]+(' '+SMALL[n%10] if n%10 else '')
    return SMALL[n//100]+' hundred'+(' and '+cardinal(n%100) if n%100 else '')

def ordinal(n):
    assert 1<=n<100
    return ORD[n] if n in ORD else TENS[n//10]+' '+ORD[n%10]

def words(text):
    s=text.lower()
    s=re.sub(r'£(\d{1,3})\.(\d{2})\b',lambda m:cardinal(int(m[1]))+' pounds and '+cardinal(int(m[2]))+' pence',s)
    s=re.sub(r'\b(\d{1,2})(st|nd|rd|th)\b',lambda m:ordinal(int(m[1])),s)
    s=re.sub(r'\b\d{1,3}\b',lambda m:cardinal(int(m[0])),s)
    s=re.sub(r'\bmm[\s-]*hm+\b','mmhm',s)
    for old,new in [('favourite','favorite'),('kilometres','kilometers'),('metres','meters'),('centre','center')]:s=s.replace(old,new)
    return base_words(s)

def wer(a,b):
    a,b=words(a),words(b);row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(row[j]+1,nxt[j-1]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]/max(1,len(a))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(a):
    import numpy as np,soundfile as sf,torch
    from scipy.signal import resample_poly
    from transformers import AutoModelForSpeechSeq2Seq,AutoProcessor,pipeline
    from portable.venus.native_app import ensure_guard
    from training.venus.spoken_audit import REVISION
    stop=json.loads(Path(a.lease).read_text())['stop_unix'];ensure_guard();assert torch.cuda.device_count()==1 and stop-time.time()>850
    root=Path(a.root);sources=[]
    for name,n in [('voice-eleven-v3-sources-v3-final',67),('british-accent-sources-v2-repaired',95),('british-accent-supplement-v2-repaired',12)]:
        receipt=root/name/'render-receipts.json';r=json.loads(receipt.read_text());assert r['complete'] and len(r['utterances'])==n
        for row in r['utterances']:
            wav=root/name/row['id']/'source.wav';assert sha(wav)==row['wav_sha256'];sources.append((name,row,wav,sha(receipt)))
    assert len(sources)==len({r['id'] for _,r,_,_ in sources})==174
    out=Path(a.output);out.mkdir(exist_ok=False);name='openai/whisper-large-v3-turbo'
    model=AutoModelForSpeechSeq2Seq.from_pretrained(name,revision=REVISION,local_files_only=True,torch_dtype=torch.float16,attn_implementation='eager').cuda().eval();enable_complete_word_timestamps(model)
    p=AutoProcessor.from_pretrained(name,revision=REVISION,local_files_only=True);pipe=pipeline('automatic-speech-recognition',model=model,tokenizer=p.tokenizer,feature_extractor=p.feature_extractor,torch_dtype=torch.float16,device=0);rows=[]
    for cohort,row,wav,receipt_sha in sources:
        ensure_guard()
        if stop-time.time()<300:raise TimeoutError('Preserve full backup/STOP reserve; no subset approval')
        audio,sr=sf.read(wav,dtype='float32');assert sr==24000 and audio.ndim==1 and np.isfinite(audio).all();peak=float(np.abs(audio).max());assert peak==row['peak'] and peak<=.995
        converted=resample_poly(audio,2,3).astype(np.float32)
        recognized=pipe(dict(array=converted,sampling_rate=16000),return_timestamps='word',generate_kwargs=dict(language='en',task='transcribe',do_sample=False));error=wer(row['text'],recognized['text'])
        result=dict(id=row['id'],profile=row.get('profile','american-rachel'),cohort=cohort,source_sha256=row['wav_sha256'],source_receipts_sha256=receipt_sha,split=row['split'],authored=row['text'],recognized=recognized['text'],word_error_rate=error,original_peak=peak,recognizer_resample_peak=float(np.abs(converted).max()),words=recognized.get('chunks'),passed=error<=.05 and peak<=.995,normalizer_schema='fixed-voice-English-orthography-v1')
        rows.append(result);(out/(row['id']+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(completed=len(rows),required=174,id=row['id'],passed=result['passed'],wer=error)),flush=True)
    report=dict(complete=len(rows)==174,all_passed=len(rows)==174 and all(r['passed'] for r in rows),required=174,passed=sum(r['passed'] for r in rows),rows=rows,source=a.source,recognizer=name,revision=REVISION,word_limit=.05,original_peak_limit=.995,normalizer_module_sha256=sha(__file__),no_transcript_hints=True,no_gain_or_clamp=True,subset_acceptance=False,training_approved=False,naturalness_approval=False)
    (out/'quality.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();[p.add_argument('--'+x,required=True) for x in ('root','output','lease','source')];main(p.parse_args())
