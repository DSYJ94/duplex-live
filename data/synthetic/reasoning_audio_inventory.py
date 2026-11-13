"""Complete-cohort raw PCM/source inventory; not independent word approval."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import wave


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--audio',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());rows=[]
    if not (a.audio/'collection-complete.json').exists():raise ValueError('Real complete collection required')
    for row in plan['exact_speech_payloads']:
        path=a.audio/(row['id']+'.wav');receipt=json.loads((a.audio/(row['id']+'.json')).read_text())
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=receipt['audio_sha256'] or hashlib.sha256(row['text'].encode()).hexdigest()!=receipt['text_sha256']:raise ValueError('Original source/PCM differs')
        with wave.open(str(path),'rb') as f:
            if (f.getframerate(),f.getnchannels(),f.getsampwidth())!=(16000,1,2):raise ValueError('Exact source PCM clock required')
            n=f.getnframes();raw=f.readframes(n)
        peak=max(abs(v[0]) for v in struct.iter_unpack('<h',raw))/32768 if raw else 0
        rows.append(dict(id=row['id'],fold=row['fold'],file=path.name,sha256=digest,frames=n,seconds=n/16000,peak=peak,raw_peak_pass=peak<=.995,nonempty=n>3200,source_text=row['text']))
    if len(rows)!=2144:raise ValueError('All2144 required')
    result=dict(rows=rows,requested=2144,complete=True,all_raw_PCM_checks_pass=all(r['raw_peak_pass'] and r['nonempty'] for r in rows),native_training_approved=False,independent_ASR_semantics_timing_still_required=True)
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
