"""All168 new source audits, one recognizer at a time; no subset approval."""
import argparse,gc,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
from . import asr
from .pacing import words,units

def word_error(reference,recognized):
    a,b=words(reference),words(recognized);row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1):new.append(min(row[j]+1,new[j-1]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]/max(1,len(a))

def main(a):
    import torch
    from portable.venus.native_app import ensure_guard
    ensure_guard();stop=json.loads(Path(a.lease).read_text())['stop_unix'];assert stop-time.time()>360
    out=Path(a.output);out.mkdir(exist_ok=False);reports={}
    for role,folder,count in [('assistant',a.assistant,83),('user',a.user,85)]:
        ensure_guard();assert stop-time.time()>210
        destination=out/(role+'-raw');options=SimpleNamespace(audio=folder,model='openai/whisper-large-v3-turbo',output=str(destination),ids_json=None)
        try:asr.main(options)
        except RuntimeError:
            if not (destination/'quality.json').exists():raise
        raw=destination/'quality.json';quality=json.loads(raw.read_text());assert quality['requested']==count and len(quality['rows'])==count
        rows=[]
        for parent in quality['rows']:
            row=dict(parent,original_reported_wer=parent['word_error_rate']);row['word_error_rate']=word_error(row['authored'],row['recognized'])
            row['passed']=row['word_error_rate']<=.05 and row['peak']<=.995
            if role=='assistant':
                try:row['native_speech_units']=units(row['authored'],row['words']);row['exact_word_timing_passed']=True
                except (ValueError,TypeError) as error:row['exact_word_timing_passed']=False;row['timing_error']=str(error)
                row['passed']=row['passed'] and row['exact_word_timing_passed']
            rows.append(row)
        report=dict(quality,rows=rows,all_passed=all(r['passed'] for r in rows),passed=sum(r['passed'] for r in rows),parent_quality_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
            normalizer='Predeclared new-source cadence orthography: existing spelling rules plus cardinal digits0..100; no phonetic substitutions',training_approved=False)
        (out/(role+'-quality.json')).write_text(json.dumps(report,indent=2)+'\n');reports[role]=dict(requested=count,passed=report['passed'],all_passed=report['all_passed'])
        gc.collect();torch.cuda.empty_cache()
    complete=dict(complete=True,reports=reports,all_passed=all(r['all_passed'] for r in reports.values()),optimizer_updates=0,external_api_calls=0)
    (out/'source-audit.json').write_text(json.dumps(complete,indent=2)+'\n')
    if not complete['all_passed']:raise RuntimeError('Complete source audit failed; repair only diagnosed recordings in a new version')

if __name__=='__main__':
    p=argparse.ArgumentParser();[p.add_argument('--'+x,required=True) for x in ('assistant','user','output','lease')];main(p.parse_args())
