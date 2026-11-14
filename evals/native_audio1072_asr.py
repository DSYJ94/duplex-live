"""Complete independent speech audit; no student weights, repairs or approval weakening."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    for key in ['plan','audio','inventory','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--complete-timestamps',action='store_true');p.add_argument('--word-policy',choices=['legacy','math-spoken-v1'],default='legacy')
    a=p.parse_args();plan=json.loads(a.plan.read_text());inventory=json.loads(a.inventory.read_text())
    if not inventory['complete'] or len(inventory['rows'])!=2144:raise ValueError('Whole2144 original source inventory required')
    by={r['id']:r for r in inventory['rows']}
    import soundfile as sf
    from training.venus.voice_compare import recognizer
    from data.venus.pacing import words
    from data.venus.repair import wer
    selected_wer=None
    if a.word_policy=='math-spoken-v1':
        from data.synthetic.math_spoken_words import wer as selected_wer
    pipe=recognizer()
    policy='whisper_boolean_word_processor_v1' if a.complete_timestamps else 'legacy_unpatched'
    if a.complete_timestamps:
        from data.venus.asr import enable_complete_word_timestamps
        enable_complete_word_timestamps(pipe.model)
    a.out.mkdir(parents=True,exist_ok=True);results=[]
    for source in plan['exact_speech_payloads']:
        row=by[source['id']];path=a.audio/row['file'];dest=a.out/(source['id']+'.json')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Source waveform changed')
        if dest.exists():
            value=json.loads(dest.read_text())
            if value.get('word_policy','legacy')!=a.word_policy:raise ValueError('Existing word policy differs; use a new audit version')
            if value.get('timestamp_policy','legacy_unpatched')!=policy:raise ValueError('Existing audit timestamp policy differs; use a new version')
            if value['source_sha256']!=row['sha256']:raise ValueError('Existing audit source differs')
        else:
            audio,sr=sf.read(path,dtype='float32')
            if sr!=16000 or audio.ndim!=1:raise ValueError('Original16000Hz mono required')
            try:
                out=pipe(dict(array=audio,sampling_rate=sr),return_timestamps='word',generate_kwargs=dict(language='en',task='transcribe',do_sample=False))
                error=wer(' '.join(words(source['text'])),' '.join(words(out['text'])))
                value=dict(id=source['id'],fold=source['fold'],source_sha256=row['sha256'],source_words=source['text'],recognized=out['text'],words=out.get('chunks',[]),WER=error,word_pass=error<=.05,peak_pass=row['raw_peak_pass'],native_training_approved=False)
            except TypeError as exc:
                # Preserve decoder failures as failed records; never approve a subset.
                if "NoneType" not in str(exc):
                    raise
                value=dict(id=source['id'],fold=source['fold'],source_sha256=row['sha256'],source_words=source['text'],recognized=None,words=[],WER=None,word_pass=False,peak_pass=row['raw_peak_pass'],decoder_error=repr(exc),native_training_approved=False)
            spans=[w.get('timestamp') for w in value.get('words',[])]
            finite=bool(spans) and all(isinstance(t,(tuple,list)) and len(t)==2 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in t) and 0<=t[0]<=t[1] for t in spans)
            value['timing_pass']=finite and spans[-1][1]>spans[-1][0] and all(spans[i+1][0]>=spans[i][0] for i in range(len(spans)-1))
            if selected_wer is not None and value['recognized'] is not None:
                value['raw_WER']=value['WER'];value['WER']=selected_wer(source['text'],value['recognized']);value['word_pass']=value['WER']<=.05
            value['word_policy']=a.word_policy
            value['timestamp_policy']=policy
            with dest.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
        results.append(value)
        print(json.dumps(dict(completed=len(results),required=2144)),flush=True)
    report=dict(complete=len(results)==2144,requested=2144,passed=sum(r['word_pass'] and r['peak_pass'] for r in results),all_passed=all(r['word_pass'] and r['peak_pass'] for r in results),rows=results,native_training_approved=False,independent_semantic_and_native_timing_approval_still_required=True,timestamp_policy=policy,word_policy=a.word_policy,all_measured_word_timings_complete=all(r.get('timing_pass',False) for r in results))
    with (a.out/'report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')


if __name__=='__main__':main()
