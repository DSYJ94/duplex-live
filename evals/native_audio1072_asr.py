"""Complete independent speech audit; no student weights, repairs or approval weakening."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    for key in ['plan','audio','inventory','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text());inventory=json.loads(a.inventory.read_text())
    if not inventory['complete'] or len(inventory['rows'])!=2144:raise ValueError('Whole2144 original source inventory required')
    by={r['id']:r for r in inventory['rows']}
    import soundfile as sf
    from training.venus.voice_compare import recognizer
    from data.venus.pacing import words
    from data.venus.repair import wer
    pipe=recognizer();a.out.mkdir(parents=True,exist_ok=True);results=[]
    for source in plan['exact_speech_payloads']:
        row=by[source['id']];path=a.audio/row['file'];dest=a.out/(source['id']+'.json')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Source waveform changed')
        if dest.exists():
            value=json.loads(dest.read_text())
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
            with dest.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
        results.append(value)
        print(json.dumps(dict(completed=len(results),required=2144)),flush=True)
    report=dict(complete=len(results)==2144,requested=2144,passed=sum(r['word_pass'] and r['peak_pass'] for r in results),all_passed=all(r['word_pass'] and r['peak_pass'] for r in results),rows=results,native_training_approved=False,independent_semantic_and_native_timing_approval_still_required=True)
    with (a.out/'report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')


if __name__=='__main__':main()
