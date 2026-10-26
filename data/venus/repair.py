"""Versioned exact-word repair, preserving all original recordings and audits."""
import argparse
import array
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import wave
from .asr import words as original_words


def words(text):
    # Same spoken words, two written conventions. Never equate homophones.
    text=text.lower().replace('’',"'")
    text=text.replace('acknowledgements','acknowledgments').replace('acknowledgement','acknowledgment')
    tokens=original_words(text)
    names='zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split()
    return [names[int(t)] if t.isdigit() and int(t)<len(names) else t for t in tokens]


def wer(reference,hypothesis):
    a,b=words(reference),words(hypothesis);row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(row[j]+1,nxt[j-1]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]/max(1,len(a))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(a):
    original=Path(a.original);parent=json.loads((original/'render-receipts.json').read_text());observed=json.loads(Path(a.observed).read_text())
    if isinstance(observed,dict):observed=observed['rows']
    checked={r['id']:r for r in observed};assert parent['complete']
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    selected=[r for r in parent['utterances'] if r['id'] in checked and wer(r['text'],checked[r['id']]['recognized'])>.05]
    assert selected,'Only actually diagnosed lexical failures may be re-recorded'
    for row in parent['utterances']:
        for ext in ('.wav','.aiff'):shutil.copy2(original/(row['id']+ext),out/(row['id']+ext))
    def one(row):
        row=dict(row);ident=row['id'];old_sha=row['sha256']
        subprocess.run(['say','-v',a.voice,'-r',str(a.rate),'-o',str(out/(ident+'.aiff')),row['text']],check=True,capture_output=True,timeout=60)
        subprocess.run(['/usr/bin/afconvert','-f','WAVE','-d','LEI16@16000','-c','1',str(out/(ident+'.aiff')),str(out/(ident+'.wav'))],check=True,capture_output=True,timeout=30)
        with wave.open(str(out/(ident+'.wav'))) as w:
            assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(16000,1,2)
            duration=w.getnframes()/16000;pcm=array.array('h',w.readframes(w.getnframes()))
        assert pcm and duration>0
        peak=max(abs(x) for x in pcm)/32768;assert peak<=.995,'No gain reduction or clipping concealment'
        row.update(seconds=duration,sha256=sha(out/(ident+'.wav')),aiff_sha256=sha(out/(ident+'.aiff')),peak=peak,voice='macOS '+a.voice,rate=a.rate,parent_wav_sha256=old_sha,repair_reason=checked[ident]['recognized'],independent_word_audit='PENDING')
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:repairs={r['id']:r for r in pool.map(one,selected)}
    rows=[repairs.get(r['id'],r) for r in parent['utterances']]
    receipt=dict(plan_sha256=parent['plan_sha256'],parent_audio_receipts_sha256=sha(original/'render-receipts.json'),failed_audit_sha256=sha(a.observed),complete=True,exact_words_preserved=True,unchanged_originals=len(rows)-len(repairs),repaired=len(repairs),utterances=rows)
    (out/'render-receipts.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (out/'repair-selection.json').write_text(json.dumps(dict(ids=list(repairs),threshold=.05,normalizer='v2 orthography-only: acknowledgment/acknowledgement,0..20digit/word',originals_preserved=True),indent=2)+'\n')
    print(json.dumps(dict(repaired=len(repairs),unchanged=len(rows)-len(repairs),all_recordings=len(rows),exact_words=True)))


def merge(a):
    parent=json.loads(Path(a.original_quality).read_text());new=json.loads(Path(a.repair_quality).read_text());audio=Path(a.audio);ar=json.loads((audio/'render-receipts.json').read_text())
    assert parent['requested']==159 and len(parent['rows'])==159
    rows={r['id']:r for r in parent['rows']}
    assert all(r['id'] in rows for r in new['rows'])
    rows.update({r['id']:r for r in new['rows']});result=[]
    for source in ar['utterances']:
        r=dict(rows[source['id']]);assert r['source_sha256']==source['sha256']==sha(audio/(source['id']+'.wav'))
        assert r['authored']==source['text']
        r['original_reported_wer']=r['word_error_rate'];r['word_error_rate']=wer(r['authored'],r['recognized']);r['passed']=r['word_error_rate']<=.05 and r['peak']<=.995
        result.append(r)
    assert len(result)==len({r['id'] for r in result})==159
    report=dict(all_passed=all(r['passed'] for r in result),requested=159,passed=sum(r['passed'] for r in result),rows=result,recognizer=parent['recognizer'],recognizer_revision=parent['recognizer_revision'],audio_receipts_sha256=sha(audio/'render-receipts.json'),wer_threshold=.05,peak_threshold=.995,subset_acceptance=False,normalizer='v2 orthography only',parent_quality_sha256=sha(a.original_quality),repair_quality_sha256=sha(a.repair_quality))
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(all_passed=report['all_passed'],passed=report['passed'],total=159,failed=[{k:r[k] for k in ('authored','recognized','word_error_rate')} for r in result if not r['passed']])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['render','merge']);p.add_argument('--original');p.add_argument('--observed');p.add_argument('--voice',default='Daniel (Enhanced)');p.add_argument('--rate',type=int,default=150);p.add_argument('--original-quality');p.add_argument('--repair-quality');p.add_argument('--audio');p.add_argument('--output',required=True);a=p.parse_args();dict(render=render,merge=merge)[a.stage](a)
