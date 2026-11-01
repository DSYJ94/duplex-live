"""Versioned whole-cohort orthography correction; raw audit remains failed.

Only trade-off/tradeoff spelling is added. No words, source recordings,
recognizer outputs, timestamps, thresholds or folds are rewritten.
"""
import argparse,hashlib,json,re
from pathlib import Path
from .context_source_quality import words as original_words,inventory


def words(text):return original_words(re.sub(r'\btrade-off\b','tradeoff',text,flags=re.I))

def wer(a,b):
    a,b=words(a),words(b);row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(row[j]+1,nxt[j-1]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]/max(1,len(a))


def main(a):
    if a.output.exists():raise ValueError('Preserve versioned derived evidence')
    raw=a.audit.read_bytes();report=json.loads(raw)
    if not report['word_audit_complete'] or len(report['rows'])!=72:raise ValueError('Full actual independent recognition required')
    local={r['id']:r for r in inventory(a.sources,a.payload)}
    rows=[]
    for original in report['rows']:
        source=local[original['id']]
        if original['sha256']!=source['sha256'] or original['text']!=source['text'] or original['fold']!=source['fold']:raise ValueError('Original source/fold changed')
        row=dict(original,original_word_error_rate=original['word_error_rate'],original_passed=original['passed'],word_error_rate=wer(original['text'],original['recognized']))
        row['passed']=row['word_error_rate']<=.05 and row['peak']<=.995
        rows.append(row)
    if len({r['id'] for r in rows})!=72:raise ValueError('Exact whole72 required')
    out=dict(schema='dl-context-source-derived-words-v2',original_audit_sha256=hashlib.sha256(raw).hexdigest(),original_audit_preserved_failed=True,original_passed=sum(x['original_passed'] for x in rows),required=72,passed=sum(x['passed'] for x in rows),all72_pass=all(x['passed'] for x in rows),rows=rows,wer_threshold=.05,peak_threshold=.995,procedure_change='Specific trade-off/tradeoff orthography equivalence, diagnosed AFTER original audit; no general hyphen deletion or semantic normalization.',fresh_recognition=False,recognizer=report['recognizer'],revision=report['revision'],source_words_audio_timestamps_folds_unchanged=True,training_approved=False,native_qualified=False,external_calls=0,optimizer_updates=0)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
    if not out['all72_pass']:raise RuntimeError('Whole cohort still fails')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--sources',type=Path,required=True);p.add_argument('--payload',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
