"""Regrade complete immutable recognition responses with declared orthography.

No provider calls, changed transcripts, threshold changes or source substitutions.
"""
import argparse,hashlib,json
from pathlib import Path
from . import math_spoken_words


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    old=json.loads(a.report.read_text())
    if not old['complete'] or old['requested']!=2144 or len(old['rows'])!=2144 or len({r['id'] for r in old['rows']})!=2144:raise ValueError('Full immutable2144 recognition required')
    rows=[]
    for row in old['rows']:
        row=dict(row);row['WER']=math_spoken_words.wer(row['source_words'],row['recognized']);row['word_pass']=row['WER']<=.05;row['word_policy']='math-spoken-v3-typographic-apostrophe-speed-unit';rows.append(row)
    result=dict(old,rows=rows,passed=sum(r['word_pass'] and r['peak_pass'] for r in rows),all_passed=all(r['word_pass'] and r['peak_pass'] for r in rows),parent_report_sha256=hashlib.sha256(a.report.read_bytes()).hexdigest(),word_policy_source_sha256=hashlib.sha256(Path(math_spoken_words.__file__).read_bytes()).hexdigest(),provider_calls=0,transcripts_changed=False,threshold_changed=False)
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,(list,dict))}))


if __name__=='__main__':main()
