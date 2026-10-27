"""Reuse complete source-bound word approvals; never re-render approved inputs."""
import argparse,hashlib,json,os,shutil
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(a):
    plan_path=Path(a.plan);plan=json.loads(plan_path.read_text());needed={t for s in plan['scenes'] for t in s['user_turns']}
    assert plan['counts']==dict(TRAIN=216,DEV=52,HOLDOUT=24) and len(needed)==218
    sources=[]
    for audio_path,quality_path in [(a.existing,a.existing_quality),(a.new,a.new_quality)]:
        audio=Path(audio_path);qpath=Path(quality_path);ar=json.loads((audio/'render-receipts.json').read_text());q=json.loads(qpath.read_text())
        assert ar['complete'] and q['all_passed'] and q['passed']==q['requested']==len(ar['utterances'])
        assert q['audio_receipts_sha256']==sha(audio/'render-receipts.json')
        recognized={r['id']:r for r in q['rows']}
        for row in ar['utterances']:
            r=recognized[row['id']];assert r['passed'] and r['source_sha256']==row['sha256'] and r['authored']==row['text']
            assert row['peak']<=.995 and row['seconds']>0 and sha(audio/row['wav'])==row['sha256']
        sources.append((audio,ar,q,recognized,qpath))
    lookup={row['text']:(source,row) for source in sources for row in source[1]['utterances']}
    assert needed<=lookup.keys()
    out=Path(a.output);out.mkdir(exist_ok=False,parents=True);clips=[];checks=[];reused=0
    for text in sorted(needed):
        (folder,ar,q,recognized,qpath),row=lookup[text]
        for name in (row['wav'],row['original_aiff']):
            original=folder/name;dest=out/name
            try:os.link(original,dest)
            except OSError:shutil.copy2(original,dest)
        assert sha(out/row['wav'])==row['sha256'] and sha(out/row['original_aiff'])==row['aiff_sha256']
        clips.append(dict(row,origin_audio_receipts_sha256=sha(folder/'render-receipts.json'),origin_quality_sha256=sha(qpath),origin_root=str(folder)))
        checks.append(dict(recognized[row['id']],source_quality_sha256=sha(qpath)))
        reused+=folder==Path(a.existing)
    receipts=dict(complete=True,plan_sha256=sha(plan_path),utterances=clips,originals_preserved=True)
    (out/'render-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
    quality=dict(all_passed=True,requested=len(checks),passed=len(checks),rows=checks,audio_receipts_sha256=sha(out/'render-receipts.json'),
        approval_method='Reuse actual complete independent word records by exact authored text and waveform SHA; not a claim of fresh recognition of old audio',reused_existing_approved_rows=reused,newly_recognized_rows=len(checks)-reused,
        wer_threshold=.05,peak_threshold=.995,subset_acceptance=False,training_approved=False)
    (out/'word-quality-complete.json').write_text(json.dumps(quality,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();[p.add_argument('--'+x,required=True) for x in ('plan','existing','existing-quality','new','new-quality','output')];main(p.parse_args())
