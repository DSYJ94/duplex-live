"""Bounded alternative recordings for two fixed failed texts; require all85."""
import argparse,hashlib,json,shutil
from pathlib import Path
from types import SimpleNamespace
from . import asr
from .cadence_repair_audit import normalized

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(a):
    root=Path(a.candidates);out=Path(a.output);out.mkdir(exist_ok=False)
    candidates=json.loads((root/'render-receipts.json').read_text());assert candidates['complete'] and len(candidates['utterances'])==6
    parent_path=Path(a.prior_user_quality);parent=json.loads(parent_path.read_text());assert parent['requested']==85 and parent['passed']==83
    prior={r['id']:r for r in parent['rows']};failed={r['id'] for r in parent['rows'] if not r['passed']};assert len(failed)==2
    assert {r['source_original_id'] for r in candidates['utterances']}==failed
    raw=out/'candidate-recognition'
    try:asr.main(SimpleNamespace(audio=str(root),model='openai/whisper-large-v3-turbo',output=str(raw),ids_json=None))
    except RuntimeError:
        if not (raw/'quality.json').exists():raise
    q=json.loads((raw/'quality.json').read_text());assert q['requested']==len(q['rows'])==6
    recognized={r['id']:normalized(r) for r in q['rows']};selection={}
    for ident in failed:
        for variant in candidates['predeclared_variant_order']:
            entry=next(e for e in candidates['utterances'] if e['source_original_id']==ident and e['variant']==variant)
            row=recognized[entry['id']];assert row['source_sha256']==entry['sha256'] and row['authored']==prior[ident]['authored']
            if row['passed']:selection[ident]=(entry,row);break
    original=Path(a.user);ar=json.loads((original/'render-receipts.json').read_text());assert ar['complete'] and len(ar['utterances'])==85
    final=Path(a.final_audio);shutil.copytree(original,final);clips=[];rows=[]
    for entry in ar['utterances']:
        ident=entry['id'];assert sha(original/entry['wav'])==entry['sha256'] and prior[ident]['source_sha256']==entry['sha256']
        if ident in selection:
            chosen,row=selection[ident];shutil.copy2(root/chosen['wav'],final/(ident+'.wav'));shutil.copy2(root/chosen['original_aiff'],final/(ident+'.aiff'))
            entry=dict(chosen,id=ident,wav=ident+'.wav',original_aiff=ident+'.aiff',selected_candidate_id=chosen['id'])
            row=dict(row,id=ident,candidate_recognition_id=chosen['id'])
        else:row=prior[ident]
        assert row['source_sha256']==entry['sha256'] and sha(final/entry['wav'])==entry['sha256']
        clips.append(entry);rows.append(row)
    (final/'render-receipts.json').write_text(json.dumps(dict(ar,utterances=clips,parent_audio_receipts_sha256=sha(original/'render-receipts.json'),candidate_receipts_sha256=sha(root/'render-receipts.json'),selection_rule='Firstpassingvariant in predeclaredorder for each of the2diagnosedfixedtexts; no scene or word dropped'),indent=2)+'\n')
    report=dict(all_passed=len(rows)==85 and all(r['passed'] for r in rows),requested=85,passed=sum(r['passed'] for r in rows),rows=rows,audio_receipts_sha256=sha(final/'render-receipts.json'),parent_quality_sha256=sha(parent_path),candidate_quality_sha256=sha(raw/'quality.json'),reused_unchanged_source_records=83,new_candidate_recognitions=6,selected_repaired_texts=len(selection),wer_threshold=.05,peak_threshold=.995,subset_acceptance=False,training_approved=False)
    (out/'user-quality.json').write_text(json.dumps(report,indent=2)+'\n')
    aq=Path(a.assistant_quality);assistant=json.loads(aq.read_text());assert assistant['all_passed'] and assistant['passed']==assistant['requested']==83
    (out/'assistant-quality.json').write_bytes(aq.read_bytes())
    result=dict(complete=True,all_passed=report['all_passed'],assistant=dict(required=83,passed=83),user=dict(required=85,passed=report['passed']),new_recognitions=6,all_variant_results_retained=True,source_audio=str(final),optimizer_updates=0)
    (out/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['all_passed']:raise RuntimeError('Bounded source repair exhausted; complete85 gate still fails')

if __name__=='__main__':
    p=argparse.ArgumentParser();[p.add_argument('--'+x,required=True) for x in ('candidates','user','prior-user-quality','assistant-quality','final-audio','output')];main(p.parse_args())
