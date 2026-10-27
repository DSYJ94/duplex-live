"""Complete83 timing sources plus complete85 USER approvals after exact repairs."""
import argparse,gc,hashlib,json
from pathlib import Path
from types import SimpleNamespace
from . import asr
from .cadence_audit import word_error
from .pacing import units

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def recognize(folder,destination,ids=None):
    options=SimpleNamespace(audio=str(folder),model='openai/whisper-large-v3-turbo',output=str(destination),ids_json=str(ids) if ids else None)
    try:asr.main(options)
    except RuntimeError:
        if not (destination/'quality.json').exists():raise
    return json.loads((destination/'quality.json').read_text())

def normalized(parent,timing=False):
    row=dict(parent,original_reported_wer=parent['word_error_rate']);row['word_error_rate']=word_error(row['authored'],row['recognized'])
    row['passed']=row['word_error_rate']<=.05 and row['peak']<=.995
    if timing:
        try:row['native_speech_units']=units(row['authored'],row['words']);row['exact_word_timing_passed']=True
        except (ValueError,TypeError) as error:row['exact_word_timing_passed']=False;row['timing_error']=str(error)
        row['passed']=row['passed'] and row['exact_word_timing_passed']
    return row

def main(a):
    import torch
    out=Path(a.output);out.mkdir(exist_ok=False);assistant=Path(a.assistant);user=Path(a.user)
    parent_path=Path(a.prior_user_quality);parent=json.loads(parent_path.read_text());prior={r['id']:r for r in parent['rows']}
    ar=json.loads((user/'render-receipts.json').read_text());assert ar['complete'] and len(ar['utterances'])==len(prior)==85
    changed=[];reused={}
    for row in ar['utterances']:
        old=prior[row['id']];assert old['authored']==row['text'] and sha(user/row['wav'])==row['sha256']
        if old['passed'] and old['source_sha256']==row['sha256']:reused[row['id']]=old
        else:
            assert not old['passed'] and old['source_sha256']!=row['sha256'],'Only diagnosed failed USER recordings may change'
            changed.append(row['id'])
    assert len(changed)==17 and len(reused)==68
    subset=out/'diagnosed-repaired-user-ids.json';subset.write_text(json.dumps(dict(ids=changed))+'\n')
    aq=recognize(assistant,out/'assistant-raw');assert aq['requested']==len(aq['rows'])==83
    arows=[normalized(r,True) for r in aq['rows']]
    assistant_report=dict(aq,rows=arows,all_passed=all(r['passed'] for r in arows),passed=sum(r['passed'] for r in arows),raw_quality_sha256=sha(out/'assistant-raw/quality.json'),timestamp_generation='booleanTrue with pipeline word postprocessing',normalizer='Fixed cardinal0..100 and existing spelling conventions; no phonetic substitutions')
    (out/'assistant-quality.json').write_text(json.dumps(assistant_report,indent=2)+'\n')
    gc.collect();torch.cuda.empty_cache()
    uq=recognize(user,out/'repaired-user-raw',subset);assert uq['requested']==len(uq['rows'])==17
    observed={r['id']:normalized(r) for r in uq['rows']};rows=[]
    for source in ar['utterances']:
        row=observed.get(source['id'],reused.get(source['id']));assert row and row['source_sha256']==source['sha256']
        rows.append(row)
    user_report=dict(all_passed=all(r['passed'] for r in rows),requested=85,passed=sum(r['passed'] for r in rows),rows=rows,
        audio_receipts_sha256=sha(user/'render-receipts.json'),reused_complete_unchanged_source_records=68,new_recognition_records=17,parent_quality_sha256=sha(parent_path),repair_quality_sha256=sha(out/'repaired-user-raw/quality.json'),wer_threshold=.05,peak_threshold=.995,subset_acceptance=False,training_approved=False)
    (out/'user-quality.json').write_text(json.dumps(user_report,indent=2)+'\n')
    result=dict(complete=True,all_passed=assistant_report['all_passed'] and user_report['all_passed'],assistant=dict(required=83,passed=assistant_report['passed']),user=dict(required=85,passed=user_report['passed']),new_recognitions=100,reused_unchanged_word_records=68,optimizer_updates=0)
    (out/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['all_passed']:raise RuntimeError('Full168 source/time audit failed; no training approval')

if __name__=='__main__':
    p=argparse.ArgumentParser();[p.add_argument('--'+x,required=True) for x in ('assistant','user','prior-user-quality','output')];main(p.parse_args())
