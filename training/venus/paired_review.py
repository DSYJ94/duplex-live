"""Review all64 saved native cases offline; no generation or auto-adoption."""
import argparse,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
from data.venus.receipts import lookup_key


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def review(root):
    root=Path(root);speech=read(root/'paired-native-report.json');results=read(root/'results.json')
    assert speech['complete'] and speech['conditions'] in (['parent406','paced-continuation'],['main','control'])
    assert len(speech['cases'])==len(results['cases'])==64
    scenes={s['id']:s for p in ('evals/fixtures/venus-control-repair-source-v2-20261027.json','evals/fixtures/venus-control-corpus-v4-20261026.json') for s in read(p)['scenes']}
    rows=[];summary={};identities=None
    for condition in speech['conditions']:
        cases=[x for x in results['cases'] if x['condition']==condition]
        audits={x['id']:x for x in speech['cases'] if x['condition']==condition}
        assert len(cases)==len(audits)==32 and len({x['id'] for x in cases})==32
        current={x['id']:x['input_sha256'] for x in cases}
        if identities is None:identities=current
        else:assert identities==current
        group=[]
        for x in cases:
            a=audits[x['id']];folder=root/condition/x['id']
            assert sha(folder/'joined-emitted.wav')==a['joined_wav_sha256']
            for chunk in a['source_chunks']:assert sha(folder/chunk['file'])==chunk['sha256']
            traces=[json.loads(s) for s in (folder/'trace.jsonl').read_text().splitlines()]
            assert len(traces)==x['input_frames'] and all(t['sequence']==i for i,t in enumerate(traces))
            ev=x['actual_worker_events'];outputs=[e['result'] for e in ev if e['type']=='worker_completed']
            row=dict(condition=condition,id=x['id'],family=x['family'],evaluation_set=x['evaluation_set'],generated_text=a['references']['intended_user_facing'],asr_text=a['asr_text'],word_pass=a['word_pass'],word_error_rate=a['user_facing_text_wer'],peak_pass=a['raw_peak_pass'],queries=[e['query'] for e in ev if e['type']=='native_delegate'],worker_results=outputs,genuine_pending_at_followup=x['genuine_pending_at_followup'],genuine_cancel=any(e['type']=='worker_cancelled' for e in ev) and any(e['type']=='cancellation_confirmed' and e.get('targets') for e in ev),qualified=False)
            if x['family'] in ('delegate','pending','cancel'):
                scene=scenes[x['id']];key=lookup_key(scene['teacher_program']['query'])
                row['correct_lookup_completed']=('Lookup '+key+': '+scene['synthetic_private_world'][key]) in outputs
            if x['family']=='quiet':row.update(early_voiced_frames=x['voiced_frames_before_cue_word'],postcue_voiced_frames=x['voiced_frames_after_cue_word'])
            group.append(row)
        quiet=[x for x in group if x['family']=='quiet'];assert len(quiet)==4
        summary[condition]=dict(cases=32,word_audio_agreement=sum(x['word_pass'] for x in group),peak_pass=sum(x['peak_pass'] for x in group),quiet_early_cases=sum(x['early_voiced_frames']>0 for x in quiet),quiet_necessary_noearly_plus_resume=sum(x['early_voiced_frames']==0 and x['postcue_voiced_frames']>0 for x in quiet),correct_lookup_completions={f:sum(x.get('correct_lookup_completed',False) for x in group if x['family']==f) for f in ('delegate','pending')},genuine_cancellations=sum(x['genuine_cancel'] for x in group if x['family']=='cancel'),genuine_pending_cases=sum(x['genuine_pending_at_followup'] is True for x in group if x['family']=='pending'),native_delegation_requests=sum(len(x['queries']) for x in group))
        rows.extend(group)
    return dict(status='COMPLETE64_DIAGNOSTIC_REJECTED_FOR_ADOPTION',recorded_utc=datetime.now(timezone.utc).isoformat(),conditions=summary,cases=rows,complete64=True,matched_input_hashes=True,all_referenced_audio_hashes_verified=True,source_report_sha256=sha(root/'paired-native-report.json'),source_results_sha256=sha(root/'results.json'),qualified=False,adopted=False,limitations=['Fixed previously inspected regression cohort; one seed. Not fresh generalization.','Correct lookup completion is a necessary backend condition, not proof of spoken fact delivery.','No-early-audio plus voiced-after-cue is necessary, not semantic pause/resume approval.','ASR/text agreement is not task correctness or independent human naturalness/accent approval.','The new general-context reasoning transport was not enabled; these tasks use the original lookup worker.','No causal conclusion about irreparable base-model capacity, dataset quality or ideal optimizer recipe from this comparison.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();d=review(a.root);Path(a.output).write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['conditions'],indent=2))
