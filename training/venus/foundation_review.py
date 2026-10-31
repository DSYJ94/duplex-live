"""Review every base/adapted record offline; no model calls or auto-adoption."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from data.venus.receipts import lookup_key


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def review(base, adapted):
    first_condition=read(Path(base)/'profile.json')['condition']
    assert first_condition in ('untouched-release','private-action-repair406')
    roots={first_condition:Path(base),'candidate406':Path(adapted)}
    scenes={s['id']:s for p in ['evals/fixtures/venus-control-repair-source-v2-20261027.json','evals/fixtures/venus-control-corpus-v4-20261026.json'] for s in read(p)['scenes']}
    rows=[]; summaries={}; identities=None
    for condition,root in roots.items():
        original=condition==first_condition
        report=read(root/('foundation-report.json' if original else 'paired-native-report.json'))
        assert report['complete']
        selected=[r for r in read(root/'results.json')['cases'] if original or r['condition']=='paced-continuation']
        audio={r['id']:r for r in report['cases'] if original or r['condition']=='paced-continuation'}
        assert len(selected)==len(audio)==32 and len({x['id'] for x in selected})==32
        current={x['id']:x['input_sha256'] for x in selected}
        if identities is None:identities=current
        else:assert identities==current, 'Matched input cohort required'
        group=[]
        for x in selected:
            a=audio[x['id']];folder=root/x['id'] if original else root/'paced-continuation'/x['id']
            assert sha(folder/'joined-emitted.wav')==a['joined_wav_sha256']
            for chunk in a['source_chunks']:assert sha(folder/chunk['file'])==chunk['sha256']
            ev=x['actual_worker_events'];queries=[e['query'] for e in ev if e['type']=='native_delegate'];outputs=[e['result'] for e in ev if e['type']=='worker_completed']
            traces=[json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
            assert len(traces)==x['input_frames'] and all(t['sequence']==i for i,t in enumerate(traces))
            row=dict(condition=condition,id=x['id'],family=x['family'],evaluation_set=x['evaluation_set'],generated_text=a['references']['intended_user_facing'],asr_text=a['asr_text'],word_pass=a['word_pass'],word_error_rate=a['user_facing_text_wer'],peak_pass=a['raw_peak_pass'],queries=queries,worker_results=outputs,genuine_pending_at_followup=x['genuine_pending_at_followup'],genuine_cancel=any(e['type']=='worker_cancelled' for e in ev) and any(e['type']=='cancellation_confirmed' and e.get('targets') for e in ev),qualified=False)
            if x['family'] in ('delegate','pending','cancel'):
                scene=scenes[x['id']];key=lookup_key(scene['teacher_program']['query'])
                row['correct_lookup_completed']=('Lookup '+key+': '+scene['synthetic_private_world'][key]) in outputs
            if x['family']=='quiet':
                row.update(early_voiced_frames=x['voiced_frames_before_cue_word'],postcue_voiced_frames=x['voiced_frames_after_cue_word'])
            group.append(row)
        quiet=[x for x in group if x['family']=='quiet'];assert len(quiet)==4
        summaries[condition]=dict(cases=32,word_audio_agreement=sum(x['word_pass'] for x in group),peak_pass=sum(x['peak_pass'] for x in group),quiet_early_cases=sum(x['early_voiced_frames']>0 for x in quiet),quiet_necessary_noearly_plus_resume=sum(x['early_voiced_frames']==0 and x['postcue_voiced_frames']>0 for x in quiet),correct_lookup_completions={f:sum(x.get('correct_lookup_completed',False) for x in group if x['family']==f) for f in ('delegate','pending')},genuine_cancellations=sum(x['genuine_cancel'] for x in group if x['family']=='cancel'),genuine_pending_cases=sum(x['genuine_pending_at_followup'] is True for x in group if x['family']=='pending'),native_delegation_requests=sum(len(x['queries']) for x in group))
        rows.extend(group)
    interpretation=('Runtime repair leaves aggregate pass counts unchanged. Native speech and behavior remain rejected; sample-level audio changes must be reviewed separately. Shared RNG consumption changes, so identical subsequent model trajectories are not claimed.' if first_condition=='private-action-repair406' else 'Adaptation improves explicit control/delegation but reduces text/audio agreement on this fixed cohort. Neither condition is product-qualified. Shared failures do not isolate irreparable model capacity; verify the implicated native runtime contract next.')
    return dict(schema='dl-native-foundation64-review-v1',recorded_utc=datetime.now(timezone.utc).isoformat(),complete64=True,new_generations=32,reused_candidate406_cases=32,all_referenced_audio_hashes_verified=True,conditions=summaries,cases=rows,qualified=False,adopted=False,ordinary_semantics_and_naturalness_not_independently_approved=True,interpretation=interpretation,causal_limits='Same reference/sampling; fixed regression cohort, one seed. No causal proof of9Bcapacity limit, generalization, accent quality or backend reasoning capability.',speech_agreement_is_not_task_accuracy=True,quiet_necessary_condition_is_not_semantic_success=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('base','adapted','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();d=review(a.base,a.adapted);Path(a.output).write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['conditions'],indent=2))
