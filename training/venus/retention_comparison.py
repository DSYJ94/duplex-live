"""Compare every native regression case; factual screens remain reviewable."""
import json,re
from pathlib import Path
from .large_inputs import sha
from data.venus.receipts import lookup_key
from .native_product_review import build as runtime_review
from .retention_evaluate import TRAINING_ROOT,completed_training

BASELINE=Path('/projects/u6ada/duplex-live/native-posttrain646-v1/native/main')
BASELINE_SOURCE='98db2c68c30376910c02b6ee54f84d6907f25249'


def factual_screen(ident,text):
    text=text.lower();terms={
      '0093':r'\b(tight|taut)\b','0095':r'\b(length|distance|width)\b',
      '0094':r'\b(sixty|60)\b','0092':r'\b(fly|flying|flight|glide)\b',
      '0220':r'\bdeep\b','0219':r'\b(light|transparent|see)\b',
      '0223':r'\b(fasten|secure|hold|tie|ties|tied)\b','0217':r'\b(twelve|12)\b',
      '0222':r'\b(guide|guides|pour|pouring|transfer|direct|directs)\b',
      '0221':r'\b(three|3)\b','0216':r'\b(distant|far|stars|space)\b',
      '0218':r'\b(pivot|fulcrum|lift|move|leverage)\b'}
    key=ident.rsplit('-',1)[-1]
    if key not in terms:return None
    return bool(re.search(terms[key],text)) and not (key=='0222' and re.search(r'\bfilter\w*\b',text))


def summarize(root,source):
    root=Path(root);report=json.loads((root/'native-report.json').read_text());results=json.loads((root/'results.json').read_text())['cases']
    if not report['complete'] or report['source']!=source or len(results)!=32 or len(report['cases'])!=32:
        raise ValueError('Complete actual32native and speech records required')
    by={r['id']:r for r in results};rows=[];scenes={}
    for p in ('evals/fixtures/venus-control-repair-source-v2-20261027.json','evals/fixtures/venus-control-corpus-v4-20261026.json'):
        scenes.update({s['id']:s for s in json.loads(Path(p).read_text())['scenes']})
    for speech in report['cases']:
        result=by[speech['id']];folder=root/speech['id']
        if sha(folder/'joined-emitted.wav')!=speech['joined_wav_sha256']:raise ValueError('Original waveform differs')
        for chunk in speech['source_chunks']:
            if Path(chunk['file']).name!=chunk['file'] or sha(folder/chunk['file'])!=chunk['sha256']:raise ValueError('Raw emitted chunk differs')
        text=speech['references']['intended_user_facing'];events=result['actual_worker_events']
        row=dict(id=result['id'],family=result['family'],input_sha256=result['input_sha256'],word_pass=speech['word_pass'],peak_pass=speech['raw_peak_pass'],text=text,recognized=speech['asr_text'],genuine_pending_at_followup=result['genuine_pending_at_followup'])
        if result['family']=='ordinary':row.update(text_factual_rule_screen=factual_screen(result['id'],text),spoken_factual_rule_screen=factual_screen(result['id'],speech['asr_text']))
        if result['family'] in ('delegate','pending','cancel'):
            scene=scenes[result['id']];expected=lookup_key(scene['teacher_program']['query'])
            requests=[e for e in events if e['type']=='native_delegate' and not str(e['query']).startswith('cancel')]
            def key(q):
                try:return lookup_key(q)
                except ValueError:return None
            correct={e['work_id'] for e in requests if key(e['query'])==expected}
            cancelled={e['work_id'] for e in events if e['type']=='worker_cancelled'}
            row.update(first_requested_key_correct=bool(requests) and key(requests[0]['query'])==expected,
                correct_requested_lookup_completed=any(e['type']=='worker_completed' and e['result']=='Lookup '+expected+': '+scene['synthetic_private_world'][expected] for e in events),
                genuine_cancel=bool(cancelled) and any(e['type']=='cancellation_confirmed' and e.get('targets') for e in events),
                correct_requested_work_cancelled=bool(correct&cancelled),actual_queries=[e['query'] for e in requests])
        if result['family']=='quiet':row['necessary_silence_and_resume']=result['voiced_frames_before_cue_word']==0 and result['voiced_frames_after_cue_word']>0
        rows.append(row)
    if len(by)!=32 or {r['id'] for r in rows}!=set(by):raise ValueError('Every distinct case required')
    summary=dict(cases=32,word_pass=sum(r['word_pass'] for r in rows),peak_pass=sum(r['peak_pass'] for r in rows),word_and_peak_pass=sum(r['word_pass'] and r['peak_pass'] for r in rows))
    summary['families']={f:dict(cases=len(v),word_pass=sum(r['word_pass'] for r in v),peak_pass=sum(r['peak_pass'] for r in v),**{k:sum(r.get(k) is True for r in v) for k in ('text_factual_rule_screen','spoken_factual_rule_screen','first_requested_key_correct','correct_requested_lookup_completed','genuine_cancel','correct_requested_work_cancelled','necessary_silence_and_resume')}) for f in sorted({r['family'] for r in rows}) for v in [[r for r in rows if r['family']==f]]}
    return dict(summary=summary,cases=rows,report_sha256=sha(root/'native-report.json'),results_sha256=sha(root/'results.json'))


def main(a):
    frozen=json.loads(Path('docs/NATIVE-RETENTION96-EVALUATION-INPUTS-2026-11-07.json').read_text())
    for name,digest in frozen['baseline_files'].items():
        if sha(BASELINE/name)!=digest:raise ValueError('Frozen parent646 baseline changed')
    parent_dev_path=Path('/projects/u6ada/duplex-live/native-training-resume8-v1/main/dev-0120.json')
    if sha(parent_dev_path)!=frozen['parent_DEV_sha256']:raise ValueError('Original full64 DEV baseline changed')
    parent_dev=json.loads(parent_dev_path.read_text())
    if parent_dev['scenes']!=64:raise ValueError('All64 original DEV required')
    collections={'parent646':(BASELINE,BASELINE_SOURCE)}
    for arm in ('expanded-data','rehearsal'):
        completed_training(TRAINING_ROOT,arm);collections[arm]=(TRAINING_ROOT/'native'/arm,a.source)
    conditions={k:summarize(root,source) for k,(root,source) in collections.items()}
    identities=[{r['id']:r['input_sha256'] for r in x['cases']} for x in conditions.values()]
    if not identities[0]==identities[1]==identities[2]:raise ValueError('All32 input hashes must match acrossall3')
    profiles=[json.loads((root/'profile.json').read_text()) for root,_ in collections.values()]
    fields=('seed','torch','cuda','precision','attention','input_sample_rate','output_sample_rate','system_prompt','decode_mode','max_new_speak_tokens_per_chunk','temperature','top_k','top_p','listen_prob_scale','tts_attention','voice_reference','nonspoken_action_channel','speech_completion','force_listen_count','listen_speak_mode')
    if any(any(p[k]!=profiles[0][k] for k in fields) for p in profiles[1:]):raise ValueError('Same native speech profile required')
    for arm in ('expanded-data','rehearsal'):
        runtime=runtime_review(TRAINING_ROOT/'native'/arm)
        (TRAINING_ROOT/'native'/arm/'runtime-review.json').write_text(json.dumps(runtime,indent=2)+'\n')
    development={'parent646':parent_dev}
    for arm in ('expanded-data','rehearsal'):
        dev=json.loads((TRAINING_ROOT/arm/'dev-0084.json').read_text())
        if dev['scenes']!=64 or set(dev['ce'])!=set(parent_dev['ce']):raise ValueError('Same full64 DEV families required')
        development[arm]=dict(report=dev,ce_change_fraction={k:(dev['ce'][k]-parent_dev['ce'][k])/parent_dev['ce'][k] for k in dev['ce']})
    report=dict(development=development,complete3_times32=True,source=a.source,conditions=conditions,all32_input_hashes_match=True,
        original_baseline_reused=True,raw_audio_unchanged=True,thresholds_unchanged=True,
        factual_rule_screens_require_full_answer_inspection=True,naturalness_and_accent_require_listening=True,
        regression_cohort_previously_inspected=True,fresh_text_reasoning_report=str(TRAINING_ROOT/'reasoning128/comparison.json'),
        native_general_reasoning_backend_not_executed=True,adopted=False)
    (TRAINING_ROOT/'native/retention96-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v['summary'] for k,v in conditions.items()}),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);main(p.parse_args())
