"""Derived diagnostics from immutable full36 traces; never behavioral approval."""
import argparse,hashlib,json,math,re
from pathlib import Path


def cue_frame(turn,word_record):
    if turn['sha256']!=word_record['source_sha256']:raise ValueError('Cue timing must match exact approved audio')
    cue=re.findall(r'[a-z]+',turn['text'].lower())[0]
    first=word_record['words'][0];spoken=re.findall(r'[a-z]+',first['text'].lower())[0]
    assert spoken==cue and first['timestamp'][1] is not None
    # Input chunk f contains audio[f,f+1), consumed BEFORE output generation.
    end=turn['start_seconds']+first['timestamp'][1]
    return math.ceil(end)-1,end


def review(collection,timelines,plan_path,words_path):
    root=Path(collection);results=json.loads((root/'results.json').read_text());assert results['all36_complete'] and len(results['cases'])==36
    plan={c['id']:c for c in json.loads(Path(plan_path).read_text())['scenes']};words={r['id']:r for r in json.loads(Path(words_path).read_text())['rows']};rows=[]
    for case in results['cases']:
        scene=plan[case['id']];timeline=json.loads((Path(timelines)/case['id']/'timeline.json').read_text())
        traces=[json.loads(s) for s in (root/case['id']/'trace.jsonl').read_text().splitlines()]
        events=case['actual_worker_events'];row=dict(id=case['id'],family=case['family'],generated_text=''.join(t['text'] for t in traces),
            actual_queries=[e['query'] for e in events if e['type']=='native_delegate'],
            actual_results=[e['result'] for e in events if e['type']=='worker_completed'],
            genuine_cancellation=any(e['type']=='worker_cancelled' for e in events) and any(e['type']=='cancellation_confirmed' and e.get('targets') for e in events),
            qualified=False)
        if case['family']=='quiet':
            turn=timeline['user_turns'][-1];eligible,end=cue_frame(turn,words[turn['audio_id']]);voiced=lambda t:t['audio_samples']>0 and t['peak']>.001
            row.update(cue_word_end_seconds=end,first_chunk_that_has_heard_cue=eligible,
                voiced_frames_before_cue_word=sum(voiced(t) and t['sequence']<eligible for t in traces),
                voiced_frames_after_cue_word=sum(voiced(t) and t['sequence']>=eligible for t in traces),
                generated_text_after_cue=''.join(t['text'] for t in traces if t['sequence']>=eligible))
        if case['family'] in ('delegate','pending','cancel'):
            key=scene['teacher_program']['query'].split()[-1].strip('.').lower();value=scene['synthetic_private_world'][key]
            row['correct_requested_lookup_completed']=any(e['type']=='worker_completed' and e['result']==f'Lookup {key}: {value}' for e in events)
        rows.append(row)
    quiet=[r for r in rows if r['family']=='quiet'];delegate=[r for r in rows if r['family']=='delegate'];cancel=[r for r in rows if r['family']=='cancel']
    return dict(complete36=True,cases=rows,quiet_early_voiced_cases=sum(r['voiced_frames_before_cue_word']>0 for r in quiet),
        quiet_voiced_after_actual_cue_cases=sum(r['voiced_frames_after_cue_word']>0 for r in quiet),
        delegate_correct_lookup_results=sum(r['correct_requested_lookup_completed'] for r in delegate),
        cancellation_successes=sum(r['genuine_cancellation'] for r in cancel),
        original_collection_sha256=hashlib.sha256((root/'results.json').read_bytes()).hexdigest(),
        metric_correction='Legacy cue_end fields use end of the entire cue-plus-question utterance. This derived report uses the source-bound cue WORD timestamp. Original traces/results are preserved.',
        independent_spoken_and_semantic_review_required=True,qualified=False,adopted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();[p.add_argument('--'+x,required=True) for x in ('collection','timelines','plan','words','output')];a=p.parse_args()
    result=review(a.collection,a.timelines,a.plan,a.words);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
