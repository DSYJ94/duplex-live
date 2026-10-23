"""Read-only lifecycle/oracle diagnostics; never substitute for speech/semantic gates.

Scoring annotations are read only here, after actual teacher capture. They never
enter a voice/backend payload. Partial progress is explicitly unable to qualify.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

from data.synthetic.filter import norm_words
from data.synthetic.native_reasoning_delegation import NAMES
from evals import native_reasoning_teacher as teacher
from evals.native_reasoning_source_quality import verify as verify_source
from evals.native_reasoning_source import sha


def bundle_errors(content,target):
    tokens=norm_words(content);present=sorted(n for n,_ in NAMES if n in tokens);reasons=[]
    if present!=target['names']:reasons.append('Actual returned bundle names differ from independent oracle')
    for field in ('cost','points'):
        needle=norm_words(str(target[field]))
        if not any(tokens[i:i+len(needle)]==needle for i in range(len(tokens))):
            reasons.append('Actual returned bundle '+field+' differs from independent oracle')
    return reasons


def case_review(row,ledger):
    tasks=ledger['tasks'];ids=list(tasks);reasons=[];family=row['family']
    if family=='direct-basic':
        if ids:reasons.append('Basic-only scene unnecessarily requested external work')
    elif not ids:reasons.append('No native-selected reasoning request')
    else:
        original=ids[0];retired=[]
        if family=='delegate-reasoning':
            current=tasks[original]
            if current.get('receipt',{}).get('operation')!='respond_to_user' or 'wire_write_completed_at_s' not in current:
                reasons.append('Actual original reasoning result was not returned')
            else:reasons+=bundle_errors(current['receipt']['content'],row['annotations_not_model_inputs']['actual_worker_returns_required'][0]['oracle'])
        else:
            operation='cancel_reasoning' if family=='cancel-reasoning' else 'replace_reasoning'
            for identity,task in tasks.items():
                receipt=task.get('receipt',{})
                if receipt.get('operation')==operation and any(t['task_id']==original for t in receipt['retired_requests']):
                    retired.append((identity,task))
            if len(retired)!=1 or not tasks[original]['cancelled']:
                reasons.append('Missing actual committed '+operation+' for original request')
            else:
                identity,current=retired[0];receipt=current['receipt']
                if 'wire_write_completed_at_s' not in current:reasons.append('Committed current status/result never returned to voice teacher')
                if receipt['remote_execution_aborted'] is not False:reasons.append('Unproven remote execution abort was claimed')
                if family=='replace-reasoning':
                    if receipt['revision']!=2:reasons.append('Replacement did not advance the actual request revision')
                    reasons+=bundle_errors(receipt['content'],row['annotations_not_model_inputs']['actual_worker_returns_required'][-1]['oracle'])
            if 'wire_write_completed_at_s' in tasks[original]:
                reasons.append('Original result was delivered despite required pending cancellation/replacement')
    return dict(id=row['id'],family=family,native_requests=len(tasks),structural_pass=not reasons,reasons=reasons,
        independent_output_audio_and_spoken_semantics_still_required=True,
        task_states={k:dict(phase=v['phase'],cancelled=v['cancelled'],provider_finished_at_s=v.get('provider_finished_at_s'),
                           wire_write_completed_at_s=v.get('wire_write_completed_at_s'),operation=v.get('receipt',{}).get('operation')) for k,v in tasks.items()})


def review(source,quality,batch,engine,tokenizer_path,out):
    verify_source(source,quality);manifest,plan=teacher.source_manifest(source);tokenizer=teacher.geometry(engine,tokenizer_path)
    snapshot=(batch/'teacher-collection.json').read_bytes();collection=json.loads(snapshot)
    scenes={s['id']:s for s in manifest['scenes']};rows={r['id']:r for r in plan['cases']}
    results=[]
    for record in collection['recordings']:
        if record['transport_complete'] is not True:continue
        implementation=teacher
        if collection['schema']=='dl-native-reasoning32-teacher-recovery-v2':
            from evals.native_reasoning_teacher_recovery import recording_implementation
            implementation=recording_implementation(record)
        identity=record['id'];folder=batch/identity;implementation.verify_record(source,scenes[identity],folder,record,tokenizer)
        r=case_review(rows[identity],json.loads((folder/'lifecycle.json').read_text()));results.append(r)
    value=dict(schema='dl-native-reasoning32-actual-lifecycle-diagnostic-v1',at_utc=datetime.now(timezone.utc).isoformat(),
        teacher_collection_snapshot_sha256=hashlib.sha256(snapshot).hexdigest(),complete32=len(results)==32,
        recorded_case_count=len(collection['recordings']),reviewed_case_count=len(results),results=results,
        observed_structural_failures=sum(not r['structural_pass'] for r in results),
        diagnostic_only=True,passing_subset_accepted=False,teacher_quality_or_semantic_approved=False,training_approved=False,
        source_oracles_sent_to_any_model_by_this_review=False,API_or_GPU_calls=0)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise ValueError('Keep previous progress diagnoses immutable')
    with out.with_suffix('.collection-snapshot.json').open('xb') as file:file.write(snapshot)
    out.write_text(json.dumps(value,indent=2)+'\n');return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','quality','batch','engine','tokenizer','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();v=review(a.source,a.quality,a.batch,a.engine,a.tokenizer,a.out)
    print(json.dumps({k:v[k] for k in ('complete32','reviewed_case_count','observed_structural_failures','diagnostic_only')}))
