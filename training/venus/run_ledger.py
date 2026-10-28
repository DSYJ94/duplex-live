"""Index actual local run receipts, without treating copied checkpoints as runs.

Reads explicit event/recipe/terminal files only. Never scans credentials, owner
conversations or private directories. Status is receipt-derived, not a provider
health check; refresh live provider state separately before further execution.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOTS = ('runs/venus-training-20261026', 'runs/venus-training-20261027')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt_run(path):
    events = read(path)
    folder = path.parent
    starts = [e for e in events if e['event'] == 'actual_unit_started']
    finishes = {e['unit']: e for e in events if e['event'] == 'actual_unit_terminal'}
    stages = []
    for event in starts:
        terminal = finishes.get(event['unit'])
        stages.append(dict(unit=event['unit'], started_utc=event['at_utc'],
                           launch_identity=event.get('state'),
                           terminal_utc=terminal['at_utc'] if terminal else None,
                           terminal=terminal.get('state') if terminal else None))
    row = dict(id=folder.name, event_receipt=str(path), event_receipt_sha256=digest(path),
               first_recorded_utc=events[0]['at_utc'] if events else None,
               last_recorded_utc=events[-1]['at_utc'] if events else None,
               source_shas=sorted({e['source'] for e in events if isinstance(e.get('source'), str)}),
               stages=stages,
               actual_training_records=[{k:v for k,v in e.items() if k != 'error'}
                   for e in events if 'training_complete' in e['event']],
               admitted_recipes=[e for e in events if e['event'] == 'actual_recipe_admitted'],
               failures=[dict(at_utc=e['at_utc'], event=e['event'], error_type=e.get('error_type'))
                   for e in events if 'failure' in e['event'] or 'error' in e['event']],
               provider_stop_records=[e for e in events if e['event'] == 'provider_stopped'],
               adoption_inferred=False)
    # Raw exception strings may include command arguments; keep them out of Git.
    for name, key in (('copyback-verified.json', 'backup'), ('provider-final.json', 'provider_snapshot')):
        file = folder/name
        if file.exists():
            value = read(file)
            row[key] = ({k:value[k] for k in ('id','status','lastStartTimestamp','lastStopTimestamp') if k in value}
                        if key == 'provider_snapshot' else value)
            row[key+'_receipt'] = str(file)
    lease = folder/'lease.json'
    if lease.exists():
        value=read(lease)
        row['lease']={k:value[k] for k in ('resource_key','vm_id','disk_id','stop_unix',
            'prior_new_gpu_seconds_upper','prior_gpu_seconds','verified_hourly_price_usd',
            'authorization_sha256','execution_plan_sha256') if k in value}
        row['lease_receipt']=str(lease)
        row['lease_receipt_sha256']=digest(lease)
    return row


def checkpoint_record(root, file, name):
    recipe = read(root/'recipe.json')
    terminal = read(root/'training-terminal.json')
    return dict(name=name, checkpoint=str(root/file), checkpoint_sha256=digest(root/file),
                source=recipe['source'], recipe=recipe['recipe'], admission=recipe.get('admission'),
                cache_sha256=recipe['cache_sha256'], approval_sha256=recipe['approval_sha256'],
                model_revision=recipe['model_revision'],
                parent_checkpoint_sha256=recipe.get('parent_checkpoint_sha256'),
                actual_training_terminal=terminal, recipe_receipt=str(root/'recipe.json'),
                recipe_receipt_sha256=digest(root/'recipe.json'),
                terminal_receipt=str(root/'training-terminal.json'),
                adopted=False, compared_loss_requires_identical_data_and_targets=True)


def build():
    runs = [receipt_run(path) for root in ROOTS for path in sorted(Path(root).glob('*/events.json'))]
    checkpoints = [
        checkpoint_record(Path('runs/venus-training-20261027/paced-recovery-v25/copyback/candidate-v13'),
                          'step-0190.pt', 'venus-control190'),
        checkpoint_record(Path('runs/venus-training-20261027/paced-recovery-v25/candidate-early-backup'),
                          'step-0054.pt', 'venus-paced244'),
    ]
    return dict(schema='dl-native-experiment-ledger-v1', recorded_utc=datetime.now(timezone.utc).isoformat(),
                scope='All direct Venus controller events found under the two named local roots; not every historical Moshi/Slurm job',
                roots=list(ROOTS), run_count=len(runs), stage_count=sum(len(r['stages']) for r in runs),
                source_of_status='Saved receipts; not a live provider read',
                historical_moshi_reviews=['docs/PROJECT-RESET-2026-10-25.md',
                    'docs/GCP-NATIVE-SAVED8-COMPLETE20-REVIEW-2026-10-25.json'],
                runs=runs, unique_trained_checkpoints=checkpoints,
                latest_learning_review='docs/NATIVE-VENUS-REACTIVE54-RESULTS-2026-10-28.json',
                owner_voice_feedback='docs/NATIVE-OWNER-VOICE-FEEDBACK-2026-10-27.json',
                raw_private_feedback_included=False, live_conversations_included=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='docs/NATIVE-EXPERIMENT-INDEX-2026-10-28.json')
    args = parser.parse_args()
    result = build()
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(runs=result['run_count'], stages=result['stage_count'],
                         unique_checkpoints=len(result['unique_trained_checkpoints']), output=args.output)))
