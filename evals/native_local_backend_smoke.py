"""Actual OpenAI worker lookup/cancellation over an authored read-only fixture.

Backend component evidence only: no native speech, learned routing, fresh-task
qualification or optimizer work. No owner records or conversations are loaded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from evals.native_control_teacher import atomic_json, collection_lock, local_key
from evals.recovery_worker import RecordWorld, run_worker


def run(out, env_file, lock):
    key = local_key(env_file)
    with collection_lock(lock):
        out.mkdir(parents=True, exist_ok=False)
        records = {'Falcon': {'access_code': 'rainy maple'}}
        world = RecordWorld(records)
        report = dict(schema='dl-local-real-backend-component-probe-v1',
                      started_utc=datetime.now(timezone.utc).isoformat(),
                      source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      worker_code_sha256=hashlib.sha256(Path('evals/recovery_worker.py').read_bytes()).hexdigest(),
                      model='gpt-6-luna', protocol='state-aware', actual_api=True,
                      response_executor_injected=False, record_store='authored-synthetic-read-only-fixture',
                      owner_conversations_or_records_used=False, status='running',
                      native_voice_or_routing_qualified=False, training_approved=False)
        target = out / 'report.json'
        atomic_json(target, report)
        previous = os.environ.get('OPENAI_API_KEY')
        os.environ['OPENAI_API_KEY'] = key
        try:
            context = [dict(role='user', text='Look up the access code for project Falcon.')]
            revision = world.reserve('synthetic-lookup', context)
            lookup = run_worker(context, world, 'synthetic-lookup', expected_revision=revision,
                                model=report['model'], dispatch_protocol='state-aware', scope_read_tools=True)
            atomic_json(out / 'lookup.json', lookup)
            if (lookup['route'] != 'delegate' or len(lookup['receipts']) != 1
                    or lookup['receipts'][0]['tool'] != 'lookup_record'
                    or lookup['receipts'][0]['value'] != records['Falcon']['access_code']
                    or not lookup['receipts'][0]['read_only']):
                raise RuntimeError('Actual lookup component failed its receipt contract')
            receipt = lookup['receipts'][0]
            world.mark_ready(receipt)
            report['lookup_before_cancellation_deliverable'] = world.deliverable(receipt)
            context = context + [dict(role='user', text='Cancel the project Falcon lookup before you speak its result.')]
            revision = world.reserve('synthetic-cancel', context)
            cancelled = run_worker(context, world, 'synthetic-cancel', expected_revision=revision,
                                   model=report['model'], dispatch_protocol='state-aware', scope_read_tools=True)
            atomic_json(out / 'cancellation.json', cancelled)
            if (cancelled['route'] != 'cancel' or not cancelled['receipts']
                    or world.deliverable(receipt)):
                raise RuntimeError('Actual cancellation or late-delivery guard failed')
            report.update(status='passed_backend_components_only',
                          lookup_after_cancellation_deliverable=False,
                          remote_execution_aborted=False,
                          actual_provider_responses=sum(len(x['events']) for x in (lookup, cancelled)))
        except Exception as exc:
            report.update(status='failed_preserved', error_type=type(exc).__name__)
            if hasattr(exc, 'result'):
                atomic_json(out / 'contract-failure.json', exc.result)
            raise
        finally:
            if previous is None:
                os.environ.pop('OPENAI_API_KEY', None)
            else:
                os.environ['OPENAI_API_KEY'] = previous
            report.update(finished_utc=datetime.now(timezone.utc).isoformat(), world_receipts=world.receipts,
                          world_tasks=world.tasks)
            atomic_json(target, report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--lock', type=Path,
                        default=Path('runs/local-migration-20261021/.private/real-backend-probe.lock'))
    args = parser.parse_args()
    report = run(args.out, args.env_file, args.lock)
    print(json.dumps({k: report[k] for k in ('status', 'actual_provider_responses', 'native_voice_or_routing_qualified')}))


if __name__ == '__main__':
    main()
