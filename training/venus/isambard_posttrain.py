"""Evaluate completed recovery120; reuse the original526 baseline honestly.

No optimizer execution, rewritten legacy proofs, or incomplete-cohort adoption.
The recovery's actual final restore is required before model loading.
"""
import asyncio
import json
import subprocess
from pathlib import Path

from .isambard_native import BASE, guarded
from .large_inputs import sha

TRAINING_ROOT = BASE / 'native-training-resume8-v1'
TRAINING_SOURCE = 'd8161fd40c80b814242f93c8d205cf33a83c4c54'
TRAINING_JOB = '8156150'
BASELINE_ROOT = BASE / 'native-evaluation-v5/native/main'
BASELINE_SOURCE = '35d40392b641d31023723d87edb72bb224af37b8'
original_training_source = TRAINING_SOURCE


def completed_training(root):
    """Bind actual successful host stages, complete phase and immutable state."""
    root = Path(root)
    execution = root / ('execution-' + TRAINING_JOB)
    host = json.loads((execution / 'terminal.json').read_text())
    stage = json.loads((execution / 'final-restore-terminal.json').read_text())
    terminal = json.loads((root / 'main/training-terminal.json').read_text())
    reference = json.loads((root / 'main/final-native-reference.json').read_text())
    restore = json.loads((root / 'main/final-native-restore.json').read_text())
    if (host['actual_exit_code'] != 0 or host['job_id'] != TRAINING_JOB
            or host['stage'] != 'train' or host['source'] != TRAINING_SOURCE
            or stage['actual_exit_code'] != 0 or stage['source'] != TRAINING_SOURCE):
        raise ValueError('Actual successful recovery and independent final stage required')
    if (terminal.get('completed') is not True or terminal['actual_new_updates'] != 120
            or terminal['restored_saved_updates'] != 8 or terminal['new_updates_this_job'] != 112
            or terminal['total_optimizer_updates'] != 646 or terminal['complete_epochs'] != 2
            or (terminal['TRAIN'], terminal['DEV']) != (240, 64)):
        raise ValueError('Whole original120-update phase required, never a subset')
    if (restore.get('passed') is not True or restore['source'] != TRAINING_SOURCE
            or reference['source'] != TRAINING_SOURCE or restore['new_optimizer_updates'] != 0
            or not all(restore[k] for k in ('actual_CUDA', 'Adam_RNG_OPAL_exact', 'native_logits_exact'))
            or restore['training_execution_id'] != reference['training_execution_id']
            or restore['training_execution_id'] == restore['restore_execution_id']
            or stage['execution_id'] != restore['restore_execution_id']):
        raise ValueError('Actual distinct complete-state restore required')
    digest = sha(root / 'main/step-0120.pt')
    if digest != restore['checkpoint_sha256'] or digest != reference['checkpoint_sha256']:
        raise ValueError('Immutable completed checkpoint differs')
    return dict(checkpoint_sha256=digest, training_source=TRAINING_SOURCE,
                training_job=TRAINING_JOB, new_optimizer_updates=0,
                final_restore_sha256=sha(root / 'main/final-native-restore.json'))


def admit(a):
    from .isambard_guard import resource
    from portable.venus.native_app import ensure_guard
    lease = resource(json.loads(Path(a.lease).read_text()))
    ensure_guard()
    if (a.source != subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            or lease['source'] != a.source or a.arm != 'main'):
        raise ValueError('Source-bound post-training main evaluation required')
    if (Path(a.root) / 'main').resolve() != (TRAINING_ROOT / 'main').resolve():
        raise ValueError('Read the exact retained recovery state, not a relabelled copy')
    return completed_training(TRAINING_ROOT)


def compare(root, source):
    root = Path(root)
    old = json.loads((BASELINE_ROOT / 'native-report.json').read_text())
    new = json.loads((root / 'native/main/native-report.json').read_text())
    old_cases = json.loads((BASELINE_ROOT / 'results.json').read_text())['cases']
    new_cases = json.loads((root / 'native/main/results.json').read_text())['cases']
    if (not old['complete'] or not new['complete'] or old['source'] != BASELINE_SOURCE
            or new['source'] != source or len(old['cases']) != 32 or len(new['cases']) != 32
            or len(old_cases) != 32 or len(new_cases) != 32):
        raise ValueError('Complete original526 and new646 cohorts required')
    identity = lambda rows: {r['id']: r['input_sha256'] for r in rows}
    if len(identity(old_cases)) != 32 or identity(old_cases) != identity(new_cases):
        raise ValueError('All32 paired input identities must match')
    for collection, report in ((BASELINE_ROOT, old), (root / 'native/main', new)):
        if {r['id'] for r in report['cases']} != set(identity(old_cases)):
            raise ValueError('All32 speech reviews must match')
        for row in report['cases']:
            if sha(collection / row['id'] / 'joined-emitted.wav') != row['joined_wav_sha256']:
                raise ValueError('Original raw audio differs')
    result = dict(complete=True, paired_scenes=32, outputs=64,
                  baseline_reused=True, baseline_global_update=526, candidate_global_update=646,
                  baseline_source=BASELINE_SOURCE, source=source,
                  baseline_report_sha256=sha(BASELINE_ROOT / 'native-report.json'),
                  candidate_report_sha256=sha(root / 'native/main/native-report.json'),
                  baseline_results_sha256=sha(BASELINE_ROOT / 'results.json'),
                  candidate_results_sha256=sha(root / 'native/main/results.json'),
                  matched_all32_input_hashes=True, baseline_rerun=False,
                  baseline_rows=old['cases'], candidate_rows=new['cases'],
                  new_optimizer_updates=0, qualified=False, adopted=False,
                  caveat='Inspected regression cohort; not fresh generalisation or product adoption.')
    (root / 'native/continuation-comparison.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    from .large_behavior import parser, collect
    p = parser()
    p.add_argument('--stage', choices=('collect', 'compare'), required=True)
    p.add_argument('--opal-source', required=True)
    a = p.parse_args()
    proof = admit(a)
    (Path(a.root) / 'completion-admission.json').write_text(json.dumps(proof, indent=2) + '\n')
    if a.stage == 'collect':
        import sys
        asyncio.run(collect(a, execution_context=sys.modules[__name__]))
    else:
        compare(a.root, a.source)


if __name__ == '__main__':
    main()
