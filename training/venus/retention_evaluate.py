"""Complete trained-arm regression evaluation, preserving actual restore evidence."""
import argparse,asyncio,json,subprocess
from pathlib import Path
from .isambard_native import BASE,guarded
from .large_inputs import sha

TRAINING_SOURCE='017578a72cbd8f8008a8c574436625633413597c'
TRAINING_JOB='8179044'
TRAINING_ROOT=BASE/'native-retention96-learning-v1'
original_training_source=TRAINING_SOURCE


def completed_training(root,arm):
    root=Path(root);execution=root/('execution-'+TRAINING_JOB)
    host=json.loads((execution/'terminal.json').read_text())
    stage=json.loads((execution/(arm+'-restore-terminal.json')).read_text())
    terminal=json.loads((root/arm/'training-terminal.json').read_text())
    reference=json.loads((root/arm/'final-native-reference.json').read_text())
    restore=json.loads((root/arm/'final-native-restore.json').read_text())
    parent=json.loads((root/arm/'parent-restore.json').read_text())
    if (host['actual_exit_code']!=0 or host['job_id']!=TRAINING_JOB or host['source']!=TRAINING_SOURCE
            or host['stage']!='retention-train' or stage['actual_exit_code']!=0 or stage['source']!=TRAINING_SOURCE
            or not terminal['completed'] or terminal['actual_new_updates']!=84
            or terminal['total_optimizer_updates']!=730 or terminal['TRAIN_exposures']!=336 or terminal['DEV']!=64
            or terminal['arm']!=arm or not parent['passed'] or not parent['Adam_RNG_OPAL_exact'] or not parent['native_logits_exact']
            or parent['source']!=TRAINING_SOURCE or parent['checkpoint_sha256']!='46e84c6d7b1d02ec16563c1d0e666230f8644963d3406b0f095bab97938950aa'):
        raise ValueError('Actual whole84/336training and original646 restore required')
    if (not restore['passed'] or restore['source']!=TRAINING_SOURCE or reference['source']!=TRAINING_SOURCE
            or restore['new_optimizer_updates']!=0 or not all(restore[k] for k in ('actual_CUDA','Adam_RNG_OPAL_exact','native_logits_exact'))
            or restore['training_execution_id']!=reference['training_execution_id']
            or restore['training_execution_id']==restore['restore_execution_id'] or stage['execution_id']!=restore['restore_execution_id']):
        raise ValueError('Successful distinct fresh final native restore required')
    digest=sha(root/arm/'step-0084.pt')
    if digest!=reference['checkpoint_sha256'] or digest!=restore['checkpoint_sha256']:raise ValueError('Actual immutable trained state differs')
    return dict(checkpoint_sha256=digest,training_source=TRAINING_SOURCE,arm=arm,optimizer_updates=730,
        final_restore_sha256=sha(root/arm/'final-native-restore.json'),adopted=False)


def admit(a):
    from portable.venus.native_app import ensure_guard
    ensure_guard();guarded(a.lease,600)
    if (Path(a.root).resolve()!=TRAINING_ROOT.resolve() or a.arm not in ('expanded-data','rehearsal')
            or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=a.source):
        raise ValueError('Source-bound exact trained arm required')
    return completed_training(TRAINING_ROOT,a.arm)


def main():
    from .large_behavior import parser,collect
    import sys
    p=parser();p.add_argument('--opal-source')
    # This separate parser extends choices without changing older admitted entrypoints.
    for action in p._actions:
        if action.dest=='arm':action.choices=('expanded-data','rehearsal')
    a=p.parse_args();asyncio.run(collect(a,execution_context=sys.modules[__name__]))


if __name__=='__main__':main()
