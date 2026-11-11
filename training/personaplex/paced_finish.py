"""Explicit final10 controller composition; immutable native/control bytes stay intact."""
import argparse
from contextlib import contextmanager
from . import matched_resume as resume, matched_finish as finish
from . import matched_evaluation_host as evaluation, matched_health as original
from . import paced_health_service as paced


@contextmanager
def controller_health():
    # Only controller-local dependencies change. The old modules/files and
    # separately launched immutable native worker are never patched on disk.
    if resume.matched_health is not original or finish.matched_health is not original:
        raise RuntimeError('Unexpected controller health dependency')
    resume.matched_health = paced
    finish.matched_health = paced
    try:
        yield
    finally:
        resume.matched_health = original
        finish.matched_health = original


def main(args):
    paced.authority()
    with controller_health():
        if args.mode == 'evaluate':
            evaluation.main(args)
        elif args.mode == 'train':
            finish.main(args)
        else:
            raise ValueError('Explicit train or evaluation-only mode required')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('train', 'evaluate'), required=True)
    parser.add_argument('--controller-source', required=True)
    for key in ('inputs', 'inputs-sha', 'previous-root', 'previous-result-sha',
                'training-root', 'phase-result-sha'):
        parser.add_argument('--'+key)
    args = parser.parse_args()
    required = ('training_root', 'phase_result_sha') if args.mode == 'evaluate' else (
        'inputs', 'inputs_sha', 'previous_root', 'previous_result_sha')
    if any(not getattr(args, key) for key in required):
        parser.error('All inputs for the selected mode are required')
    main(args)
