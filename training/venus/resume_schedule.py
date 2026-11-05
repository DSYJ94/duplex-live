"""Account for completed ordered work without changing data or timing margins."""
import math
import random

from .workload import epoch_seconds


def remaining_schedule(work, main_seconds, control, main_dev_seconds, saved_step):
    if (type(saved_step) is not int or not 0 < saved_step < 60
            or control['optimizer_updates'] != 0 or control['effective_batch'] != 4
            or any(not math.isfinite(x) or x <= 0 for x in
                   (main_seconds, control['seconds'], main_dev_seconds, control['full_dev_seconds']))):
        raise ValueError('Actual saved clock and original complete timing evidence required')
    full = epoch_seconds(main_seconds, control['seconds'], work)
    ordered = list(work['train_ids'])
    random.Random(20261104).shuffle(ordered)
    completed, remaining = ordered[:saved_step * 4], ordered[saved_step * 4:]
    if (len(completed) + len(remaining) != 240 or set(completed) & set(remaining)
            or set(completed + remaining) != set(work['train_ids'])):
        raise ValueError('Every original TRAIN scene must occur exactly once')
    remaining_main = (main_seconds * sum(work['weights'][i] for i in remaining)
                      / work['calibration_weight'] + len(remaining))
    required = ((remaining_main + full['control']) * 1.35
                + 2 * (main_dev_seconds + control['full_dev_seconds']) * 1.25 + 4200)
    return dict(saved_step=saved_step, completed_train_ids=completed,
                remaining_train_ids=remaining, complete_train_per_epoch=240,
                updates_each_arm=60, remaining_main_updates=60-saved_step,
                remaining_control_updates=60, full_dev_per_review=64,
                original_full_main_epoch_seconds=full['main'],
                remaining_main_epoch_seconds=remaining_main,
                complete_control_epoch_seconds=full['control'],
                joint_required_seconds=required, train_margin=1.35, dev_margin=1.25,
                final_reserve_seconds=4200, original_both_baselines_still_reserved=True,
                selection_uses_losses_or_heldout=False)
