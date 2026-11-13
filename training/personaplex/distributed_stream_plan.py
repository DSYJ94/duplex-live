"""Align collective call counts without changing real recorded stream boundaries."""
import copy


def align(calls_by_rank):
    if not calls_by_rank or any(not calls for calls in calls_by_rank):
        raise ValueError('Every participating rank needs one recorded stream')
    for calls in calls_by_rank:
        end = 0
        for call in calls:
            if (call['start'] != end or type(call['frames']) is not int or call['frames'] < 1
                    or type(call['active_targets']) is not int or not 0 <= call['active_targets'] <= call['frames']):
                raise ValueError('Contiguous original calls and exact target counts required')
            if call.get('phase') in ('prefix', 'span') and call['active_targets']:
                raise ValueError('Provided history cannot acquire a training target')
            end += call['frames']
    slots = []
    for index in range(max(map(len, calls_by_rank))):
        ranks = []
        for calls in calls_by_rank:
            actual = copy.deepcopy(calls[index]) if index < len(calls) else None
            ranks.append(dict(original_call=actual, padding_only=actual is None,
                              active_targets=actual['active_targets'] if actual else 0))
        slots.append(dict(index=index, forward_on_all_ranks=True,
                          backward_on_all_ranks=any(r['active_targets'] for r in ranks), ranks=ranks))
    return dict(slots=slots, real_recorded_boundaries_unchanged=True,
                padding_training_targets=0, native_execution_qualified=False)
