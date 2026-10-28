"""Choose original or adapted Venus for one isolated microphone conversation."""
from contextlib import nullcontext
from training.venus.lora import disabled


def variant_context(model, variant):
    if variant == 'released':
        return disabled(model)
    if variant == 'candidate':
        return nullcontext()
    raise ValueError('Choose the original Venus or experimental trained candidate')


def variant_profile(profile, variant):
    if variant not in ('released', 'candidate'):
        raise ValueError('Unknown live model variant')
    result = dict(profile, live_variant=variant)
    if variant == 'released':
        result.pop('candidate', None)
        result.update(optimizer_updates=0, adaptation_enabled=False)
    else:
        result['adaptation_enabled'] = bool(profile.get('candidate'))
    return result
