"""Couple complete native and qualified language epochs to one update cursor.

This grants no admission and performs no learning. The native executor consumes
the resulting language window in the same OPAL step as retained/speech losses.
"""
from . import expanded_schedule, language_schedule


def build(groups, speech_rows, documents, spec, *, epochs, language_budget,
          coefficient, tokenizer_sha256, qualification_sha256, seed):
    speech = expanded_schedule.build(groups, speech_rows, world_size=4, epochs=epochs)
    language = language_schedule.build(documents, spec,
        tokenizer_sha256=tokenizer_sha256, qualification_sha256=qualification_sha256,
        world_size=4, epochs=epochs, max_global_target_tokens=language_budget,
        coefficient=coefficient, seed=seed, updates_per_epoch=speech['steps_per_epoch'])
    body = dict(schema='native-mixed-complete-epochs-v1', speech=speech, language=language)
    return dict(body, sha256=language_schedule.digest(body))


def next_window(plan, documents, spec, *, rank, restored_step, identity, device='cpu'):
    """Select only the restored next update; never wrap or create a new cursor."""
    body = {k: v for k, v in plan.items() if k != 'sha256'}
    if (language_schedule.digest(body) != plan['sha256']
            or identity.get('mixed_schedule_sha256') != plan['sha256']
            or identity.get('language_training') != plan['language']['identity']):
        raise ValueError('Exact checkpoint-bound mixed schedule required')
    if type(rank) is not int or not 0 <= rank < 4 or type(restored_step) is not int:
        raise ValueError('Exact native rank and restored integer update required')
    speech = [row for epoch in plan['speech']['schedule'] for row in epoch]
    if len(speech) != len(plan['language']['updates']) or not 0 <= restored_step < len(speech):
        raise ValueError('Unfinished complete paired schedule required; no reset or replay')
    row = speech[restored_step]
    language = language_schedule.window(plan['language'], documents, spec,
        rank=rank, next_update=restored_step, device=device)
    if language['epoch'] != row['epoch']:
        raise ValueError('Native and language epoch boundaries must coincide')
    return dict(step=restored_step, retained=row['retained'],
        expanded_coefficient=row['new_coefficient'],
        speech_slots=[slot[rank] for slot in row['microbatches']],
        language_window=language,
        epoch_boundary=(restored_step + 1) % plan['speech']['steps_per_epoch'] == 0)


def update_next(runtime, base, opal, *, plan, documents, spec, step, identity,
                retained_groups, references, load_speech, admission,
                checkpoint_path, log_path):
    """Use one existing native update/OPAL owner for all three objectives.

    Caller must first prove fresh native/Adam/RNG/OPAL restoration and data
    admission. The underlying executor checks these before any backward call.
    """
    import torch.distributed as dist
    from .distributed_train import update
    if not dist.is_initialized() or dist.get_world_size() != 4:
        raise ValueError('Four-rank native collective required')
    window = next_window(plan, documents, spec, rank=dist.get_rank(),
        restored_step=step, identity=identity,
        device=next(iter(runtime.parameters())).device)
    slots = [dict(row=load_speech(slot['id']), valid=slot['valid'])
             for slot in window['speech_slots']]
    return update(runtime, base, opal, step=step, retained_groups=retained_groups,
        references=references, exposures=window['retained'], expanded_row=None,
        valid=True, expanded_coefficient=window['expanded_coefficient'],
        identity=identity, checkpoint_path=checkpoint_path, log_path=log_path,
        expanded_microbatches=slots, admission=admission,
        language_window=window['language_window'])
