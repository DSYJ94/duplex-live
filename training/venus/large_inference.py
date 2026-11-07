"""Explicit experimental loader for the new capacity schema, never adoption."""
from .large_inputs import PARENT_SHA, MANIFEST_SHA
from .lora import install, load_adapter, ALL_PROJECTIONS
from .large_capacity import expand_broad

def load_continuation(llm,state,checkpoint_sha256):
    """Explicit new phase; never reinterpret the old466->526 phase as new work."""
    meta=state['metadata'];recipe=meta['recipe'];proof=meta['independent_parent_restore'];transition=meta['phase_transition']
    if (meta['capacity_schema']!='native-capacity-continuation-v1' or meta['arm']!='main' or meta['parent_training_steps']!=526
            or meta['parent_checkpoint_sha256']!='a89a39b7dd1e2a08f06ee2776d92adcd41e285245d6a91fd1d36d85297a42e81'
            or meta['cache_sha256']!=MANIFEST_SHA or meta['adopted'] or not 0<state['step']<=recipe['updates']<=240
            or recipe['updates']!=recipe['epochs']*60 or not proof['passed'] or not proof['actual_CUDA']
            or proof['pid']==proof['reference_pid'] or proof['source']!=meta['parent_evaluation_source']
            or meta['parent_evaluation_source']!='35d40392b641d31023723d87edb72bb224af37b8'
            or proof['checkpoint_sha256']!=meta['parent_checkpoint_sha256']
            or not transition['adam_moments_and_counters_exact'] or not transition['rng_exact']):raise ValueError('Exact independently restored526 continuation required')
    if any(recipe[k]!=v for k,v in dict(rank=256,alpha=256,last_layers=36,ordinary_kl_weight=.1,ordinary_hidden_weight=.01,action_aux_weight=.25,microbatch=1,accumulation=4,forward_path='native-segmented-bptt-v1').items()) or recipe['projections']!=list(ALL_PROJECTIONS):raise ValueError('Native architecture/objectives changed')
    install(llm,rank=64,alpha=64,last_layers=1)
    spec=expand_broad(llm,last_layers=36,rank=256,alpha=256,seed=20261104)
    if spec!=meta['expansion'] or spec['parameters']!=698351616:raise ValueError('Exact preserved broad adapter required')
    load_adapter(llm,state['adapter']);llm.requires_grad_(False)
    return dict(checkpoint_sha256=checkpoint_sha256,optimizer_updates=526+state['step'],current_round_updates=state['step'],parent_training_steps=526,planned_updates=526+recipe['updates'],adapter=spec,training_source=meta['source'],model_revision=meta['model_revision'],experimental_candidate=True,adopted=False,native_behavior_qualified=False,arm='main')


def load(llm, state, checkpoint_sha256):
    meta = state['metadata']; recipe = meta['recipe']; arm = meta['arm']
    proof = meta['independent_native_restore']
    if (meta['capacity_schema'] != 'native-capacity-trained-v1' or arm not in ('main','control')
            or meta['parent_training_steps'] != 466 or meta['parent_checkpoint_sha256'] != PARENT_SHA
            or meta['cache_sha256'] != MANIFEST_SHA or meta['adopted']
            or not 0 < state['step'] <= recipe['updates'] <= 480
            or recipe['updates'] != recipe['epochs']*60
            or not proof['passed'] or not proof['actual_CUDA'] or proof['new_optimizer_updates'] != 0
            or proof['initialization_pid'] == proof['restore_pid'] or proof['source'] != meta['source']
            or meta['expanded_init_checkpoint_sha256'] != proof['checkpoint_sha256']):
        raise ValueError('Exact independently restored capacity phase and parent required')
    if (recipe['ordinary_kl_weight'] != .1 or recipe['ordinary_hidden_weight'] != .01
            or recipe['action_aux_weight'] != .25 or recipe['microbatch'] != 1 or recipe['accumulation'] != 4):
        raise ValueError('Original loss and effective batch required')
    if arm == 'main':
        if (recipe['forward_path'] != 'native-segmented-bptt-v1' or recipe['last_layers'] != 36
                or recipe['rank'] != 256 or recipe['alpha'] != 256 or recipe['projections'] != list(ALL_PROJECTIONS)):
            raise ValueError('Exact broad native candidate required')
    elif (recipe['forward_path'] != 'native-frozen-prefix-v1' or recipe['last_layers'] != 1
            or recipe['rank'] != 64 or recipe['alpha'] != 64 or recipe['projections'] != ['q_proj','v_proj']):
        raise ValueError('Exact same-parent small-adapter control required')
    spec = install(llm, rank=64, alpha=64, last_layers=1)
    if arm == 'main':
        spec = expand_broad(llm, last_layers=36, rank=256, alpha=256, seed=20261104)
        if spec != meta['expansion'] or spec['parameters'] != 698351616:
            raise ValueError('Declared broad tensor geometry differs')
    load_adapter(llm, state['adapter']); llm.requires_grad_(False)
    return dict(checkpoint_sha256=checkpoint_sha256, optimizer_updates=466+state['step'],
                current_round_updates=state['step'], parent_training_steps=466,
                planned_updates=466+recipe['updates'], adapter=spec, training_source=meta['source'],
                model_revision=meta['model_revision'], experimental_candidate=True,
                adopted=False, native_behavior_qualified=False, arm=arm)


def validate_data_phase(state):
    from .retention_train import PARENT_SHA as parent_sha,PARENT_SOURCE
    meta=state['metadata'];recipe=meta['recipe'];transition=meta['phase_transition']
    if (meta['capacity_schema']!='native-retention96-data-v1' or meta['arm'] not in ('expanded-data','rehearsal')
            or meta['source']!='017578a72cbd8f8008a8c574436625633413597c'
            or meta['parent_training_steps']!=646 or meta['parent_checkpoint_sha256']!=parent_sha
            or meta['original_capacity_metadata']['source']!=PARENT_SOURCE or meta['adopted']
            or not 0<state['step']<=84 or recipe['updates']!=84 or recipe['epochs']!=1
            or meta['composite_manifest']!=dict(schema='dl-preserved304-plus-native96-v1',old_manifest_sha256=MANIFEST_SHA,
                new_manifest_sha256='c33f520d1fddc8778e8e8fb9e0f9f2cb7c978df4cdb194a78f48e49dafef6cd8',TRAIN=336,DEV=64)
            or not transition['adam_moments_and_counters_exact'] or not transition['rng_exact']):
        raise ValueError('Exact independently restored646 data comparison required')
    required=dict(rank=256,alpha=256,last_layers=36,ordinary_kl_weight=.1,ordinary_hidden_weight=.01,
        action_aux_weight=.25,microbatch=1,accumulation=4,forward_path='native-segmented-bptt-v1')
    if any(recipe[k]!=v for k,v in required.items()) or recipe['projections']!=list(ALL_PROJECTIONS):
        raise ValueError('Original native architecture/objectives required')
    return meta


def load_data_phase(llm,state,checkpoint_sha256):
    meta=validate_data_phase(state)
    install(llm,rank=64,alpha=64,last_layers=1)
    spec=expand_broad(llm,last_layers=36,rank=256,alpha=256,seed=20261104)
    if spec!=meta['expansion'] or spec['parameters']!=698351616:raise ValueError('Exact preserved broad adapter required')
    load_adapter(llm,state['adapter']);llm.requires_grad_(False)
    return dict(checkpoint_sha256=checkpoint_sha256,optimizer_updates=646+state['step'],current_round_updates=state['step'],
        parent_training_steps=646,planned_updates=730,adapter=spec,training_source=meta['source'],
        model_revision=meta['model_revision'],experimental_candidate=True,adopted=False,native_behavior_qualified=False,arm=meta['arm'])
