"""A separately approved complete native-v2 corpus beside unchanged old retention.

This adds native assistant text/control/silence supervision. It does not repeat
the rejected standalone audio pilot or alter its old 25-teacher guards.
"""
import copy
import json
from pathlib import Path
import shutil

from evals.moshi_model_download import sha256


def validate_teacher_onsets(teachers):
    """Fail before an optimizer if a claimed word predates any recorded PCM."""
    import numpy as np
    import soundfile as sf
    checked = []
    for teacher in teachers:
        cache = json.loads((Path(teacher['cache']) / 'cache.json').read_text())
        conversion = json.loads((Path(cache['conversion']) / 'ingestion.json').read_text())
        row = next(r for r in conversion['accepted'] if r['scenario_id'] == teacher['id'])
        wav = Path(row['path'])
        if sha256(wav) != row['wav_sha256']:
            raise ValueError('Native teacher waveform differs from its approved receipt')
        side = json.loads(wav.with_suffix('.json').read_text())
        pcm, rate = sf.read(wav, always_2d=True)
        nonzero = np.flatnonzero(pcm[:, 0])
        first = side['alignments'][0][1][0]
        if not len(nonzero) or first < nonzero[0]/rate-.2:
            raise ValueError('Native teacher has a physically impossible onset; retain original and reapprove a versioned timing repair: '+teacher['id'])
        checked.append(teacher['id'])
    return checked


def validate_configuration(args):
    names = ('native_v2_cache', 'native_v2_approval', 'native_v2_scope', 'native_v2_preflight')
    present = [bool(getattr(args, name, None)) for name in names]
    if not any(present):
        return False
    if not all(present):
        raise ValueError('Additional native-v2 supervision needs its complete cache/input/scope/numerical chain')
    if (not args.actual_temporal_fit or not args.actual_mixed_data or not args.warm_start_proof
            or args.resume_training or args.actual_fit_updates != 8
            or args.actual_native_teacher_coverage != 'all-approved'
            or args.actual_retention_teacher_weight != 1.
            or args.actual_base_retention_weight != 1.
            or args.actual_sampling != 'balanced-kind'
            or args.actual_retention_sampling != 'balanced-kind'
            or args.actual_native_teacher_loss != 'balanced-spoken'
            or args.actual_response_first_word_weight != 16.
            or not args.actual_native_teacher_approval or not args.actual_native_teacher_scope
            or not args.actual_native_teacher_preflight):
        raise ValueError('Native-v2 is a separate eight-update warm start with all old retention preserved')
    return True


def prepare(args, spm, voice, retention, old_teacher_policy):
    from contextspan.model.sequence_convention import persona_prefix
    from evals.lychee_context import BASE_INSTRUCTIONS
    from data.synthetic.native_curriculum_v2 import specifications
    from evals.recovery_data import validate_cache, validate_conversion
    from evals.context_training_data import load_training
    from evals.context_native_teacher_objective import prepare as approved_teacher

    validate_configuration(args)
    parent = json.loads((args.warm_start_proof / 'report.json').read_text())
    parent_data = json.loads((args.warm_start_proof / 'data.json').read_text())
    if (parent['optimizer_steps'] != 520 or parent['retention_teacher_text'] != old_teacher_policy
            or len(old_teacher_policy['teachers']) != 25
            or len(parent_data['components']) != 28
            or sha256(args.out / 'data.json') != parent['data_sha256']):
        raise ValueError('Native-v2 must retain the exact native520 parent corpus and old 25-teacher policy')
    validate_teacher_onsets(old_teacher_policy['teachers'])
    cache = validate_cache(args.native_v2_cache)
    conversion = validate_conversion(Path(cache['conversion']))
    expected = {'teacher-' + s['scenario_id'] for s in specifications() if s['split'] == 'train'}
    if (conversion.get('schema') != 'dl-teacher-native-v1' or not conversion.get('semantic_approval')
            or len(cache['examples']) != 24 or len(conversion['accepted']) != 24
            or {e['scenario_id'] for e in cache['examples']} != expected
            or any(e['split'] != 'train' for e in cache['examples'])):
        raise ValueError('Native-v2 requires every independently approved new TRAIN teacher, never a subset')
    examples, _ = load_training([args.native_v2_cache], spm, persona_prefix(BASE_INSTRUCTIONS, voice, spm),
                               text_padding='legacy-epad0', request_policy='native-prelead-epad')
    # Reuse the complete native numerical checks without overwriting old approval files.
    separate = copy.copy(args)
    separate.out = args.out / 'additional-native-v2'
    separate.actual_native_teacher_approval = args.native_v2_approval
    separate.actual_native_teacher_scope = args.native_v2_scope
    separate.actual_native_teacher_preflight = args.native_v2_preflight
    if not args.reload:
        separate.out.mkdir(exist_ok=False)
        shutil.copyfile(args.out / 'voice.pt', separate.out / 'voice.pt')
    elif sha256(separate.out / 'voice.pt') != sha256(args.out / 'voice.pt'):
        raise ValueError('Additional native-v2 voice changed at independent restore')
    policy, batches = approved_teacher(examples, spm, voice, retention, separate)
    if len(batches) != 24 or {b[3]['id'] for b in batches} != expected:
        raise ValueError('Native-v2 input/numerical approval omits a new TRAIN teacher')
    policy = dict(schema='dl-additional-native-v2-text-objective-v1', teacher=policy,
                  cache=str(args.native_v2_cache), cache_sha256=sha256(args.native_v2_cache / 'cache.json'),
                  conversion_sha256=sha256(Path(cache['conversion']) / 'ingestion.json'),
                  old_28_components_unchanged=True, old_25_teachers_unchanged=True,
                  reference_kl_weight=1., reference_hidden_mse_weight=10., additional_teacher_weight=1.,
                  code_sha256=sha256(Path(__file__)),
                  objective='balanced assistant text, native request/control and genuine silence; no acoustic loss',
                  optimizer_reset=True, maximum_updates_before_behavioral_review=8,
                  prefix_loss_targets=0, span_loss_targets=0, user_loss_targets=0, audio_loss_targets=0)
    path = separate.out / 'policy.json'
    if args.reload:
        if json.loads(path.read_text()) != policy:
            raise ValueError('Additional native-v2 objective changed at independent restore')
    else:
        path.write_text(json.dumps(policy, indent=2) + '\n')
    return policy, batches
