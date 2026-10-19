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
    full24 = bool(getattr(args, 'native_v2_full24_per8', False))
    rate = getattr(args, 'native_v2_learning_rate', 1e-5)
    review = getattr(args, 'native_v2_extension_review', None)
    if not any(present):
        if full24 or rate != 1e-5 or review:
            raise ValueError('Full24 recipe options require the complete native-v2 approval chain')
        return False
    if not all(present):
        raise ValueError('Additional native-v2 supervision needs its complete cache/input/scope/numerical chain')
    if (not args.actual_temporal_fit or not args.actual_mixed_data or not args.warm_start_proof
            or args.actual_native_teacher_coverage != 'all-approved'
            or args.actual_retention_teacher_weight != 1.
            or args.actual_base_retention_weight != 1.
            or args.actual_sampling != 'balanced-kind'
            or args.actual_retention_sampling != 'balanced-kind'
            or args.actual_native_teacher_loss != 'balanced-spoken'
            or args.actual_response_first_word_weight != 16.
            or not args.actual_native_teacher_approval or not args.actual_native_teacher_scope
            or not args.actual_native_teacher_preflight
            or not getattr(args, 'actual_native_teacher_cache', None)):
        raise ValueError('Native-v2 is a separate eight-update warm start with all old retention preserved')
    if rate not in (1e-5, 2e-6) or (not full24 and (rate != 1e-5 or review)):
        raise ValueError('A changed native learning rate needs the explicit full24 bounded recipe')
    if full24 and getattr(args, 'native_v2_gradient_audit', False):
        raise ValueError('The separate frozen-parent gradient diagnostic retains its original schedule policy')
    if args.resume_training or args.actual_fit_updates != 8 or review:
        if not (full24 and args.resume_training and review and args.actual_fit_updates == 32):
            raise ValueError('No native-v2 continuation without a complete committed native review')
        from evals.context_native_extension_review import validate
        validate(args)
    return True


def learning_rate(args):
    return getattr(args, 'native_v2_learning_rate', 1e-5) if getattr(args, 'native_v2_cache', None) else 1e-5


def validate_old25_lineage(original, repaired):
    """Only diagnosed timing and hash-bound approval fields may differ."""
    from evals.native_old25_timing_repair import REPAIRED
    policy_fields = {'teachers', 'native_input_approval_sha256', 'teacher_scope_sha256',
                     'numerical_preflight_sha256', 'native_teacher_data_sha256'}
    if ({k: v for k, v in original.items() if k not in policy_fields}
            != {k: v for k, v in repaired.items() if k not in policy_fields}
            or len(original['teachers']) != 25 or len(repaired['teachers']) != 25
            or [r['id'] for r in original['teachers']] != [r['id'] for r in repaired['teachers']]):
        raise ValueError('Repaired old25 policy changed coverage, weights, retention or objective')
    for before, after in zip(original['teachers'], repaired['teachers']):
        allowed = {'cache', 'cache_sha256', 'ingestion_sha256'}
        if before['id'] in REPAIRED:
            allowed |= {'codes_sha256', 'sidecar_sha256', 'targets_sha256'}
        if ({k: v for k, v in before.items() if k not in allowed}
                != {k: v for k, v in after.items() if k not in allowed}):
            raise ValueError('Old25 repair changed words/audio/masks/frames or undiagnosed targets: '+before['id'])


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
    if (parent['optimizer_steps'] != 520 or len(old_teacher_policy['teachers']) != 25
            or len(parent_data['components']) != 28
            or sha256(args.out / 'data.json') != parent['data_sha256']):
        raise ValueError('Native-v2 must retain the exact native520 parent corpus and all25 teachers')
    validate_old25_lineage(parent['retention_teacher_text'], old_teacher_policy)
    repaired_cache = validate_cache(args.actual_native_teacher_cache)
    if repaired_cache.get('schema') != 'dl-native-old25-timing-cache-v1':
        raise ValueError('Old25 timing repairs need their complete explicitly versioned native cache')
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
                  old_28_components_unchanged=True, old_25_words_audio_receipts_unchanged=True,
                  old_25_timing_repair_cache=str(args.actual_native_teacher_cache),
                  old_25_timing_repair_cache_sha256=sha256(args.actual_native_teacher_cache / 'cache.json'),
                  reference_kl_weight=1., reference_hidden_mse_weight=10., additional_teacher_weight=1.,
                  code_sha256=sha256(Path(__file__)),
                  objective='balanced assistant text, native request/control and genuine silence; no acoustic loss',
                  optimizer_reset=True, maximum_updates_before_behavioral_review=8,
                  prefix_loss_targets=0, span_loss_targets=0, user_loss_targets=0, audio_loss_targets=0)
    if getattr(args, 'native_v2_full24_per8', False):
        policy.update(complete24_per8=True, learning_rate=learning_rate(args),
            coverage_code_sha256=sha256(Path(__file__).with_name('context_native_teacher_coverage.py')),
            sampling='Average exact half of each new teacher kind per visit; all24 in every8 updates; unchanged old/reference sampling and total new kind weights',
            continuation='Only32 additional updates after independent full-state restore and committed complete10 native review with ordinary core retained; never adoption')
    path = separate.out / 'policy.json'
    if args.reload:
        if json.loads(path.read_text()) != policy:
            raise ValueError('Additional native-v2 objective changed at independent restore')
    else:
        path.write_text(json.dumps(policy, indent=2) + '\n')
    return policy, batches
