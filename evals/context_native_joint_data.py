"""Approve assistant audio targets beside the existing native teacher text.

All original approvals remain mandatory. Only the user conditioning is replaced
by its independently recorded streaming encoding. No words, audio codes, result
arrival, target frame or source fold are invented by this loader.
"""
import hashlib
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def digest(tensor):
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def load(args, spm, voice):
    import torch
    from contextspan.model.sequence_convention import persona_prefix, assemble_training_sequence
    from contextspan.moshi.models.lm import _delay_sequence
    from evals.lychee_context import BASE_INSTRUCTIONS
    from evals.context_training_data import load_training
    from evals.context_native_teacher_data import load as approved_native
    from evals.context_observed_teacher import canonical
    from evals.context_retention_teacher import select_teachers

    parent = json.loads((args.parent / 'report.json').read_text())
    if (not parent['passed'] or parent['optimizer_steps'] != 520
            or parent['checkpoint_sha256'] != sha256(args.parent / 'adapters.safetensors')
            or parent['voice_sha256'] != sha256(args.parent / 'voice.pt')):
        raise ValueError('Requires the exact previously restored native520 parent')
    policy = parent['retention_teacher_text']
    if policy['coverage'] != 'all-approved':
        raise ValueError('Requires the complete independent native teacher corpus')
    prefix = persona_prefix(BASE_INSTRUCTIONS, voice, spm)
    examples, _ = load_training(args.cache, spm, prefix,
        text_padding='legacy-epad0', request_policy='native-prelead-epad')
    manifest, approved = approved_native(args.approval, args.scope, args.assets,
        args.parent / 'voice.pt', examples, prefix, spm)
    encoded_manifest = json.dumps(manifest, indent=2) + '\n'
    gate_file = args.numerical / 'native-teacher-math-audit.json'
    gate = json.loads(gate_file.read_text())
    if (not gate['passed'] or gate['optimizer_steps'] != 0 or len(approved) != 25
            or hashlib.sha256(encoded_manifest.encode()).hexdigest() != policy['native_teacher_data_sha256']
            or gate['data_sha256'] != policy['native_teacher_data_sha256']
            or sha256(gate_file) != policy['numerical_preflight_sha256']
            or [x['teacher'] for x in gate['cases']] != [x['teacher'] for x in approved]
            or any(not x['passed'] or x['causal_prefix_maximum_error'] != 0. for x in gate['cases'])):
        raise ValueError('Complete unchanged native temporal numerical approval is required')

    selected = {x['record']['id']: canonical(x, spm) for x in select_teachers(examples, True)}
    initial = torch.full((1, 17, 1), 2048, dtype=torch.int64)
    initial[:, 0] = 32000
    batches, records = [], []
    for row, source in zip(approved, manifest['cases']):
        e = selected[row['teacher']['id']]
        saved = torch.load(Path(source['capture']) / 'teacher-engine-inputs.pt',
            map_location='cpu', weights_only=True)
        clean = e['clean_codes'].clone()
        # Source assistant targets stay bit-for-bit fixed, including genuine silence.
        clean[9:17] = saved['encoded_user_codes'][0]
        assembled, tm, am = assemble_training_sequence(clean,
            torch.ones(clean.shape[1], dtype=torch.bool), e['record']['spans'], spm)
        codes = torch.cat([prefix, assembled], dim=1)[None]
        tm = torch.cat([torch.zeros(prefix.shape[1], dtype=torch.bool), tm])[None]
        am = torch.cat([torch.zeros(prefix.shape[1], dtype=torch.bool), am])[None]
        delayed = torch.cat([initial, _delay_sequence(
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], codes, initial)], dim=2)[:, :, :-1]
        if (not torch.equal(delayed, row['inputs'])
                or not torch.equal(codes[:, :9], e['codes'][None, :9])
                or not torch.equal(tm[0], row['mask'])
                or not torch.equal(codes[0, 0], row['targets'])
                or not torch.equal(am[0], e['audio_mask'])
                or tm[:, :prefix.shape[1]].any() or am[:, :prefix.shape[1]].any()
                or (am & ~tm).any()):
            raise ValueError('Native inputs, audio ownership or existing text targets changed')
        targets = codes[:, 1:9].transpose(1, 2)[am]
        if (targets < 0).any() or (targets >= 2048).any() or not am.any():
            raise ValueError('Unapproved or missing real assistant codec targets')
        record = dict(teacher=row['teacher'], source_capture=source['capture'],
            native_input_sha256=digest(delayed), codes_sha256=digest(codes),
            text_mask_sha256=digest(tm), audio_mask_sha256=digest(am),
            assistant_targets_sha256=digest(targets),
            assistant_audio_targets_per_head=int(am.sum()), user_audio_loss_targets=0,
            prefix_loss_targets=0, supplied_span_loss_targets=0)
        batches.append(dict(codes=codes, text_mask=tm, audio_mask=am,
            delayed_inputs=delayed, record=record))
        records.append(record)
    proof = dict(schema='dl-native-assistant-joint-data-v1', cases=records,
        native_teacher_manifest=manifest, parent_report_sha256=sha256(args.parent / 'report.json'),
        native_temporal_approval_sha256=sha256(gate_file), loader_sha256=sha256(Path(__file__)),
        assistant_targets_from_approved_external_teacher=True,
        all25_native_inputs_exact=True, heldout_labels=0, owner_data_used=False,
        user_audio_loss_targets=0, prefix_loss_targets=0, supplied_span_loss_targets=0)
    return proof, batches
