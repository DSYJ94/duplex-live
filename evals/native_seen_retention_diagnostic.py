"""Unforced native comparison on the four ordinary TRAIN anchor questions."""
from pathlib import Path

SCENES=tuple('recovery-train-%03d'%i for i in range(8,12))
CHECKPOINTS=('7510331','7556562','7558824')


def case(index):
    if type(index) is not int or not 0<=index<12:raise ValueError('Complete twelve-case TRAIN comparison required')
    return dict(index=index,scenario=SCENES[index//3],checkpoint_job=CHECKPOINTS[index%3],
        role=('native520','clock8','clock40')[index%3],text_temperature=.7,acoustic_temperature=.8,
        diagnostic_only=True,diagnostic_outputs_must_not_be_trained_on=True,qualification_approved=False)


def policy(args):
    proof=Path(args.adapter_proof) if args.adapter_proof else None
    if (args.native_sampling_diagnostic or args.native_text_temperature!=.7
            or args.backend_mode!='observe-only' or args.expected_kind!='simple' or not args.train_history
            or proof is None or proof.parent!=Path('/projects/u6si/duplex-live/runs')
            or proof.name not in ['context-training-proof-'+j for j in CHECKPOINTS]
            or args.adapter_disabled or args.adapter_zero or args.warm_audio or args.ret_logit_bias
            or args.pending_status or args.persona_profile!='recorded-records' or args.lookup_delivery_delay_s
            or args.scenario not in SCENES or args.voice!='accepted-synthetic'
            or Path(args.data)!=Path('/projects/u6si/duplex-live/runs/recovery-data-7407554')):
        raise ValueError('Seen retention is a fixed TRAIN-only three-checkpoint diagnostic, never qualification or training data')
    return dict(mode='native-seen-train-retention-diagnostic-v1',text_temperature=.7,acoustic_temperature=.8,
        seed=42424242,source_words_audio_persona_voice_and_weights_unchanged=True,
        optimizer_steps=0,diagnostic_outputs_must_not_be_trained_on=True,
        native_quality_or_extension_approved=False,forced_reference_words=False,
        scope='same four approved ordinary TRAIN stimuli across native520/clock8/clock40; not generalization')
