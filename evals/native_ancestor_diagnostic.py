"""Read-only three-way adapter comparison; original four ordinary inputs/profile."""
from pathlib import Path
ORDINARY=tuple('recovery-train-%03d'%i for i in range(12,16))
ROLES=('published-base','native520','audio49-eight')


def case(index):
    if type(index) is not int or not 0<=index<12:raise ValueError('Complete twelve-case ancestor comparison required')
    role=ROLES[index%3]
    return dict(index=index,scenario=ORDINARY[index//3],role=role,
        checkpoint_job='7575817' if role=='audio49-eight' else '7510331',adapter_disabled=role=='published-base',
        text_temperature=.7,acoustic_temperature=.8,seed=42424242,diagnostic_only=True,
        optimizer_steps=0,training_approved=False,diagnostic_outputs_must_not_be_trained_on=True)


def policy(args):
    row=case(args.native_ancestor_diagnostic_case)
    if (not args.native_ancestor_diagnostic or args.scenario!=row['scenario'] or args.expected_kind!='simple'
            or args.backend_mode!='observe-only' or not args.train_history or not args.adapter_proof
            or Path(args.adapter_proof).name!='context-training-proof-'+row['checkpoint_job']
            or args.adapter_disabled!=row['adapter_disabled'] or args.adapter_zero or args.native_text_temperature!=.7
            or args.warm_audio or args.ret_logit_bias or args.pending_status or args.persona_profile!='recorded-records'
            or args.lookup_delivery_delay_s or args.native_sampling_diagnostic or args.native_seen_retention_diagnostic
            or args.native_persona_diagnostic or getattr(args,'native_persona_diagnostic_case',None) is not None
            or getattr(args,'layout_audit',None) or getattr(args,'math_attention',False)
            or getattr(args,'backend_reference_profile','answer-only')!='answer-only'
            or getattr(args,'event_log_mode','immediate')!='immediate'):
        raise ValueError('Only adapter presence/weights may change in the locked original-profile ancestor comparison')
    return dict(row,mode='stock-native-three-ancestor-diagnostic-v1',source_words_audio_persona_voice_unchanged=True,
        native_quality_or_extension_approved=False,forced_reference_words=False)
