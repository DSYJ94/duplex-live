"""Read-only paired diagnostic of the stock native greedy text sampler."""
ORDINARY=tuple('recovery-train-%03d'%i for i in range(12,16))


def policy(args):
    enabled=args.native_sampling_diagnostic;temperature=args.native_text_temperature
    if not enabled:
        if temperature!=.7:raise ValueError('Nondefault decoding requires explicit diagnostic opt-in')
        return None
    if (temperature!=0. or args.backend_mode!='observe-only' or args.expected_kind!='simple'
            or not args.train_history or not args.adapter_proof or args.adapter_disabled or args.adapter_zero
            or args.warm_audio or args.ret_logit_bias or args.pending_status
            or args.persona_profile!='recorded-records' or args.lookup_delivery_delay_s
            or args.scenario not in ORDINARY):
        raise ValueError('Greedy text is a bounded paired ordinary diagnostic, never a training/adoption approval')
    return dict(mode='stock-native-greedy-text-diagnostic-v1',text_temperature=0.,acoustic_temperature=.8,
        seed=42424242,source_words_audio_persona_voice_and_weights_unchanged=True,
        optimizer_steps=0,diagnostic_outputs_must_not_be_trained_on=True,
        native_quality_or_extension_approved=False,forced_reference_words=False)


def case(index,candidate):
    if type(index) is not int or not 0<=index<8 or not str(candidate).isdigit():
        raise ValueError('All8 paired cases and one exact candidate checkpoint required')
    return dict(index=index,scenario=ORDINARY[index//2],role='baseline' if index%2==0 else 'candidate',
        checkpoint_job='7510331' if index%2==0 else str(candidate),text_temperature=0.,acoustic_temperature=.8,
        diagnostic_only=True,training_approved=False)


def verify_engine(engine,declared):
    if declared is None:return
    gen=engine.lm_gen
    if gen.temp_text!=0. or gen.temp!=.8 or not gen.use_sampling:
        raise RuntimeError('Actual native sampler differs from declared text-only diagnostic')
    declared.update(actual_lmgen_text_temperature=float(gen.temp_text),actual_lmgen_acoustic_temperature=float(gen.temp),
        native_acoustic_sampling_enabled=bool(gen.use_sampling))
