"""A single inference-only scoping correction; original training persona is frozen."""
import json
from pathlib import Path
from evals.lychee_context import BASE_INSTRUCTIONS

PROFILE='records-request-only'
OLD_SUFFIX='Ask which project if the user has not identified it.'
NEW_SUFFIX='Ask which project only when the user requests a project record lookup without identifying the project.'


def scoped_persona():
    if not BASE_INSTRUCTIONS.endswith(OLD_SUFFIX) or BASE_INSTRUCTIONS.count(OLD_SUFFIX)!=1:
        raise ValueError('The original recorded persona changed')
    return BASE_INSTRUCTIONS[:-len(OLD_SUFFIX)]+NEW_SUFFIX


def case(index):
    if type(index) is not int or not 0<=index<10:raise ValueError('Complete ten-case persona diagnostic required')
    old=json.loads((Path(__file__).with_name('fixtures')/'native-v2-development-20261019.json').read_text())['cases'][index]
    return dict(old,index=index,checkpoint_job='7558824',persona_profile=PROFILE,
        diagnostic_only=True,outputs_must_not_be_trained_on=True,qualification_approved=False)


def policy(args):
    index=getattr(args,'native_persona_diagnostic_case',None);spec=case(index)
    data='/projects/u6si/duplex-live/runs/'+('native-curriculum-v2-data-7541875' if spec['source']=='native-v2' else 'recovery-data-7407554')
    if (args.persona_profile!=PROFILE or args.native_sampling_diagnostic
            or getattr(args,'native_seen_retention_diagnostic',False) or args.native_text_temperature!=.7
            or args.scenario!=spec['scenario'] or args.expected_kind!=spec['kind'] or args.backend_mode!=spec['backend']
            or args.train_history!=(index<6) or str(args.data)!=data
            or str(args.adapter_proof)!='/projects/u6si/duplex-live/runs/context-training-proof-7558824'
            or args.private_value!=spec.get('value','red circle') or args.voice!='accepted-synthetic'
            or args.adapter_disabled or args.adapter_zero or args.warm_audio or args.ret_logit_bias
            or args.pending_status or args.lookup_delivery_delay_s
            or getattr(args,'backend_reference_profile','answer-only')!='answer-only'
            or getattr(args,'event_log_mode','immediate')!='immediate' or getattr(args,'math_attention',False)):
        raise ValueError('Persona correction is one locked inference-only complete10 diagnostic, never a TRAIN collector')
    scoped_persona()
    return dict(mode='native-records-request-only-persona-diagnostic-v1',text_temperature=.7,acoustic_temperature=.8,
        seed=42424242,source_words_audio_voice_and_weights_unchanged=True,
        persona_change='Only scope the original final project-clarification instruction to actual record lookup requests',
        optimizer_steps=0,diagnostic_outputs_must_not_be_trained_on=True,
        native_quality_or_extension_approved=False,forced_reference_words=False,
        raw_train_stimulus_capture_is_diagnostic_only=True,case_index=index)
