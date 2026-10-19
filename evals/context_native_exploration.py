"""Disabled until an explicit, committed owner authorization exists for clock8+32.

This accepts the failed development review as evidence of a training-only
experiment. It never converts that review into an ordinary/fresh/adoption pass.
"""
import json
import subprocess
from pathlib import Path
from evals.moshi_model_download import sha256

PARENT_JOB = '7556562'
PARENT_SOURCE = '3061dcba1de571d2dad58d466f7c3f3ca43e41ec'
CHECKPOINT = '1ad464e2b5ca2258ef777f013ccdfe057721a65cd974e17a32a1070b1f05016b'
STATE = '3303bf1624cf3c36ff071e9fb69a239b3b93fda6a0d914db67848b7543cad675'
RELOAD = '440b665312b3125f57dd26531212db5f2c90e8b2d5a08dcf1d04b5e18b7c8b9a'
REVIEW = 'd293e7444e867c67763cbbc0113377a5ed85845b3dee1906b4f68f3c58146a9b'
PROPOSAL = '6cf74309b29dfac9da2041af5e5c1706e640975242524191b2b27e14475394c0'


def validate_authorization(doc, identity, parent, source):
    expected = dict(schema='dl-native-bounded32-owner-authorization-v1',
        operation='one-training-only-clock8-plus32', explicit_owner_selection=True,
        parent_job=PARENT_JOB, parent_source=PARENT_SOURCE, execution_source=source,
        checkpoint_sha256=CHECKPOINT, training_state_sha256=STATE,
        independent_reload_report_sha256=RELOAD, failed_complete10_review_sha256=REVIEW,
        proposal_sha256=PROPOSAL, starting_optimizer_steps=8,
        additional_optimizer_updates=32, final_optimizer_steps=40,
        optimizer_reset=False, learning_rate=2e-6, maximum_gpus=1,
        maximum_gpu_walltime_minutes=30, reservation='interactive',
        all_data_losses_voice_targets_masks_unchanged=True,
        failed_early_ordinary_and_speech_gate_acknowledged=True,
        full_native10_fresh20_and_adoption_gates_unchanged=True,
        candidate_adoption_authorized=False, further_training_after40_authorized=False)
    if (not isinstance(source,str) or len(source)!=40
            or any(c not in '0123456789abcdef' for c in source) or source==PARENT_SOURCE
            or set(doc)!=set(expected)|{'owner_reply','at_utc'}
            or any(doc.get(k)!=v for k,v in expected.items())
            or not isinstance(doc['owner_reply'],str) or not doc['owner_reply'].strip()
            or not isinstance(doc['at_utc'],str) or not doc['at_utc'].strip()
            or identity['checkpoint_sha256']!=CHECKPOINT
            or identity['training_state_sha256']!=STATE or identity['reload_report_sha256']!=RELOAD
            or parent['repo_commit']!=PARENT_SOURCE or parent['optimizer_steps']!=8
            or not parent['additional_native_v2'].get('complete24_per8')
            or parent['additional_native_v2'].get('learning_rate')!=2e-6
            or parent.get('warm_ordinary_retention',{}).get('weight')!=64
            or parent.get('warm_ordinary_clock_retention',{}).get('weight')!=64):
        raise ValueError('No exact explicit owner authorization for this one bounded training exploration')
    return doc


def validate(args):
    from evals.context_native_extension_review import committed_document, validate_cases
    from evals.context_rollout_conditioning import verify_checkpoint
    proof=Path(args.resume_training).parent
    if (proof.name!='context-training-proof-'+PARENT_JOB
            or Path(args.resume_training)!=proof/'training-state.pt'
            or args.actual_fit_updates!=32 or args.native_v2_learning_rate!=2e-6
            or not args.native_v2_full24_per8 or args.native_v2_warm_ordinary_weight!=64
            or not args.native_v2_warm_ordinary_clock
            or getattr(args,'native_v2_gradient_audit',False)
            or getattr(args,'native_v2_retention_drift_proof',None)):
        raise ValueError('Owner-scoped exploration cannot change its parent, budget or anchored recipe')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    identity,parent=verify_checkpoint(proof)
    approval=Path(args.native_v2_exploration_authorization)
    doc=validate_authorization(committed_document(approval),identity,parent,source)
    review_path=Path(args.native_v2_extension_review)
    review=committed_document(review_path)
    if (sha256(review_path)!=REVIEW
            or review['schema']!='dl-native-v2-complete10-native-review-v1'
            or review['candidate_job']!=PARENT_JOB or review['complete_cases']!=10
            or not review['all_cases_retained'] or not review['no_passing_subset']
            or review['all4_ordinary_core_correct'] or not review['native_candidate_rejected_for_extension']
            or review['training_approved'] or review['candidate_adopted']
            or review['fresh_native_inference_authorized']):
        raise ValueError('The complete rejected pilot review must remain frozen and rejected')
    # Schema-only translation retains every failed verdict, word and waveform.
    bound=dict(review,native_evaluation_array_job=review['evaluation_case_group'],
        speech_audit_job=review['independent_speech_job'], cases=[dict(c,
            run_report_sha256=c['report_sha256'],behavior_verdict=c['behavior_review']) for c in review['cases']])
    validate_cases(bound,proof,parent,identity,require_ordinary=False)
    transition=dict(schema='dl-native-source-transition-v1',from_repo_commit=PARENT_SOURCE,
        to_repo_commit=source,training_state_sha256=STATE,authorization_sha256=sha256(approval),
        review_sha256=REVIEW,starting_optimizer_steps=8,additional_optimizer_updates=32,final_optimizer_steps=40)
    return dict(authorization=doc,source_transition=transition)
