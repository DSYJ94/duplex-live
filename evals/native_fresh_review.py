"""Require a committed complete native development review before fresh V2 outputs."""
from pathlib import Path
from evals.context_native_extension_review import committed_document, validate_cases
from evals.moshi_model_download import sha256
from evals.native_fresh_inputs_v2 import PLAN_SHA256


def validate_review(review, proof, parent, identity):
    candidate=Path(proof).name.removeprefix('context-training-proof-')
    plan=Path(__file__).with_name('fixtures')/'native-v2-development-20261019.json'
    if (not candidate.isdigit() or review['schema']!='dl-native-v2-development-review-v1'
            or review['candidate_job']!=candidate or review['fresh_plan_sha256']!=PLAN_SHA256
            or review['candidate_checkpoint_sha256']!=identity['checkpoint_sha256']
            or review['candidate_training_state_sha256']!=identity['training_state_sha256']
            or review['independent_reload_report_sha256']!=identity['reload_report_sha256']
            or review['evaluation_plan_sha256']!=sha256(plan)
            or not review['all10_native_cases_reviewed'] or not review['all4_ordinary_core_correct']
            or not review['fresh_evaluation_authorized'] or review['checkpoint_adopted']
            or parent['optimizer_steps'] not in (8,40)
            or not parent['additional_native_v2'].get('complete24_per8')):
        raise ValueError('Fresh qualification needs the exact restored full24 candidate and committed all10 native review')
    if parent['optimizer_steps']==40 and (parent['initial_optimizer_steps']!=8
            or parent['updates_this_run']!=32 or not parent['resumed_from_training_state']):
        raise ValueError('Fresh extended candidate lacks its exact independently restored continuation')
    return validate_cases(review,proof,parent,identity)


def validate(path, proof, actual_proof):
    if path is None or proof is None:raise ValueError('Committed complete10 behavior review required before fresh V2 inference')
    proof=Path(proof).resolve()
    if Path(actual_proof).resolve() not in (proof,proof.parent/'context-training-proof-7510331'):
        raise ValueError('Fresh comparison permits only the locked candidate and accepted native520 baseline')
    from evals.context_rollout_conditioning import verify_checkpoint
    identity,parent=verify_checkpoint(proof)
    return validate_review(committed_document(path),proof,parent,identity)
