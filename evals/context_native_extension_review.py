"""Bind a bounded native continuation to a committed complete behavior review."""
import json
import subprocess
from pathlib import Path
from evals.moshi_model_download import sha256


def committed_document(path):
    path=Path(path).resolve()
    root=Path(subprocess.check_output(['git','-C',str(path.parent),'rev-parse','--show-toplevel'],text=True).strip())
    relative=str(path.relative_to(root))
    subprocess.check_call(['git','-C',str(root),'ls-files','--error-unmatch',relative],stdout=subprocess.DEVNULL)
    if subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip():
        raise ValueError('Native continuation review must be deployed in a clean immutable Git checkout')
    committed=subprocess.check_output(['git','-C',str(root),'show','HEAD:'+relative])
    if committed!=path.read_bytes():raise ValueError('Uncommitted continuation review')
    return json.loads(committed)


def validate_review(review, proof, parent, identity):
    """Check every run and independent audio record, including failed-task evidence."""
    proof=Path(proof);runs=proof.parent
    candidate=proof.name.removeprefix('context-training-proof-')
    if not proof.name.startswith('context-training-proof-') or not candidate.isdigit():
        raise ValueError('Native continuation needs an exact numeric pilot job')
    plan=Path(__file__).with_name('fixtures')/'native-v2-development-20261019.json'
    cases=json.loads(plan.read_text())['cases']
    if (review['schema']!='dl-native-v2-bounded-extension-review-v1'
            or review['candidate_job']!=candidate
            or review['evaluation_plan_sha256']!=sha256(plan)
            or review['candidate_checkpoint_sha256']!=identity['checkpoint_sha256']
            or review['candidate_training_state_sha256']!=identity['training_state_sha256']
            or review['independent_reload_report_sha256']!=identity['reload_report_sha256']
            or not review['all10_native_cases_reviewed'] or not review['all4_ordinary_core_correct']
            or not review['bounded_extension_authorized'] or review['candidate_adopted']
            or review['native_candidate_rejected_for_extension'] or review['extension_optimizer_updates']!=32
            or parent['optimizer_steps']!=8
            or not parent['additional_native_v2'].get('complete24_per8')
            or [c['index'] for c in review['cases']]!=list(range(10))):
        raise ValueError('Bounded continuation lacks a complete ordinary-retaining native review of this exact full24 pilot')
    array=str(review['native_evaluation_array_job']);speech_job=str(review['speech_audit_job'])
    if not array.isdigit() or not speech_job.isdigit():raise ValueError('Invalid native evidence job identity')
    speech=runs/('native-v2-development-asr-'+speech_job)/'report.json'
    if sha256(speech)!=review['independent_speech_report_sha256']:raise ValueError('Changed full10 independent speech audit')
    audited=json.loads(speech.read_text())['results']
    if len(audited)!=10:raise ValueError('Independent speech audit omits native cases')
    for index,(case,audio) in enumerate(zip(review['cases'],audited)):
        run=runs/('native-v2-development-'+array+'_'+str(index));report=run/'report.json'
        r=json.loads(report.read_text());adapter=r['loading']['adapter']
        if (sha256(report)!=case['run_report_sha256'] or sha256(run/'model.wav')!=case['audio_sha256']
                or r['audio_sha256']!=case['audio_sha256'] or audio['wav_sha256']!=case['audio_sha256']
                or Path(audio['run'])!=run or adapter['checkpoint_sha256']!=identity['checkpoint_sha256']
                or adapter['voice_codes_sha256']!=parent['voice_sha256']
                or adapter['zero'] or adapter['disabled'] or not adapter['installed_before_native_graphs']
                or r['loading'].get('input_warmup') or r['scenario']!=cases[index]['scenario']
                or not case['behavior_and_receipts_reviewed'] or not case['speech_quality_issue_reviewed']
                or not case['reason'].strip() or case['behavior_verdict'] not in ('pass','partial','fail')
                or r['scheduled_dispatch'] or r['word_or_acoustic_tokens_forced'] or r['response_start_forced']
                or r['real_user_samples_processed']!=r['real_user_samples_expected']):
            raise ValueError('Native continuation evidence differs, is forced or omits a full task review')
        if index<4 and (not case['ordinary_core_correct'] or not audio['asr_time_bounds_valid']
                or not audio['asr_word_bounds_valid'] or not audio['word_starts_within_recorded_audio']
                or not 0<=audio['wer_against_model_text']<=.08):
            raise ValueError('Do not extend a candidate with ordinary reasoning or speech regression')
    return review


def validate(args):
    from evals.context_rollout_conditioning import verify_checkpoint
    proof=Path(args.resume_training).parent
    if Path(args.resume_training)!=proof/'training-state.pt':raise ValueError('Exact complete native training state required')
    identity,parent=verify_checkpoint(proof)
    if (parent['repo_commit']!=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
            or parent['additional_native_v2'].get('learning_rate')!=args.native_v2_learning_rate):
        raise ValueError('Native continuation requires the original immutable recipe source and learning rate')
    review=committed_document(args.native_v2_extension_review)
    return validate_review(review,proof,parent,identity)
