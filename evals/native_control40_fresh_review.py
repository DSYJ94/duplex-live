"""Complete portable native20/audio20 review before any sealed fresh output.

Real checkpoint and artifact paths replace Slurm array identities. All original
ordinary, speech, full-ten review, restore and unforced-runtime gates remain.
This authorizes measurement only, never continuation or model adoption.
"""
import hashlib
import json
from pathlib import Path
import subprocess

from evals.moshi_model_download import sha256
from evals.native_fresh_inputs_v2 import PLAN_SHA256

SCHEMA='dl-native-control40-development-review-v1'
ROLES=('baseline520','candidate-control40')
REASONING_SCHEMA='dl-native-reasoning32-development-review-v1'
REASONING_ROLES=('baseline520','candidate-reasoning32')


def cohort(schema):
    if schema==SCHEMA:
        return ROLES,'dl-native-control40-retained154-stream49-proof-v1'
    if schema==REASONING_SCHEMA:
        return REASONING_ROLES,'dl-native-reasoning32-retained154-stream49-control40-proof-v1'
    raise ValueError('Explicit complete control40 or reasoning32 development review required')


def implementations(commit,files=None):
    from portable.context.native_development import INFERENCE_FILES
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Exact native evaluation source commit required')
    root=Path(__file__).resolve().parents[1];result={}
    for name in INFERENCE_FILES if files is None else files:
        content=subprocess.check_output(['git','show',commit+':'+name],cwd=root)
        result[name]=hashlib.sha256(content).hexdigest()
        if sha256(root/name)!=result[name]:raise ValueError('Fresh native execution differs from reviewed implementation: '+name)
    return result


def validate_review(review,proof,parent,identity):
    from portable.context.native_development import artifact_hashes,CASE_FILES,PARITY_FILES,compare_native
    from portable.context.native_speech_followup import SPEECH_FILES
    roles,parent_schema=cohort(review['schema'])
    proof=Path(proof).resolve();plan=Path(__file__).with_name('fixtures')/'native-v2-development-20261019.json'
    cases=json.loads(plan.read_text())['cases']
    if (Path(review['candidate_proof']).resolve()!=proof
            or review['candidate_checkpoint_sha256']!=identity['checkpoint_sha256']
            or review['candidate_training_state_sha256']!=identity['training_state_sha256']
            or review['independent_reload_report_sha256']!=identity['reload_report_sha256']
            or review['evaluation_plan_sha256']!=sha256(plan) or review['fresh_plan_sha256']!=PLAN_SHA256
            or review['all10_native_cases_reviewed'] is not True or review['all4_ordinary_core_correct'] is not True
            or review['fresh_evaluation_authorized'] is not True or review['checkpoint_adopted'] is not False
            or review['native_asr_placement']!='shared-gpu-evaluation'
            or parent['schema']!=parent_schema
            or parent['updates_this_run']!=8 or parent['adam_parameter_steps_after']!=[528]
            or not parent['additional_native_v2'].get('complete24_per8')
            or [c['index'] for c in review['cases']]!=list(range(10))):
        raise ValueError('Complete restored candidate and unchanged full10/ordinary4 review required')
    baseline=Path(parent['warm_start']['proof']).resolve()
    if baseline.name!='context-training-proof-7510331':raise ValueError('Exact original native520 comparator required')
    pair=Path(review['native_pair_root']);speech=Path(review['independent_speech_root'])
    if not pair.is_absolute() or not speech.is_absolute():raise ValueError('Explicit durable native/audio artifact paths required')
    p=json.loads((pair/'execution.json').read_text());s=json.loads((speech/'controller.json').read_text())
    if (sha256(pair/'execution.json')!=review['native_pair_execution_sha256']
            or sha256(speech/'controller.json')!=review['independent_speech_controller_sha256']
            or p['schema']!='dl-runpod-paired-native10-v1' or p['phase']!='complete' or p['complete_paired20'] is not True
            or p['gpu_mode']!='serial-one-gpu' or p['runtime_profile']!='container-default'
            or p['runtime_comparison_only'] is not False or p['optimizer_updates']!=0
            or p['provider_credential_transferred'] is not False or len(p['gpus'])!=1
            or list(p['proofs'])!=list(roles)
            or Path(p['proofs'][roles[0]]).resolve()!=baseline or Path(p['proofs'][roles[1]]).resolve()!=proof
            or p['plan_sha256']!=sha256(plan) or p['source_commit']!=review['native_execution_source_commit']
            or p['execution_implementation_sha256']!=implementations(p['source_commit'])
            or s['phase']!='complete_speech20_only' or s['complete20'] is not True
            or s['paired_source']!=p['source_commit'] or s['gpu_mode']!='serial-one-gpu'
            or s['schema']!='dl-runpod-paired20-independent-speech-followup-v1'
            or s['source_commit']!=p['source_commit']
            or s['speech_implementation_sha256']!=implementations(s['source_commit'],SPEECH_FILES)
            or s['paired_execution_sha256']!=review['native_pair_execution_sha256']
            or s['paired_roles']!=list(roles)):
        raise ValueError('Complete paired20 native and independent speech20 with unchanged runtime provenance required')
    for role in roles:
        actor=json.loads((pair/role/'execution.json').read_text())
        if (sha256(pair/role/'execution.json')!=review['actor_execution_sha256'][role]
                or actor['schema']!='dl-native-paired-development-actor-v1'
                or actor['phase']!='complete' or actor['complete10'] is not True or actor['parity']['passed'] is not True
                or actor['role']!=role or actor['optimizer_updates']!=0 or actor['provider_credential_present'] is not False
                or actor['source_commit']!=p['source_commit'] or actor['gpu_mode']!='serial-one-gpu'
                or actor['runtime_profile']!='container-default' or actor['assigned_main_gpu']!=0 or actor['assigned_asr_gpu']!=0
                or Path(actor['proof']).resolve()!=Path(p['proofs'][role]).resolve()
                or [r['index'] for r in actor['cases']]!=list(range(10)) or any(r['exit_code']!=0 for r in actor['cases'])):
            raise ValueError('All ten native cases and real PCM/token/input parity required for both models')
        for name in ('parity-cold','parity-intervening','parity-reused'):
            if actor['parity_artifact_sha256'][name]!=artifact_hashes(pair/role/name,PARITY_FILES):
                raise ValueError('Changed cold/intervening/reused parity artifacts')
            parity_raw=json.loads((pair/role/name/'report.json').read_text())
            if parity_raw['scenario']!=cases[1 if name=='parity-intervening' else 0]['scenario']:
                raise ValueError('Actual different intervening conversation must precede reused parity')
        if actor['parity']!=compare_native(pair/role/'parity-cold',pair/role/'parity-reused'):
            raise ValueError('Native PCM/token/input parity must independently reproduce')
        audit=speech/role/'report.json'
        if sha256(audit)!=s['reports_sha256'][role]:raise ValueError('Changed independent speech artifact')
        speech_report=json.loads(audit.read_text());rows=speech_report['results']
        if (speech_report['schema']!='dl-native-speech-audit-v1' or speech_report['device']!='cuda'
                or len(rows)!=10):raise ValueError('Complete independent ten-case CUDA speech audit required')
        for i,audio in enumerate(rows):
            run=pair/role/('case-%02d'%i);raw=json.loads((run/'report.json').read_text())
            loading=raw['loading'];adapter=loading['adapter'];placement=loading['native_asr_resource_placement']
            expected_checkpoint=(parent['warm_start']['checkpoint_sha256'] if role=='baseline520'
                                 else identity['checkpoint_sha256'])
            if (actor['cases'][i]['artifact_sha256']!=artifact_hashes(run,CASE_FILES)
                    or Path(audio['run']).resolve()!=run.resolve() or audio['wav_sha256']!=sha256(run/'model.wav')
                    or raw['audio_sha256']!=audio['wav_sha256'] or raw['scenario']!=cases[i]['scenario']
                    or audio['source_kind']!='native' or audio['model_text']!=raw['text']
                    or raw['scheduled_dispatch'] or raw['word_or_acoustic_tokens_forced'] or raw['response_start_forced']
                    or raw['real_user_samples_processed']!=raw['real_user_samples_expected']
                    or raw['real_user_samples_expected']<=0 or loading.get('input_warmup')
                    or loading.get('native_sampling_diagnostic') or loading.get('native_runtime_comparison_only')
                    or loading.get('ret_logit_calibration') or adapter.get('runtime_precision_diagnostic')
                    or adapter['zero'] or adapter['disabled'] or not adapter['installed_before_native_graphs']
                    or adapter['checkpoint_sha256']!=expected_checkpoint
                    or Path(adapter['proof']).resolve()!=Path(p['proofs'][role]).resolve()
                    or adapter['voice_codes_sha256']!=parent['voice_sha256']
                    or placement['profile']!='shared-gpu-evaluation' or placement['distinct_physical_gpus'] is not False
                    or placement['separate_native_and_asr_processes'] is not True):
                raise ValueError('Every native/audio case must retain unforced same-voice, same-resource actual execution')
            if cases[i]['backend']=='native-asr' and (raw['distinct_lm_asr_gpu_processes'] is not False
                    or raw['separate_native_and_asr_processes'] is not True
                    or raw['asr_environment']['reference_transcript_supplied'] is not False
                    or raw['asr_environment']['assigned_visible_gpu']!='0'):
                raise ValueError('Actual arrived-audio ASR placement and independent process required')
            if role==roles[1]:
                case=review['cases'][i]
                if (adapter['checkpoint_sha256']!=identity['checkpoint_sha256']
                        or case['run_report_sha256']!=sha256(run/'report.json') or case['audio_sha256']!=audio['wav_sha256']
                        or case['behavior_and_receipts_reviewed'] is not True or case['speech_quality_issue_reviewed'] is not True
                        or case['behavior_verdict'] not in ('pass','partial','fail') or not case['reason'].strip()):
                    raise ValueError('Complete per-case semantic/receipt/audio review of this exact candidate required')
                if i<4 and (case['ordinary_core_correct'] is not True or case['behavior_verdict']!='pass'
                        or not raw['text'].strip() or not audio['asr_text'].strip() or audio.get('audit_failure')
                        or not any(seg.get('words') for seg in audio['asr_segments'])
                        or not audio['asr_time_bounds_valid'] or not audio['asr_word_bounds_valid']
                        or not audio['word_starts_within_recorded_audio'] or not 0<=audio['wer_against_model_text']<=.08):
                    raise ValueError('All four ordinary reasoning and independent speech gates remain mandatory')
    for i in range(10):
        if sha256(pair/roles[0]/('case-%02d'%i)/'user_in.wav')!=sha256(pair/roles[1]/('case-%02d'%i)/'user_in.wav'):
            raise ValueError('Baseline and candidate must receive byte-identical development user PCM')
    return review


def validate(review,proof,actual_proof):
    from portable.context.native_development import restored_control40,restored_reasoning32
    roles,_=cohort(review['schema'])
    restored=(restored_reasoning32 if roles==REASONING_ROLES else restored_control40)(proof)
    parent=json.loads((Path(proof)/'report.json').read_text())
    if sha256(Path(proof)/'report.json')!=restored['report_sha256']:raise ValueError('Candidate changed after strict restore verification')
    allowed=(Path(proof).resolve(),Path(parent['warm_start']['proof']).resolve())
    if Path(actual_proof).resolve() not in allowed:raise ValueError('Only locked candidate and original native520 comparator permitted')
    return validate_review(review,proof,parent,restored['identity'])
