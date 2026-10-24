"""Record the complete failed development review; never authorize fresh inference.

Consumes the already hash-bound complete20 extraction and closed Mac backup.
No model, neural recognition, provider call, training or waveform mutation.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from evals.native_control40_fresh_review import validate_review, SCHEMA
from evals.native_fresh_inputs_v2 import PLAN_SHA256

ROOT = Path('runs/local-migration-20261023')
INPUT = ROOT/'native-control40-complete20-review-inputs-v1.json'
EXPECTED = 'fcbba6904f6603b58ff4437f2bb3334e52d536c4efaa12f1a36f25140b86b5df'

REASONS = {
 'baseline520': [
  ('fail', False, 'Incorrect ice explanation; no lower-density account.'),
  ('partial', True, 'Correct vibration core, followed by irrelevant project questions and an invented police-department project.'),
  ('fail', False, 'Incomplete make-their-own answer; no food/photosynthesis explanation. Independent speech WER .56 is also failed.'),
  ('partial', True, 'Correct rapid heat-transfer core; unnecessary temperature-lookup offer.'),
  ('fail', None, 'Speaks project question inside the explicit quiet interval. Later timer answer also invents an energy-expulsion function.'),
  ('fail', None, 'Speaks Okay, I am ready during required pause and invents/restarts the requested list rather than preserving book/shelf and resuming cupboard.'),
  ('fail', None, 'Actual yellow-circle receipt reaches context and value appears in speech, but ordinary lightning/thunder answer and added marker story are incorrect.'),
  ('fail', None, 'Actual changed copper-diamond receipt is used, but ordinary question is unanswered accurately and speech confuses marker with thunder/light.'),
  ('fail', None, 'No native request, project clarification or brown-square lookup; evaporation answer degenerates into unsupported ether claims. Independent WER .95 and clocks fail.'),
  ('fail', None, 'Actual cancel_lookup receipt exists for ret-0, but speech does not acknowledge cancellation appropriately and continues an irrelevant marker/lightning story. Invalid word-start flag prevents qualified precise post-cancel timing.'),
 ],
 'candidate-control40': [
  ('fail', False, 'Incorrect ice/heat explanation; no lower-density account and irrelevant project questions.'),
  ('pass', True, 'Correct vibration-of-air explanation with relevant optional follow-up; independent speech agrees.'),
  ('pass', True, 'Correct photosynthesis and light-to-chemical-energy core. Unnecessary species/light questions and vague chloroplast tail retained. One full-scale PCM sample is diagnostic, not proof of audible clipping.'),
  ('fail', False, 'Incorrect blood-flow explanation and claim that metal is less conductive; matched baseline had correct heat-transfer core.'),
  ('fail', None, 'Says I am listening and asks Which project during the required quiet interval, before cue release.'),
  ('fail', None, 'Says I am staying quiet during the required quiet interval, invents an alarm and fails the requested list/resumption.'),
  ('fail', None, 'Says room marker is two at9.09-9.31s before actual yellow-circle result12.8257s/applied12.9517s; never corrects the value. Ordinary answer also starts incorrectly and becomes repetitive. WER .1081 exceeds ordinary speech bar.'),
  ('fail', None, 'Same user PCM, different actual copper-diamond result14.5560s/applied14.6328s, but says two at9.07-9.27s and never corrects it. Correct light-speed words begin14.89s after result, so pending-work ordinary response is not established.'),
  ('fail', None, 'No native request, project clarification or brown-square lookup. Evaporation core is followed by unsupported aroma claim; WER .2241 and clock bounds fail.'),
  ('fail', None, 'No actual cancel_lookup receipt: second native request at10.6083s hears only incomplete earlier user input and backend clarifies. Speech implies cancellation and confuses light/marker/action; actual action was not executed. Invalid word-start flag retained.'),
 ]}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert sha(INPUT) == EXPECTED
    raw = json.loads(INPUT.read_text())
    backup = json.loads((ROOT/'native-control40-complete20-backup-review-v1.json').read_text())
    assert raw['all20_native_and20_independent_speech_records_bound'] is True
    assert backup['all_files_rehashed_on_Mac'] and backup['complete_paired20_and_independent_speech20']
    models = {}
    for role, model in raw['models'].items():
        rows = []
        assert [c['index'] for c in model['cases']] == list(range(10))
        for c, (verdict, core, reason) in zip(model['cases'], REASONS[role]):
            a = c['independent_speech']
            ready = []
            for event in c['worker_events']:
                if event['event'] != 'backend_result_ready':
                    continue
                result = event['result']
                ready.append(dict(task_id=event['task_id'], ready_at_s=event['ready_at_s'],
                    applied_at_s=event.get('applied_at_s'), route=result['route'],
                    content=result['content'], actual_receipts=result.get('receipts', []),
                    actual_arrived_ASR=event.get('asr', {}).get('text'),
                    response_ids=[e['response_id'] for e in result.get('events', [])]))
            row = dict(index=c['index'], scenario=c['scenario'],
                run_report_sha256=c['report_sha256'], audio_sha256=c['audio_sha256'],
                user_audio_sha256=c['user_audio_sha256'], model_text=c['model_text'],
                independent_asr_text=a['asr_text'], behavior_verdict=verdict,
                ordinary_core_correct=core, reason=reason,
                behavior_and_receipts_reviewed=True, speech_quality_issue_reviewed=True,
                human_listening_qualification=False, PCM=c['PCM'],
                speech={k:a[k] for k in ('wer_against_model_text', 'asr_time_bounds_valid',
                    'asr_word_bounds_valid', 'word_starts_within_recorded_audio')},
                runtime=c['runtime'], actual_backend_results=ready)
            if 'conservative_quiet_diagnostic' in c:
                row['conservative_quiet_diagnostic'] = c['conservative_quiet_diagnostic']
            rows.append(row)
        assert sum(c['ordinary_core_correct'] is True for c in rows[:4]) == 2
        models[role] = dict(actor_execution_sha256=model['actor_execution_sha256'],
            speech_report_sha256=model['speech_report_sha256'], ordinary_cores_correct=2,
            ordinary_cores_total=4, all10_reviewed=True, cases=rows)
    proof = '/workspace/migration/native-control40-precise-pilot-v11-environment-retry/training'
    plan = Path('evals/fixtures/native-v2-development-20261019.json')
    review = dict(schema=SCHEMA, candidate_proof=proof,
        candidate_checkpoint_sha256='0cb85b4a3b712ba5687e3e23dba6541891cbad954bde4a155ae9cb913517bcff',
        candidate_training_state_sha256='86dde08eb1591d0bc04a36c47be0dc7ee1b90019d7ae520f1d049cc37d8dc278',
        independent_reload_report_sha256='9dec1e0a98dd38c27c8366ef330b508665b0d71190f22c69b2ca52b362c3ffd3',
        evaluation_plan_sha256=sha(plan), fresh_plan_sha256=PLAN_SHA256,
        all10_native_cases_reviewed=True, all4_ordinary_core_correct=False,
        fresh_evaluation_authorized=False, checkpoint_adopted=False,
        native_asr_placement='shared-gpu-evaluation',
        native_pair_root='/workspace/migration/native-control40-native20-v1',
        independent_speech_root='/workspace/migration/native-control40-native20-speech-v1',
        native_pair_execution_sha256=raw['pair_execution_sha256'],
        independent_speech_controller_sha256=raw['speech_controller_sha256'],
        native_execution_source_commit=raw['source_commit'],
        actor_execution_sha256={k:v['actor_execution_sha256'] for k,v in models.items()},
        cases=models['candidate-control40']['cases'])
    identity={k:review['candidate_'+k] for k in ('checkpoint_sha256','training_state_sha256')}
    identity['reload_report_sha256']=review['independent_reload_report_sha256']
    parent=json.loads((Path(backup['snapshot'])/'native-control40-precise-pilot-v11-environment-retry/training/report.json').read_text())
    try:
        validate_review(review, proof, parent, identity)
    except ValueError as error:
        rejected = str(error)
        assert rejected == 'Complete restored control40 candidate and unchanged full10/ordinary4 review required'
    else:
        raise AssertionError('Failed complete review must never authorize sealed fresh inference')
    out=dict(schema='dl-native-control40-complete20-behavior-review-v1',
        at_utc=datetime.now(timezone.utc).isoformat(), source_commit=raw['source_commit'],
        extraction_sha256=EXPECTED, complete20_native_and_speech_reviewed=True, models=models,
        development_entry_review=review, unchanged_fresh_validator_rejected=True,
        rejection=rejected, sealed_fresh_payload_or_outputs_opened=False,
        current_candidate_behavior_qualified=False, candidate_adopted=False,
        training_extension_authorized=False, complete20_backup=backup,
        inference='Correct actual backend receipts plus full exact restore do not fix native output selection, silence or reasoning. Optimizer/data/coverage causality remains unisolated; do not prescribe batch size or longer training from this result.',
        scope='Reused development cases, complete diagnostic review; no held-out generalization or human listening qualification.')
    Path('docs/NATIVE-CONTROL40-COMPLETE20-BEHAVIOR-REVIEW-2026-10-24.json').write_text(json.dumps(out,indent=2)+'\n')
    (ROOT/'native-control40-development-review-v1.json').write_text(json.dumps(review,indent=2)+'\n')
    print(json.dumps(dict(reviewed=20, baseline_ordinary=2, candidate_ordinary=2,
        fresh_gate_rejected=True, training_extension_authorized=False)))


if __name__ == '__main__':
    main()
