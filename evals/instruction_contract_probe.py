"""Declared frozen-model instruction contrasts with actual fixture dispatch.

No external calls, training, captured owner data, output repair or routing.
New fixtures are authored here before any model output is observed.
"""
import argparse
import copy
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def resume_fixture(status, variant, model):
    from evals.duplex_reasoning_worker import ReasoningWorld, run_reasoning_worker
    def forbidden(question):
        raise RuntimeError('No external reasoning requests permitted')
    world = ReasoningWorld({}, reasoner=forbidden)
    target, other = 'blue-count', 'red-count'
    history = [dict(role='user', text='Count arrangements for the blue exercise.'),
               dict(role='user', text='Separately count arrangements for the red exercise.')]
    for identity, colour in ((target, 'blue'), (other, 'red')):
        world.reserve(identity)
        world.tasks[identity].update(operation='delegate_reasoning', phase='running', question=f'Count arrangements for the {colour} exercise.')
    if status != 'active':
        world.tasks[target].update(closed=True, phase='completed')
        history.append(dict(role='assistant', text='The blue exercise has twelve arrangements.',
                            control_source='actual-speech-generation', speech_task_id=target,
                            first_speech_event_id='fixture-complete', latest_speech_event_id='fixture-complete',
                            playback_segments=1, speech_delivery='server-playout-complete', physical_hearing_confirmed=False))
    if status == 'finished':
        world.tasks[other].update(closed=True, phase='completed')
    ordinary = 'Give two steps for storing a scarf.'
    prepared = 'Fold the scarf neatly. Place it in a drawer.'
    history.extend([dict(role='user', text=ordinary), dict(role='assistant', text=prepared,
        control_source='actual-speech-generation', speech_task_id='scarf-steps',
        first_speech_event_id='fixture-retained', latest_speech_event_id='fixture-retained',
        playback_segments=1, speech_delivery='generated-not-fully-played', physical_hearing_confirmed=False)])
    latest = ('Cancel the blue exercise and continue your scarf instructions.',
              'Stop delivery of the blue counting result; carry on from where you paused.',
              'Withdraw the blue result, leave the red exercise alone, and finish the same explanation.')[variant]
    history.append(dict(role='user', text=latest))
    revision = world.reserve('latest', history)
    pending = [dict(id='scarf-steps', request_text=ordinary, text=prepared)]
    resume = dict(speech_id='scarf-steps', event_id='fixture-retained')
    def run(executor):
        result = run_reasoning_worker(history, world, 'latest', model, revision, 'state-aware', pending, executor,
            resumable_speech=resume, bound_resume_target=True, allow_cancel_resume=True)
        checks = dict(same_generator=result.get('resume_target') == resume, no_new_words=result['content'] == '',
            route=result['route'] == 'resume', no_retirement=result['consume_pending_speech_ids'] == [],
            matching_cancellation=world.tasks[target]['cancelled'] == (status == 'active'),
            unrelated_preserved=not world.tasks[other]['cancelled'],
            actual_receipts=len(result['receipts']) == (1 if status == 'active' else 0))
        return result, checks, world.tasks
    return dict(status=status, variant=variant, context=history, pending_speech=pending, resume_target=resume,
                actual_world_tasks=copy.deepcopy(world.tasks), synthetic_fixture=True, real_playback=False), run


def record_fixture(variant, form, model):
    from evals.recovery_worker import RecordWorld, run_worker
    from data.synthetic.filter import norm_words
    project = ('Silver Orchard', 'Amber Harbor', 'Jade Valley')[variant]
    value = ('violet canoe', 'bronze feather', 'orange lantern')[variant]
    if form == 'changed-value':
        value = ('white anchor', 'purple ribbon', 'teal button')[variant]
    request = f'Look up the dispatch emblem for Project {project}. '
    request += 'Reply in one full sentence naming the project and its value.' if form == 'sentence' else 'Reply with only its value.'
    world = RecordWorld({project: {'dispatch emblem': value}, 'Other Project': {'dispatch emblem': 'wrong value'}})
    context = [dict(role='user', text=request)]
    revision = world.reserve('latest', context)
    def run(executor):
        result = run_worker(context, world, 'latest', model, revision, 'state-aware', [], executor, scope_read_tools=True)
        checks = dict(actual_read=len(result['receipts']) == 1 and result['receipts'][0].get('executed') is True
            and result['receipts'][0].get('project') == project and result['receipts'][0].get('value') == value)
        words = norm_words(result['content'])
        checks['requested_form'] = (' '.join(norm_words(project)) in ' '.join(words) and ' '.join(norm_words(value)) in ' '.join(words)
            and len(words) > len(norm_words(value))) if form == 'sentence' else words == norm_words(value)
        return result, checks, world.tasks
    return dict(variant=variant, form=form, context=context, records=world.records, expected_value=value,
                synthetic_fixture=True, external_private_access=False), run


def captured_result(payload, response, model):
    from evals.duplex_reasoning_worker import ReasoningWorld, run_reasoning_worker
    state = payload['local_worker_state']; context = json.loads(payload['input'])
    world = ReasoningWorld({}, reasoner=lambda question: (_ for _ in ()).throw(RuntimeError('External request prohibited')))
    world.tasks = copy.deepcopy(state['available']['tasks'])
    revision = world.reserve('capture-replay', context)
    class Replay:
        provider_name = 'actual-local-output-synthetic-state-replay'
        def __call__(self, new):
            if new != payload:
                raise ValueError('Actual captured worker input changed')
            return response
    options = {}
    if state.get('resumable_speech') is not None:
        options = dict(resumable_speech=state['resumable_speech'], bound_resume_target=True, allow_cancel_resume=True)
    result = run_reasoning_worker(context, world, 'capture-replay', model, revision, 'state-aware', state['pending_speech'], Replay(), **options)
    active = [k for k, v in state['available']['tasks'].items() if v.get('operation') == 'delegate_reasoning']
    checks = dict(same_generator=result.get('resume_target') == state.get('resumable_speech'),
        route=result['route'] == 'resume', no_new_words=result['content'] == '',
        no_retirement=result['consume_pending_speech_ids'] == [],
        cancellation=sorted(k for k in active if world.tasks[k]['cancelled']) == sorted(active))
    return result, checks, world.tasks


def focused(executor, folder, runs, original=False):
    from evals.duplex_actual_controller import capture
    from evals.failed_controller_capture import capture_failure
    folder.mkdir(); rows = []
    for identity, payload, provenance in [
        ('actual-closed', *capture_failure(runs/'duplex-live-transport-7525920/live-session-0/turn-3-failure.json')),
        ('actual-active', *capture(runs/'duplex-live-transport-7524610', 'turn-3'))]:
        row = dict(id=identity, source=provenance, payload=payload, passed=False)
        try:
            response = executor(copy.deepcopy(payload))
            result, checks, tasks = captured_result(payload, response, executor.model_id)
            row.update(response=response, result=result, checks=checks, actual_world_after=tasks, passed=all(checks.values()))
        except Exception as error:
            row.update(error_type=type(error).__name__, message=str(error), evidence=getattr(error, 'result', None))
        inference = row.get('response', {}).get('local_inference') or (row.get('evidence') or {}).get('local_generation', {})
        row['historical_input_length_exact'] = inference.get('input_tokens') == provenance['original_input_tokens']
        row['historical_raw_output_exact'] = inference.get('raw_text') == provenance['original_raw_text']
        if original and not row['historical_input_length_exact']:
            raise ValueError('Original captured request token length changed')
        rows.append(row); (folder/(identity+'.json')).write_text(json.dumps(row, indent=2)+'\n')
    for kind, variants in [('resume', [(s, v) for s in ('active', 'finished', 'unrelated-active') for v in range(3)]),
                           ('receipt', [(v, f) for v in range(3) for f in ('only-value', 'sentence', 'changed-value')])]:
        for args in variants:
            identity = kind+'-'+'-'.join(map(str, args)); row = dict(id=identity, kind=kind, passed=False)
            try:
                scene, run = (resume_fixture if kind == 'resume' else record_fixture)(*args, executor.model_id)
                row['scene'] = scene
                result, checks, tasks = run(executor)
                row.update(result=result, checks=checks, actual_world_after=tasks, passed=all(checks.values()))
            except Exception as error:
                row.update(error_type=type(error).__name__, message=str(error), evidence=getattr(error, 'result', None))
            rows.append(row); (folder/(identity+'.json')).write_text(json.dumps(row, indent=2)+'\n')
    (folder/'report.json').write_text(json.dumps(dict(results=rows, passed=sum(r['passed'] for r in rows), cases=len(rows)), indent=2)+'\n')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assets', 'kernel-assets', 'runs', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    from evals.qwen35_controller import Qwen35ResponseExecutor
    from evals.qwen35_prefill_warmup import warm_long_prefill
    from evals.instruction_contract import InstructionContractExecutor
    from training.control.head_adapter import parameter_hashes
    executor = Qwen35ResponseExecutor(args.assets, kernel_assets=args.kernel_assets,
        pending_speech_protocol='explicit-actions', compact_replies=True)
    warm_long_prefill(executor)
    before = parameter_hashes(dict(executor.model.named_parameters()))
    results = {}; requests = {}; suites = {}
    for mode in ('original', 'resume', 'receipt'):
        current = executor if mode == 'original' else InstructionContractExecutor(executor, mode)
        rows = focused(current, args.out/mode, args.runs, original=mode=='original'); results[mode] = rows
        if mode != 'original':
            requests[mode] = current.requests
            relevant = [r for r in rows if r.get('kind') == mode or (mode == 'resume' and r['id'].startswith('actual-'))]
            gate = all(r['passed'] for r in relevant)
            if gate:
                from training.control.context_evaluate import evaluate
                from training.control.paired_curriculum import specifications
                from training.control.capability_curriculum import specifications as capabilities
                from training.control.capability_evaluate import evaluate as evaluate_capabilities
                suites[mode] = dict(retention=evaluate(current, [s for s in specifications() if s['split']=='heldout'], args.out/(mode+'-retention')),
                    capabilities=evaluate_capabilities(current, [s for s in capabilities() if s['split']=='heldout'], args.out/(mode+'-capabilities')),
                    semantic_review_required=True)
            else:
                suites[mode] = dict(focused_gate_passed=False, broad_evaluation_not_run=True)
            (args.out/(mode+'-requests.json')).write_text(json.dumps(current.requests, indent=2)+'\n')
        print(json.dumps(dict(mode=mode, focused_passed=sum(r['passed'] for r in rows), focused_cases=len(rows), suites=suites.get(mode))))
    if before != parameter_hashes(dict(executor.model.named_parameters())):
        raise ValueError('Inference modified model parameters')
    report = dict(schema='dl-instruction-contract-probe-v1', results=results, suites=suites,
        all_parameter_hashes_unchanged=True, source_model=executor.proof, optimizer_steps=0,
        actual_external_requests=0, output_repair=False, function_forcing=False,
        qualified=False, production_adopted=False, owner_demo_changed=False, do_not_train=True,
        wrapper_sha256=sha256(Path('evals/instruction_contract.py')), code_sha256=sha256(Path(__file__)))
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
