"""Explicit DEVELOPMENT-only attachment; publisher load proof stays separate."""
import hashlib
import json
from pathlib import Path

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def selector_metadata(training_run,assets,capability_profile=False,pending_speech_only=False):
    folder=Path(training_run);report=json.loads((folder/'report.json').read_text());checkpoint=folder/'candidate-adapter';m=json.loads((checkpoint/'adapter.json').read_text())
    if (not report['component_execution_completed'] or not report['all_source_parameter_hashes_unchanged']
            or not report['strict_checkpoint_restore_logits_exact'] or not report['optimizer_state_restore_exact']
            or not report['rng_state_restore_exact'] or report['optimizer_steps']!=648
            or m!=report['adapter_metadata'] or m['purpose']!='trained-development-candidate'
            or m['base_assets_sha256']!=sha(assets) or m['weights_sha256']!=sha(checkpoint/'adapter.safetensors')
            or m['implementation_sha256']!=sha(Path(__file__).with_name('head_adapter.py'))):raise ValueError('Development selector lacks exact completed training/base/checkpoint proof')
    result=dict(schema='dl-development-function-selector-inference-v1',mode='function-name-only',training_report_sha256=sha(folder/'report.json'),
        adapter_weights_sha256=m['weights_sha256'],base_assets_sha256=m['base_assets_sha256'],rank=m['rank'],scale=m['scale'],
        implementation_sha256={name:sha(Path(__file__).with_name(name)) for name in ('inference.py','head_adapter.py','function_boundary.py','scoped_head.py')},
        development_only=True,qualified=False,content_projection='published-original')
    if capability_profile:
        from training.control.capabilities import PROFILE_SHA256
        result['capability_profile_sha256']=PROFILE_SHA256
        result['implementation_sha256']['capabilities.py']=sha(Path(__file__).with_name('capabilities.py'))
    if pending_speech_only:result['application_condition']='pending-speech-actions-provided'
    return result


def attach_development_selector(executor,training_run,assets,capability_profile=False,pending_speech_only=False):
    metadata=selector_metadata(training_run,assets,capability_profile,pending_speech_only)
    if executor.proof.get('function_selector') is not None:raise ValueError('Selector already attached')
    from training.control.head_adapter import load_delta
    from training.control.scoped_head import FunctionNameHead,FunctionNameGenerator
    delta,checked=load_delta(Path(training_run)/'candidate-adapter',sha(assets),executor.model.device);delta.requires_grad_(False)
    executor.model.lm_head=FunctionNameHead(executor.model.lm_head,delta);executor.generator=FunctionNameGenerator(executor.generator,executor.model.lm_head,pending_speech_only=pending_speech_only)
    if capability_profile:
        from training.control.capabilities import CapabilityGenerator
        executor.generator=CapabilityGenerator(executor.generator)
    if any(p.requires_grad for p in executor.model.parameters()):raise ValueError('Inference selector must leave every parameter frozen')
    executor.proof=dict(executor.proof,function_selector=metadata,effective_inference_model_parameters=sum(p.numel() for p in executor.model.parameters()),
        output_parameter_mapping=dict(original='lm_head.weight',wrapped='lm_head.base.weight',added=['lm_head.delta.down','lm_head.delta.up']))
    # An actual model request verifies the new head/hook before socket readiness.
    from evals.duplex_reasoning_worker import ReasoningWorld,run_reasoning_worker
    def blocked(question):raise RuntimeError('Selector warmup forbids external requests')
    world=ReasoningWorld({},blocked);context=[dict(role='user',text='What is two plus three? Give only the answer.')];revision=world.reserve('selector-warmup',context)
    result=run_reasoning_worker(context,world,'selector-warmup',executor.model_id,revision,'state-aware',[],executor)
    if result['route']!='answer' or result['receipts'] or result['events'][0]['local_inference'].get('output_correction_scope')!='function-name-only':raise ValueError('Development selector warmup did not use its exact local boundary')
    executor.warmup=dict(original=executor.warmup,development_function_selector=dict(result=result,external_requests=0,training=False))
    if pending_speech_only:
        context=[dict(role='user',text='Please explain something.'),dict(role='assistant',text='This is an unfinished explanation.'),dict(role='user',text='Stop speaking and say nothing else.')]
        pending=[dict(id='warmup-owed',request_text=context[0]['text'],text=context[1]['text'])]
        fixture=ReasoningWorld({},blocked);revision=fixture.reserve('selector-state-warmup',context)
        checked=run_reasoning_worker(context,fixture,'selector-state-warmup',executor.model_id,revision,'state-aware',pending,executor)
        if checked['route']!='silence' or checked['consume_pending_speech_ids']!=['warmup-owed'] or not checked['events'][0]['local_inference']['function_name_correction_enabled']:raise ValueError('Pending-speech selector warmup did not exercise its actual correction')
        executor.warmup['development_function_selector']['pending_state_fixture']=dict(result=checked,synthetic_fixture=True,physical_speech=False)
    return metadata
