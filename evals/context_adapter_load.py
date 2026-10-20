"""Load our verified adapter format before native Engine builds CUDA graphs."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from evals.moshi_model_download import sha256


@contextmanager
def before_engine(engine_module, proof, disabled=False, zero=False):
    import torch
    from safetensors.torch import load_file
    from evals.context_spanning_download import SOURCE_COMMIT
    from evals.personaplex_adapters import install_adapters, adapter_parameters
    if proof is None:
        if disabled or zero: raise RuntimeError('Adapter comparison requires a verified adapter proof')
        yield None
        return
    if disabled and zero: raise RuntimeError('Choose disabled or zero adapter comparison')
    report = json.loads((proof / 'report.json').read_text())
    reload = json.loads((proof / 'reload-report.json').read_text())
    digest = sha256(proof / 'adapters.safetensors')
    if report['schema'] == 'dl-context-result-head-proof-v1':
        data = json.loads((proof/'data.json').read_text())
        specific = (report['last_layers'] == 4 and report['only_text_head_trained'] and report['zero_adapter_feature_logits_exact']
            and report['all_component_audits_passed'] and report['past_loss_targets'] == 0
            and report['audio_loss_targets'] == 0 and report['base_audio_parameters_frozen']
            and set(report['trainable_names']) == {'text_linear.lora_A.weight','text_linear.lora_B.weight'}
            and reload['schema'] == 'dl-context-result-head-reload-v1'
            and reload['all_feature_logits_exact'] and reload['actual'] == report['expected_logits']
            and reload['data_sha256'] == report['data_sha256']
            and data['schema'] == 'dl-context-result-head-data-v1' and data['all_component_audits_passed']
            and data['past_loss_targets'] == data['audio_loss_targets'] == 0
            and bool(data['components']) and all(c['passed'] for c in data['components']))
    elif report['schema'] == 'dl-context-actual-adapter-proof-v1':
        data = json.loads((proof/'data.json').read_text())
        preflight = proof/'approved-temporal-audit.json'
        gate = json.loads(preflight.read_text())
        specific = (report['last_layers'] == 4 and report['optimizer_steps'] >= 8
            and report['zero_adapter_native_batch_logits_exact'] and report['all_actual_temporal_audits_passed']
            and report['all_component_audits_passed'] and report['past_loss_targets'] == report['audio_loss_targets'] == 0
            and report['base_audio_parameters_frozen'] and report['actual_temporal_audit_sha256'] == sha256(preflight)
            and gate['passed'] and gate['data_sha256'] == report['data_sha256']
            and data['all_component_audits_passed'] and bool(data['components'])
            and all(c['passed'] for c in data['components'])
            and {r['kind'] for r in report['retention_examples']} == {'private','simple','clarify','cancel'}
            and reload['schema'] == 'dl-context-actual-adapter-reload-v1'
            and reload['all_valid_logits_exact'] and reload['actual'] == report['expected_logits']
            and reload['data_sha256'] == report['data_sha256'])
        if report.get('training_state_sha256'):
            specific = (specific and sha256(proof/'training-state.pt') == report['training_state_sha256']
                and reload['training_state_sha256'] == report['training_state_sha256']
                and reload['optimizer_rng_state_restored'] is True)
    elif report['schema'] == 'dl-context-actual-audio49-adapter-proof-v1':
        from evals.context_native_audio49_training import verify_checkpoint as verify_audio49
        verify_audio49(proof)
        specific=True
    elif report['schema'] == 'dl-native-assistant-joint-proof-v1':
        from evals.context_native_joint import verify_inference
        specific = verify_inference(proof)
    elif report['schema'] == 'dl-context-adapter-proof-v1':
        specific = (report['zero_adapter_all_valid_logits_exact'] and report['native_depth_parity']['passed']
            and report['canonical_logits_dtype'] == 'float32' and reload['all_valid_logits_exact'])
    else:
        specific = False
    if (not specific or not report['passed'] or not report['frozen_weights_unchanged']
            or not reload['passed'] or not reload['fresh_process']
            or report['source_commit'] != SOURCE_COMMIT or report['checkpoint_sha256'] != digest
            or reload['checkpoint_sha256'] != digest or report['rank'] != 8 or report['scaling'] != 2.
            or report['voice_sha256'] != sha256(proof / 'voice.pt')
            or report['data_sha256'] != sha256(proof / 'data.json')):
        raise RuntimeError('Unverified or changed Context adapter proof')
    protocol=None
    if report['schema'] in ('dl-context-actual-adapter-proof-v1','dl-context-result-head-proof-v1',
            'dl-native-assistant-joint-proof-v1','dl-context-actual-audio49-adapter-proof-v1'):
        # These legacy schemas were produced only through the audited bootstrap/
        # provided-token-preserving replay. Keep their exact implementation pin.
        protocol=report.get('native_protocol',{'name':'bootstrap-preserve-provided-v1',
            'code_sha256':'bbf0699025b40be9376491334f787b488fe04a9b6262c2509f44888121250320'})
        if (protocol['name']!='bootstrap-preserve-provided-v1' or protocol['code_sha256']!=
                hashlib.sha256(Path(__file__).with_name('context_native_layout.py').read_bytes()).hexdigest()):
            raise RuntimeError('Checkpoint native prefix/span protocol differs from this runtime')
        if protocol.get('attention_backend','math')!='math':
            raise RuntimeError('Unknown checkpoint attention protocol')
        protocol={**protocol,'attention_backend':'math'}
        from evals.context_execution import math_attention
        math_attention()
    original = engine_module.load_model
    original_init=engine_module.Engine.__init__
    calls = []
    def load(*args, **kwargs):
        lm, mimi, spm = original(*args, **kwargs)
        if calls: raise RuntimeError('Unexpected repeated model loading')
        lm.requires_grad_(False)
        if not disabled:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                names = install_adapters(lm, last_layers=report['last_layers'])
            state = load_file(proof / 'adapters.safetensors', device=str(lm.device))
            learned = adapter_parameters(lm)
            if names != report['adapter_modules'] or set(state) != set(learned):
                raise RuntimeError('Cold adapter module/key coverage differs')
            if not zero:
                with torch.no_grad():
                    for n, p in learned.items(): p.copy_(state[n])
            lm.requires_grad_(False).eval()
        calls.append(True)
        return lm, mimi, spm
    engine_module.load_model = load
    if protocol is not None:
        def init_with_protocol(engine,*args,**kwargs):
            original_init(engine,*args,**kwargs)
            from evals.context_native_layout import install
            install(engine)
            engine._dl_checkpoint_native_protocol=dict(protocol)
        engine_module.Engine.__init__=init_with_protocol
    evidence = {'proof': str(proof), 'checkpoint_sha256': digest, 'disabled': disabled, 'zero': zero,
        'installed_before_native_graphs': not disabled, 'voice_codes_sha256': report['voice_sha256'],
        'voice_reference_sha256': report['voice_reference_sha256'], 'quality_qualified': False,
        'native_protocol':protocol}
    try:
        yield evidence
        if not calls: raise RuntimeError('Native engine did not invoke verified model loader')
    finally:
        engine_module.load_model = original
        engine_module.Engine.__init__=original_init
