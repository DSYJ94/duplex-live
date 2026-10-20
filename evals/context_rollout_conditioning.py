"""Bind corrective TRAIN conditioning to an exactly verified trained collector."""
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def verify_checkpoint(proof):
    from evals.context_spanning_download import SOURCE_COMMIT
    proof=Path(proof)
    report=json.loads((proof/'report.json').read_text());reload=json.loads((proof/'reload-report.json').read_text())
    if report['schema']=='dl-context-actual-audio49-adapter-proof-v1':
        from evals.context_native_audio49_training import verify_checkpoint as verify_audio49
        return verify_audio49(proof)
    protocol=report.get('native_protocol')
    if (report['schema']!='dl-context-actual-adapter-proof-v1' or report['passed'] is not True
            or report['frozen_weights_unchanged'] is not True or report['base_audio_parameters_frozen'] is not True
            or report['all_actual_temporal_audits_passed'] is not True
            or report['all_component_audits_passed'] is not True
            or report['zero_adapter_native_batch_logits_exact'] is not True
            or report['source_commit']!=SOURCE_COMMIT or report['last_layers']!=4
            or report['rank']!=8 or report['scaling']!=2.
            or report['past_loss_targets']!=0 or report['audio_loss_targets']!=0
            or report['optimizer_steps']<8 or reload['schema']!='dl-context-actual-adapter-reload-v1'
            or reload['passed'] is not True
            or reload['fresh_process'] is not True or reload['all_valid_logits_exact'] is not True
            or reload['optimizer_rng_state_restored'] is not True
            or reload['actual']!=report['expected_logits'] or protocol is None
            or protocol.get('attention_backend')!='math' or protocol.get('name')!='bootstrap-preserve-provided-v1'
            or protocol['code_sha256']!=sha256(Path(__file__).with_name('context_native_layout.py'))):
        raise RuntimeError('Collector proof or its native protocol is not verified')
    for filename,key in (('adapters.safetensors','checkpoint_sha256'),('training-state.pt','training_state_sha256'),
            ('data.json','data_sha256'),('voice.pt','voice_sha256'),
            ('approved-temporal-audit.json','actual_temporal_audit_sha256')):
        if sha256(proof/filename)!=report[key]:raise RuntimeError('Collector artifact changed')
    gate=json.loads((proof/'approved-temporal-audit.json').read_text())
    data=json.loads((proof/'data.json').read_text())
    if (gate['schema']!='dl-context-actual-temporal-audit-v1' or gate['passed'] is not True
            or gate['optimizer_steps']!=0 or gate['data_sha256']!=report['data_sha256']
            or not gate['components'] or any(c['passed'] is not True or c['causal_prefix_maximum_error']!=0.
                or any(not 0<=v['maximum_absolute_error']<=.001 for v in c['comparisons'].values())
                for c in gate['components'])
            or data['all_component_audits_passed'] is not True
            or data['past_loss_targets']!=0 or data['audio_loss_targets']!=0):
        raise RuntimeError('Collector temporal audit or data scope is not verified')
    if (reload['checkpoint_sha256']!=report['checkpoint_sha256']
            or reload['training_state_sha256']!=report['training_state_sha256']
            or reload['data_sha256']!=report['data_sha256']):
        raise RuntimeError('Actual conditioning differs from the trained collector or voice')
    identity={'proof':str(proof),'report_sha256':sha256(proof/'report.json'),
        'reload_report_sha256':sha256(proof/'reload-report.json'),
        'checkpoint_sha256':report['checkpoint_sha256'],'training_state_sha256':report['training_state_sha256'],
        'native_protocol':protocol,'optimizer_steps':report['optimizer_steps']}
    return identity,report


def verify(loading):
    adapter=loading.get('adapter')
    if (not adapter or adapter['zero'] or adapter['disabled'] or loading.get('input_warmup')
            or not loading.get('layout') or not adapter['installed_before_native_graphs']):
        raise RuntimeError('Trained corrective conditioning needs an enabled verified collector')
    identity,report=verify_checkpoint(adapter['proof'])
    if (adapter['native_protocol']!=report['native_protocol']
            or loading['layout'].get('native_protocol')!=report['native_protocol']
            or adapter['checkpoint_sha256']!=report['checkpoint_sha256']
            or adapter['voice_codes_sha256']!=report['voice_sha256']):
        raise RuntimeError('Actual conditioning differs from the trained collector or voice')
    return identity
