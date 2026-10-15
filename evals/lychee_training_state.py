"""Retain actual TRAIN-only student history, never substitute teacher history."""
import json
from evals.moshi_model_download import sha256


def save(session, folder, name, source_row, frame, receipt, instructions):
    if source_row['split']!='train' or source_row['kind']!='private':
        raise RuntimeError('Only approved private TRAIN history may enter this collection')
    if receipt['status']!='completed' or receipt['received_at_s']>frame['snapshot_elapsed_s']:
        raise RuntimeError('Training state requires an actually received completed result')
    import torch
    def cpu(value):
        if torch.is_tensor(value):return value.detach().cpu().clone()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,list):return [cpu(v) for v in value]
        return value
    payload={'prefix':session.get_prefix(),'system_ids':session.prefix_input_ids.detach().cpu().clone(),
        'audio_cache':cpu(session.export_incremental_audio_cache()),
        'audio_tail':torch.from_numpy(session._audio_tail.copy()),
        'stoken_mapping':session.stoken_mapping.detach().cpu().clone(),
        'next_chunk_pos':session.next_chunk_pos,'native_listening_state':session.get_listening_state(),
        'last_ss_pos':session.last_ss_pos,'instructions':instructions,
        'actual_receipt':receipt,'source_row':source_row,'frame':frame}
    path=folder/(name+'.pt');torch.save(payload,path)
    record={'schema':'dl-lychee-actual-train-history-v1','file':path.name,'sha256':sha256(path),
        'source_scenario':source_row['scenario_id'],'source_split':'train',
        'source_wav_sha256':source_row['wav_sha256'],'source_sidecar_sha256':source_row['sidecar_sha256'],
        'actual_receipt':receipt,'frame':frame,'native_listening_state':session.get_listening_state(),
        'audio_input_tokens':payload['audio_cache']['audio_input_token_count'],
        'native_body_tokens':len(payload['prefix']['input_ids']),
        'body_history':'all actual sampled student text/speech/control tokens; no teacher substitution',
        'context_channel':'actual completed worker result in nonspoken system prefix',
        'backend_context':'causal measured words; not ASR','timing_qualification':False,
        'training_approved':False,'gold_future_targets':'not yet assembled or approved'}
    (folder/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    return record
