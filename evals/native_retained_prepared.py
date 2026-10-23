"""Strict input-only serialization of the independently reconstructed full49.

This never caches model outputs, gradients or approval decisions. The original
complete source inventory is rehashed on each load, and every serialized target
is checked against the unchanged original approved manifest.
"""
import hashlib
import json
from pathlib import Path

from evals.moshi_model_download import sha256

DATA_SHA='180ffb2c749c3917de0a02f8218d0f8e27bb0067c3b014fd1ef83ae0c583469f'
PREPARED_SHA='ea62a5660e2c9800fbbb39414dc57a1c51cd90acff8186ea0908a885e43e18f6'
REPORT_SHA='38bdf12ec47eb3e3f4412af0632d4876f0708b9d49005e0d908fe9db43f26208'
INVENTORY_SHA='8a09b2544fd0f9e20ca5d7fb388814c0e466441cd7bcbf4396448c3f3865ed96'
VOCAB_SHA='78d4336533ddc26f9acf7250d7fb83492152196c6ea4212c841df76933f18d2d'


def verify_policy(policy,commit):
    import subprocess
    expected=dict(schema='dl-native-retained49-prepared-input-use-v1',data_sha256=DATA_SHA,
                  prepared_sha256=PREPARED_SHA,reconstruction_receipt_sha256=REPORT_SHA,
                  source_inventory=dict(files=10390,bytes=24661548106,inventory_sha256=INVENTORY_SHA),
                  all49_tensor_and_target_hashes_exact=True,
                  model_outputs_gradients_or_validation_decisions_cached=False,
                  loader_sha256=hashlib.sha256(subprocess.check_output(
                      ['git','show',commit+':evals/native_retained_prepared.py'],cwd=Path(__file__).resolve().parents[1])).hexdigest())
    if policy!=expected:raise ValueError('Changed full49 prepared input policy or committed loader')
    return True


def validate_batches(data,batches):
    import torch
    from evals.context_native_audio49_data import complete_inventory,digest
    complete_inventory(data['cases'])
    if len(batches)!=49:raise ValueError('All49 serialized input histories required')
    for row,batch in zip(data['cases'],batches):
        if set(batch)!={'codes','text_mask','audio_mask','delayed_inputs','record'} or batch['record']!=row:
            raise ValueError('Changed prepared49 record identity or order')
        codes,tm,am,inputs=(batch[k] for k in ('codes','text_mask','audio_mask','delayed_inputs'))
        if (any(not isinstance(t,torch.Tensor) or t.device.type!='cpu' for t in (codes,tm,am,inputs))
                or codes.dtype!=torch.int64 or codes.ndim!=3 or codes.shape[:2]!=(1,17)
                or inputs.dtype!=torch.int64 or inputs.shape!=codes.shape
                or tm.dtype!=torch.bool or am.dtype!=torch.bool
                or tm.shape!=(1,codes.shape[2]) or am.shape!=tm.shape or (am&~tm).any()
                or int(am.sum())!=row['assistant_audio_targets_per_head']):
            raise ValueError('Changed complete49 input shape, dtype or assistant ownership')
        targets=codes[:,1:9].transpose(1,2)[am]
        if (targets<0).any() or (targets>=2048).any():raise ValueError('Invalid prepared acoustic targets')
        for tensor,key in ((codes,'codes_sha256'),(tm,'text_mask_sha256'),(am,'audio_mask_sha256'),
                           (inputs,'native_input_sha256'),(targets,'assistant_targets_sha256')):
            if digest(tensor)!=row[key]:raise ValueError('Changed serialized49 tensor: '+key)
    return True


def verify_inventory(path,root):
    path,root=Path(path),Path(root)
    if sha256(path)!=INVENTORY_SHA:raise ValueError('Exact original runs inventory required')
    inventory=json.loads(path.read_text());seen=set();total=0
    for row in inventory['files']:
        name=Path(row['path']);source=root/name
        if (name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0]!='runs'
                or str(name) in seen or source.is_symlink()
                or not source.resolve().is_relative_to((root/'runs').resolve())
                or source.stat().st_size!=row['bytes'] or sha256(source)!=row['sha256']):
            raise ValueError('Changed original49 source/archive input: '+str(name))
        seen.add(str(name));total+=row['bytes']
    if len(seen)!=10390 or total!=24661548106:raise ValueError('Complete original runs inventory required')
    return dict(files=len(seen),bytes=total,inventory_sha256=INVENTORY_SHA)


def load(folder,inventory,assets,voice_file,spm):
    import torch
    from evals.context_native_audio49_data import ROOT
    from evals.context_rollout_conditioning import verify_checkpoint
    folder=Path(folder);report_file=folder/'report.json';prepared=folder/'prepared49.pt'
    if (sha256(report_file)!=REPORT_SHA or sha256(prepared)!=PREPARED_SHA
            or sha256(folder/'audio49-data.json')!=DATA_SHA):
        raise ValueError('Exact independently reconstructed prepared49 artifact required')
    report=json.loads(report_file.read_text())
    if (report['passed'] is not True or report['complete49'] is not True or report['original28'] is not True
            or report['cuda_used'] is not False or report['optimizer_updates']!=0
            or report['training_approved'] is not False or report['original_approved_data_sha256']!=DATA_SHA
            or report['prepared_sha256']!=PREPARED_SHA
            or report['loader_code_sha256']!=sha256(Path(__file__).with_name('context_native_audio49_data.py'))
            or report['all_serialized_tensor_and_assistant_target_hashes_match_original_approval'] is not True):
        raise ValueError('Complete original49 CPU reconstruction receipt required')
    source_inventory=verify_inventory(inventory,ROOT)
    if torch.serialization.get_unsafe_globals_in_checkpoint(prepared):raise ValueError('Unsafe prepared input globals')
    saved=torch.load(prepared,map_location='cpu',weights_only=True)
    if set(saved)!={'data','batches'}:raise ValueError('Input-only prepared artifact required')
    data,batches=saved['data'],saved['batches']
    encoded=json.dumps(data,indent=2)+'\n'
    if hashlib.sha256(encoded.encode()).hexdigest()!=DATA_SHA:raise ValueError('Original49 manifest changed')
    identity,parent=verify_checkpoint(ROOT/'runs/context-training-proof-7510331')
    if (identity!=data['parent'] or sha256(voice_file)!=parent['voice_sha256']
            or hashlib.sha256(spm.serialized_model_proto()).hexdigest()!=VOCAB_SHA
            or Path(assets).resolve()!=(ROOT/'runs/context-spanning-assets-7454205/manifest.json').resolve()):
        raise ValueError('Prepared49 parent, voice, vocabulary or assets changed')
    validate_batches(data,batches)
    policy=dict(schema='dl-native-retained49-prepared-input-use-v1',data_sha256=DATA_SHA,
                prepared_sha256=PREPARED_SHA,reconstruction_receipt_sha256=REPORT_SHA,
                source_inventory=source_inventory,all49_tensor_and_target_hashes_exact=True,
                model_outputs_gradients_or_validation_decisions_cached=False,
                loader_sha256=sha256(Path(__file__)))
    return data,batches,policy
