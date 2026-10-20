"""One fixed TRAIN conditioning diagnosis, zero optimizer updates.

Replay approved day/night teacher words with native520 sampled acoustics, then
score the same history with native520 and clock40 in separate fresh processes.
This deliberately forced-text probe is neither reasoning nor training approval.
The existing teacher capture/approval implementations remain unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from evals.moshi_model_download import sha256

TEACHER='teacher-recovery-train-010'
PARENT='7510331'
CHILD='7558824'
ROOT=Path('/projects/u6si/duplex-live')
OLD_APPROVAL='native-old25-native-approval-7544805'


def policy(scope, parent, child):
    from evals.context_native_learning_curve import PINS
    pin=PINS[1]
    if (scope['schema']!='dl-context-all-teacher-scope-audit-v2' or not scope['passed']
            or scope['training_approved'] is not False or scope['optimizer_steps']!=0
            or len(scope['teachers'])!=25 or len({t['id'] for t in scope['teachers']})!=25
            or sum(t['id']==TEACHER and t['source_split']=='train' for t in scope['teachers'])!=1
            or parent['optimizer_steps']!=520 or child['optimizer_steps']!=40
            or child['checkpoint_sha256']!=pin['checkpoint']
            or child['training_state_sha256']!=pin['state']
            or child['warm_start']['checkpoint_sha256']!=parent['checkpoint_sha256']
            or parent['voice_sha256']!=child['voice_sha256']
            or parent['assets_manifest_sha256']!=child['assets_manifest_sha256']):
        raise ValueError('Conditioning diagnosis requires exact complete25 scope and frozen520/40 lineage')
    return dict(teacher=TEACHER,optimizer_updates=0,teacher_text_forced=True,
        agent_acoustics_forced=False,diagnostic_only=True,training_approved=False,
        native_behavior_qualified=False,outputs_must_not_be_trained_on=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('capture','score'),required=True)
    p.add_argument('--checkpoint',choices=(PARENT,CHILD),default=PARENT)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.phase=='capture' and (a.checkpoint!=PARENT or a.capture!=a.out):
        p.error('Only native520 may generate the one fixed diagnostic history')
    a.out.mkdir(parents=True,exist_ok=False)
    import torch
    from safetensors.torch import load_file
    from evals.context_rollout_conditioning import verify_checkpoint
    from evals.context_spanning_download import SOURCE_COMMIT,MODEL_REVISION,WEIGHT_SHA256
    from evals.personaplex_train import MODEL_HASHES
    from evals.context_spanning_probe import audited_loads
    from evals.personaplex_adapters import install_adapters,adapter_parameters
    from evals.context_execution import math_attention
    assets=ROOT/'runs/context-spanning-assets-7454205/manifest.json'
    scope=ROOT/'runs/native-old25-native-scope-7543889/report.json'
    cache=ROOT/'runs/native-old25-codes-7543730'
    parent_identity,parent=verify_checkpoint(ROOT/('runs/context-training-proof-'+PARENT))
    child_identity,child=verify_checkpoint(ROOT/('runs/context-training-proof-'+CHILD))
    rules=policy(json.loads(scope.read_text()),parent,child)
    approval=ROOT/'runs'/OLD_APPROVAL/'report.json'
    if sha256(approval)!=child['retention_teacher_text']['native_input_approval_sha256']:
        raise ValueError('Changed complete25 independent native input approval')
    manifest=json.loads(assets.read_text());source=Path(manifest['source']);weights=Path(manifest['path'])
    if (manifest['source_commit']!=SOURCE_COMMIT or manifest['revision']!=MODEL_REVISION
            or subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
            or manifest['files']['context_spanning_7b.pt']['sha256']!=WEIGHT_SHA256):
        raise ValueError('Changed pinned native source or model identity')
    for n,r in manifest['files'].items():
        if sha256(weights/n)!=r['sha256']:raise ValueError('Changed native assets')
    base=ROOT/'models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309'
    for n,h in MODEL_HASHES.items():
        if sha256(base/n)!=h:raise ValueError('Changed base assets')
    proof=ROOT/('runs/context-training-proof-'+a.checkpoint)
    identity,report=(parent_identity,parent) if a.checkpoint==PARENT else (child_identity,child)
    if sha256(assets)!=report['assets_manifest_sha256']:raise ValueError('Different model manifest')
    os.environ.update(CS_BASE_DIR=str(base),CS_WEIGHTS_DIR=str(weights),CS_VOICES_DIR=str(weights/'voices'))
    os.environ['TRITON_LIBCUDA_PATH']=subprocess.check_output([sys.executable,'-m','evals.cuda_linker_path','--out',str(a.out/'driver')],text=True).strip()
    sys.path.insert(0,str(source))
    from contextspan.model.weights import load_model
    from contextspan.model.sequence_convention import persona_prefix
    from evals.lychee_context import BASE_INSTRUCTIONS
    from evals.context_training_data import load_training
    checkpoint=weights/'context_spanning_7b.pt'
    if torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint):raise ValueError('Unsafe checkpoint globals')
    math_attention();loads=[]
    with audited_loads(loads):lm,mimi,spm=load_model(str(checkpoint),'cuda')
    final=[r for r in loads if r['module_identity']==id(lm)][-1]
    if not final['strict'] or any(r['missing'] or r['unexpected'] for r in loads) or lm.dep_q!=16:
        raise ValueError('Incomplete native load')
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):names=install_adapters(lm,last_layers=report['last_layers'])
    learned=adapter_parameters(lm);state=load_file(proof/'adapters.safetensors',device='cuda')
    if names!=report['adapter_modules'] or set(state)!=set(learned):raise ValueError('Adapter coverage changed')
    with torch.no_grad():
        for n,t in learned.items():
            if state[n].shape!=t.shape or state[n].dtype!=t.dtype or not torch.isfinite(state[n]).all():raise ValueError('Invalid adapter state')
            t.copy_(state[n])
    lm.eval().requires_grad_(False)
    shutil.copyfile(proof/'voice.pt',a.out/'voice.pt')
    voice=torch.load(a.out/'voice.pt',weights_only=True,map_location='cpu')['codes']
    examples,_=load_training([cache],spm,persona_prefix(BASE_INSTRUCTIONS,voice,spm),text_padding='legacy-epad0',request_policy='native-prelead-epad')
    provenance=dict(schema='dl-native-conditioning-probe-v1',policy=rules,phase=a.phase,checkpoint=identity,
        parent=parent_identity,child=child_identity,scope_sha256=sha256(scope),cache_sha256=sha256(cache/'cache.json'),
        source_commit=os.environ['DL_TASK_COMMIT'],implementation_sha256=sha256(Path(__file__)),source_loads=loads,
        native_capture_helper_sha256=sha256(Path(__file__).with_name('context_teacher_engine_inputs.py')))
    if a.phase=='capture':
        from types import SimpleNamespace
        from evals.context_teacher_engine_inputs import run
        args=SimpleNamespace(teacher_engine_native_audio=True,teacher_engine_input_streamed_user=False,
            teacher_engine_input_audit=scope,teacher_engine_input_id=TEACHER,out=a.out,assets=assets)
        (a.out/'conditioning-probe-start.json').write_text(json.dumps(provenance,indent=2)+'\n')
        result=run(lm,mimi,spm,voice,examples,args)
        provenance.update(capture_report_sha256=sha256(a.out/'teacher-text-render-audit.json'),capture_tensor_sha256=result['sha256'],capture_audio_sha256=result['audio_sha256'],structural_capture_passed=result['passed'],independent_quality_review='pending')
    else:
        del mimi
        capture=json.loads((a.capture/'conditioning-probe.json').read_text())
        if (capture['phase']!='capture' or capture['checkpoint']!=parent_identity
                or capture['policy']!=rules or capture['scope_sha256']!=sha256(scope)
                or capture['capture_report_sha256']!=sha256(a.capture/'teacher-text-render-audit.json')
                or capture['capture_tensor_sha256']!=sha256(a.capture/'teacher-text-native-audio.pt')
                or not capture['structural_capture_passed']):raise ValueError('Changed native diagnostic capture')
        tensors=torch.load(a.capture/'teacher-text-native-audio.pt',weights_only=True,map_location='cpu')
        from evals.context_native_teacher_data import load
        _,rows=load(ROOT/'runs'/OLD_APPROVAL/'report.json',scope,assets,a.out/'voice.pt',examples,persona_prefix(BASE_INSTRUCTIONS,voice,spm),spm)
        teacher=next(r for r in rows if r['teacher']['id']==TEACHER)
        if not torch.equal(tensors['text_targets'],teacher['targets']) or not torch.equal(tensors['text_mask'],teacher['mask']):raise ValueError('Source words or masks changed')
        from evals.context_training_parity import precise_native
        from evals.context_native_learning_curve import target_metrics
        metrics={}
        with torch.no_grad(),precise_native(lm):
            for label,inputs in [('approved_teacher_acoustics',teacher['inputs']),('sampled_native520_acoustics',tensors['delayed_inputs'])]:
                _,z=lm.forward_codes(inputs.cuda())
                metrics[label]=target_metrics(z,teacher['mask'].cuda(),teacher['targets'].cuda())
                del z
        provenance.update(metrics=metrics,capture_provenance_sha256=sha256(a.capture/'conditioning-probe.json'),numerical_scope='FP32 batched diagnostic; no new numerical or training approval')
    if sha256(proof/'adapters.safetensors')!=identity['checkpoint_sha256']:raise ValueError('Checkpoint mutated')
    (a.out/'conditioning-probe.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance),flush=True)


if __name__=='__main__':main()
