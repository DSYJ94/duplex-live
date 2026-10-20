"""Complete49 zero-update assistant audio ownership/depth audit, never a trainer."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from evals.moshi_model_download import sha256
from evals.context_native_audio49_data import ROOT,load as load_data


def implementations():
    return {n:sha256(Path(__file__).with_name(n)) for n in ('context_native_audio49_audit.py',
        'context_native_audio49_data.py','context_native_audio49_loss.py','personaplex_depth.py',
        'context_training_parity.py','context_native_teacher_data.py','context_teacher_native_approval.py')}



def depth_checks(lm,codes,indices):
    import torch
    from contextspan.moshi.models.lm import _delay_sequence
    from evals.personaplex_depth import native_depth_context
    initial=lm._get_initial_token();delayed=torch.cat([initial,_delay_sequence(lm.delays,codes,initial)],dim=2)
    hidden,_=lm.forward_codes(delayed[:,:,:-1]);rows=[]
    for index in indices:
        targets=delayed[:,:,index+1:index+2];state=hidden[:,index:index+1]
        original=lm.forward_depformer_training(targets,state);stream=[]
        with lm.depformer.streaming(1):
            for q in range(16):stream.append(lm.forward_depformer(q,targets[:,q:q+1],state))
        streamed=torch.cat(stream,dim=1)
        with native_depth_context(lm):matched=lm.forward_depformer_training(targets,state)
        finite=bool(torch.isfinite(matched).all() and torch.isfinite(streamed).all())
        errors=(matched-streamed).abs().amax((0,2,3)).tolist()
        rows.append(dict(delayed_frame=index,published_batch_per_head_max_error=(original-streamed).abs().amax((0,2,3)).tolist(),
            context15_batch_per_head_max_error=errors,finite=finite,passed=finite and all(0<=x<=.001 for x in errors)))
    return dict(passed=all(x['passed'] for x in rows),frames=rows,threshold=.001,dtype='float32',
        compatibility='Unchanged batched depth context15; original published-depth discrepancies retained')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('preflight','audit'),required=True)
    p.add_argument('--preflight',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.phase=='audit' and not a.preflight:p.error('Independent complete49 CPU preflight required')
    a.out.mkdir(parents=True,exist_ok=False)
    import torch
    import sentencepiece
    from safetensors.torch import load_file
    from evals.context_spanning_download import SOURCE_COMMIT,MODEL_REVISION,WEIGHT_SHA256
    from evals.personaplex_train import MODEL_HASHES
    from evals.context_rollout_conditioning import verify_checkpoint
    assets=ROOT/'runs/context-spanning-assets-7454205/manifest.json';manifest=json.loads(assets.read_text())
    source=Path(manifest['source']);weights=Path(manifest['path']);base=ROOT/'models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309'
    if (manifest['source_commit']!=SOURCE_COMMIT or manifest['revision']!=MODEL_REVISION
            or subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
            or manifest['files']['context_spanning_7b.pt']['sha256']!=WEIGHT_SHA256):
        raise ValueError('Changed pinned Context Spanning source/assets')
    for n,r in manifest['files'].items():
        if sha256(weights/n)!=r['sha256']:raise ValueError('Changed native weight or voice assets')
    for n,h in MODEL_HASHES.items():
        if sha256(base/n)!=h:raise ValueError('Changed pinned PersonaPlex assets')
    parent=ROOT/'runs/context-training-proof-7510331';identity,parent_report=verify_checkpoint(parent)
    if identity['optimizer_steps']!=520:raise ValueError('Only accepted native520 supplies audit weights')
    shutil.copyfile(parent/'voice.pt',a.out/'voice.pt')
    voice=torch.load(a.out/'voice.pt',weights_only=True,map_location='cpu')['codes']
    sys.path.insert(0,str(source))
    spm=sentencepiece.SentencePieceProcessor(str(base/'tokenizer_spm_32k_3.model'))
    data,batches=load_data(assets,a.out/'voice.pt',spm,voice)
    data_file=a.out/'audio49-data.json';data_file.write_text(json.dumps(data,indent=2)+'\n')
    settings=dict(schema='dl-native-audio49-audit-settings-v1',source_commit=os.environ['DL_TASK_COMMIT'],
        implementations=implementations(),parent=identity,data_sha256=sha256(data_file),assets_sha256=sha256(assets),
        voice_sha256=sha256(a.out/'voice.pt'),complete49=True,threshold=.001,causal_prefix_error_required=0.,
        optimizer_updates=0,training_approved=False,original28_actual_histories_required=True,
        original25_and_new24_text_losses_required=True,reference_kl_weight=1.,reference_hidden_mse_weight=10.,
        gradient_smoke='Four predeclared longest examples, one per task kind; not an optimizer or behavioral approval',
        numerical_depth_positions='First, first spoken and last available all-eight-head target frame in EACH of all49; unique indices',
        numerical_temporal_positions='Every approved text target in all49; original complete native temporal gates also retained')
    if a.phase=='preflight':
        result=dict(schema='dl-native-audio49-cpu-preflight-v1',passed=True,settings=settings,
            complete_cases=49,all28_actual_component_references_preserved=len(data['original28_actual_components'])==28,
            all49_source_audio_words_and_masks_exact=True,cuda_used=False,optimizer_updates=0,training_approved=False)
        (a.out/'audio49-preflight.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(cpu_complete49_passed=True,cuda_used=False)),flush=True)
        return
    pre=json.loads((a.preflight/'audio49-preflight.json').read_text())
    if (not pre['passed'] or pre['settings']!=settings or pre['complete_cases']!=49 or pre['cuda_used']
            or sha256(a.preflight/'audio49-data.json')!=settings['data_sha256']):
        raise ValueError('Changed independent all49 CPU ownership preflight')
    os.environ.update(CS_BASE_DIR=str(base),CS_WEIGHTS_DIR=str(weights),CS_VOICES_DIR=str(weights/'voices'))
    os.environ['TRITON_LIBCUDA_PATH']=subprocess.check_output([sys.executable,'-m','evals.cuda_linker_path','--out',str(a.out/'driver')],text=True).strip()
    from contextspan.model.weights import load_model
    from evals.context_spanning_probe import audited_loads
    from evals.personaplex_adapters import install_adapters,adapter_parameters,checkpoint_language_layers
    from evals.context_execution import math_attention
    from evals.context_training_parity import precise_native
    from evals.personaplex_depth import native_depth_context
    from evals.context_native_audio49_loss import loss
    from evals.moshi_training_smoke import tensor_hashes
    checkpoint=weights/'context_spanning_7b.pt'
    if torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint):raise ValueError('Unsafe native globals')
    torch.backends.cudnn.enabled=False;math_attention();loads=[]
    with audited_loads(loads):lm,mimi,_=load_model(str(checkpoint),'cuda')
    del mimi;torch.cuda.empty_cache()
    final=[r for r in loads if r['module_identity']==id(lm)][-1]
    if not final['strict'] or any(r['missing'] or r['unexpected'] for r in loads) or lm.dep_q!=16:raise ValueError('Incomplete native load')
    lm.eval().requires_grad_(False)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):modules=install_adapters(lm,last_layers=4)
    learned=adapter_parameters(lm);state=load_file(parent/'adapters.safetensors',device='cuda')
    if modules!=parent_report['adapter_modules'] or set(state)!=set(learned):raise ValueError('Native520 adapter coverage changed')
    with torch.no_grad():
        for n,t in learned.items():
            if state[n].shape!=t.shape or state[n].dtype!=t.dtype or not torch.isfinite(state[n]).all():raise ValueError('Invalid parent adapter')
            t.copy_(state[n])
    frozen={n:p for n,p in lm.named_parameters() if n not in learned}
    before=tensor_hashes(lm.named_parameters());checks=[];gradients=[]
    probes=[max((b for b in batches if b['record']['teacher']['kind']==kind),key=lambda b:b['codes'].shape[-1]) for kind in ['private','simple','clarify','cancel']]
    def forward(codes):
        with native_depth_context(lm):return lm.forward_train(codes)
    torch.cuda.reset_peak_memory_stats()
    with precise_native(lm):
        with torch.no_grad():
            for row in batches:
                codes=row['codes'].cuda();tm=row['text_mask'].cuda();am=row['audio_mask'].cuda();inputs=row['delayed_inputs'].cuda()
                out=forward(codes);_,z=lm.forward_codes(inputs);keep=tm[0]&out.text_mask[0,0]
                alignment=float((z[0,0,keep]-out.text_logits[0,0,keep]).abs().max())
                value,detail=loss(out,codes,am)
                eligible=(am[0]&out.mask[0,:8].all(0)).nonzero().flatten()
                spoken=(am[0]&(codes[0,0]>=5)&out.mask[0,:8].all(0)).nonzero().flatten()
                if not len(eligible) or not len(spoken):raise ValueError('Missing predeclared numerical depth positions')
                indices=sorted(set([int(eligible[0]),int(spoken[0]),int(eligible[-1])]))
                # Existing helper retains the failed published-depth comparison,
                # then requires <=.001 for the unchanged context15 compatibility.
                parity=depth_checks(lm,codes,indices)
                changed=inputs.clone();changed[:,0,-1]=(changed[:,0,-1]+1)%32000
                _,future=lm.forward_codes(changed);causal=float((future[:,:,:-1]-z[:,:,:-1]).abs().max())
                record=dict(id=row['record']['teacher']['id'],group=row['record']['group'],text_alignment_max_error=alignment,
                    native_depth_parity=parity,causal_prefix_maximum_error=causal,finite_acoustic_ce=float(value),objective=detail,
                    passed=alignment<=.001 and causal==0. and parity['passed'])
                checks.append(record);print(json.dumps(record),flush=True)
                (a.out/'numerical-progress.json').write_text(json.dumps(checks,indent=2)+'\n')
                del out,z,future,codes,tm,am,inputs,changed,value
        if len(checks)!=49:raise ValueError('Complete49 numerical execution required')
        if all(x['passed'] for x in checks):
            for row in probes:
                for t in learned.values():t.grad=None
                codes=row['codes'].cuda();am=row['audio_mask'].cuda()
                with checkpoint_language_layers(lm):
                    out=forward(codes);value,detail=loss(out,codes,am);value.backward()
                temporal={n:t for n,t in learned.items() if not n.startswith('text_linear.')}
                good=all(t.grad is not None and torch.isfinite(t.grad).all() for t in temporal.values())
                count=sum(int(torch.count_nonzero(t.grad)) for t in temporal.values() if t.grad is not None)
                head=sum(int(torch.count_nonzero(t.grad)) for n,t in learned.items() if n.startswith('text_linear.') and t.grad is not None)
                if not good or not count or head or any(t.grad is not None for t in frozen.values()):raise ValueError('Unexpected acoustic gradient ownership')
                gradients.append(dict(id=row['record']['teacher']['id'],kind=row['record']['teacher']['kind'],acoustic_ce=float(value.detach()),
                    finite_temporal_gradients=True,nonzero_temporal_gradient_elements=count,text_head_audio_gradient_elements=head,
                    frozen_parameter_gradients=0,objective=detail))
                for t in learned.values():t.grad=None
                del out,value,codes,am
    unchanged=tensor_hashes(lm.named_parameters())==before;peak=torch.cuda.max_memory_allocated()
    result=dict(schema='dl-native-audio49-numerical-gradient-audit-v1',settings=settings,passed=all(x['passed'] for x in checks) and len(gradients)==4 and unchanged and peak<80_000_000_000,
        complete49_numerical_cases=checks,predeclared_longest_kind_gradient_smoke=gradients,source_loads=loads,
        all_model_parameters_unchanged=unchanged,peak_allocated_bytes=peak,optimizer_updates=0,new_checkpoint_written=False,
        cpu_preflight_sha256=sha256(a.preflight/'audio49-preflight.json'),training_approved=False,native_behavior_qualified=False,
        required_before_any_optimizer='New explicit combined-objective policy retaining actual28/old25/new24/KL1hidden10; independent full training-state restore and complete native behavior review after bounded pilot.')
    (a.out/'audio49-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],complete49=len(checks),gradients=len(gradients),optimizer_updates=0)),flush=True)
    if not result['passed']:raise RuntimeError('Complete49 assistant audio numerical/gradient audit failed')


if __name__=='__main__':main()
