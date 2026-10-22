"""All40 independent native numerical checks; never an optimizer or behavior pass.

Use the approved old49 reference arithmetic (FP32 state, FP64 linear work) with
temporary wide matrices. Actual BF16 gradient readiness remains a separate gate.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from data.synthetic.native_control_render import sha256


def differences(batch,streamed):
    import torch
    if batch.shape!=streamed.shape or not torch.isfinite(batch).all() or not torch.isfinite(streamed).all():
        raise ValueError('Complete finite native reference outputs required')
    return dict(maximum_absolute_error=float((batch-streamed).abs().max()),
                mean_absolute_error=float((batch-streamed).abs().mean()))


def target_positions(mask,codes,output):
    import torch
    if mask.dtype!=torch.bool or mask.ndim!=1 or codes.dtype!=torch.int64 or codes.shape!=(1,17,len(mask)):
        raise ValueError('Complete native control masks/codes required')
    expected_mask=torch.ones((1,16,len(mask)),dtype=torch.bool,device=mask.device)
    expected_mask[:,1:8,-1]=False;expected_mask[:,9:16,-1]=False
    if (output.mask.dtype!=torch.bool or not torch.equal(output.mask,expected_mask)
            or output.text_mask.dtype!=torch.bool
            or not torch.equal(output.text_mask,torch.ones((1,1,len(mask)),dtype=torch.bool,device=mask.device))):
        raise ValueError('Exact native delay mask required; no arbitrary target subset')
    eligible=(mask&output.mask[0,:8].all(0)).nonzero().flatten()
    spoken=(mask&(codes[0,0]>=5)&output.mask[0,:8].all(0)).nonzero().flatten()
    expected=[int(mask.sum())]+[int(mask.sum())-1]*7
    actual=(mask[None,None]&output.mask[:,:8]).sum((0,2)).tolist()
    if not len(eligible) or not len(spoken) or actual!=expected or not torch.equal(mask&output.text_mask[0,0],mask):
        raise ValueError('Every available text/audio target required; only pinned native delay tail may be unavailable')
    return sorted({int(eligible[0]),int(spoken[0]),int(eligible[-1])}),actual


def load_model(a):
    from evals.context_rollout_conditioning import verify_checkpoint
    from evals.context_spanning_download import SOURCE_COMMIT,MODEL_REVISION,WEIGHT_SHA256
    from evals.personaplex_train import MODEL_HASHES
    identity,_=verify_checkpoint(a.proof)
    assets=json.loads(a.assets.read_text());source=Path(assets['source']);weights=Path(assets['path'])
    if (a.proof.name!='context-training-proof-7510331' or identity['optimizer_steps']!=520
            or assets['source_commit']!=SOURCE_COMMIT or assets['revision']!=MODEL_REVISION
            or subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
            or sha256(weights/'context_spanning_7b.pt')!=WEIGHT_SHA256
            or any(sha256(a.base/name)!=digest for name,digest in MODEL_HASHES.items())):
        raise ValueError('Unchanged accepted native520 and published source/assets required')
    sys.path.insert(0,str(source))
    os.environ.update(CS_BASE_DIR=str(a.base),CS_WEIGHTS_DIR=str(weights),CS_VOICES_DIR=str(weights/'voices'),HF_HUB_OFFLINE='1')
    os.environ['TRITON_LIBCUDA_PATH']=subprocess.check_output([sys.executable,'-m','evals.cuda_linker_path','--out',str(a.out/'driver')],text=True).strip()
    from portable.context.native_audition import configure_cuda_toolkit
    configure_cuda_toolkit()
    import torch
    import contextspan.model.engine as native
    from contextspan.moshi.utils.compile import no_compile,no_cuda_graph
    from evals.context_adapter_load import before_engine
    from evals.context_spanning_probe import audited_loads
    from evals.context_execution import math_attention
    math_attention();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    native.seed_all(42424242);loads=[]
    with before_engine(native,a.proof) as adapter,audited_loads(loads),no_compile(),no_cuda_graph():
        lm,mimi,spm=native.load_model(str(weights/'context_spanning_7b.pt'),'cuda')
    final=[r for r in loads if r['module_identity']==id(lm)]
    if not final or not final[-1]['strict'] or final[-1]['missing'] or final[-1]['unexpected']:
        raise ValueError('Strict native model load required')
    del mimi;torch.cuda.empty_cache()
    return lm,identity,adapter


def run(lm,engine,out,parent,adapter):
    import torch
    from evals.context_training_parity import precise_native
    from evals.native_control_reference import linear_reference
    from evals.context_native_audio49_training import forward
    from evals.context_native_audio49_loss import loss as acoustic_loss
    from evals.context_native_audio49_audit import depth_checks
    from evals.context_teacher_text_balance import loss as text_loss
    from evals.context_execution import snapshot
    from evals.moshi_training_smoke import tensor_hashes
    before=tensor_hashes(lm.named_parameters());torch.cuda.reset_peak_memory_stats()
    manifest=json.loads((engine/'engine-inputs.json').read_text())
    result=dict(schema='dl-native-control40-numerical-v1',created_utc=datetime.now(timezone.utc).isoformat(),
                engine_report_sha256=sha256(engine/'engine-inputs.json'),cases=[],complete40=False,passed=False,
                threshold=.001,causal_prefix_error_required=0.,optimizer_updates=0,new_checkpoint_written=False,
                training_approved=False,behavior_qualified=False,bf16_gradient_approval_required=True,
                parent=parent,adapter=adapter,execution=snapshot(lm),code_sha256=sha256(Path(__file__)),
                reference_code_sha256=sha256(Path('evals/native_control_reference.py')),
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    with torch.no_grad(),precise_native(lm),linear_reference() as reference:
        for record in manifest['cases']:
            saved=torch.load(engine/record['id']/record['file'],weights_only=True,map_location='cpu')
            inputs=saved['inputs'].cuda();codes=saved['codes'][None].cuda()
            mask=saved['mask'].cuda();targets=saved['targets'].cuda()
            bh,bz=lm.forward_codes(inputs);hidden=[];logits=[]
            with lm.streaming(1):
                for call in record['calls']:
                    i,n=call['start'],call['frames'];h,z=lm.forward_codes(inputs[:,:,i:i+n])
                    hidden.append(h.detach().clone());logits.append(z.detach().clone())
            comparisons=dict(hidden=differences(bh,torch.cat(hidden,dim=1)),text_logits=differences(bz,torch.cat(logits,dim=2)))
            changed=inputs.clone();changed[:,0,-1]=(changed[:,0,-1]+1)%32000
            _,future=lm.forward_codes(changed);causal=float((future[:,:,:-1]-bz[:,:,:-1]).abs().max())
            text_ce,detail=text_loss(bz[0,0],targets,mask)
            output=forward(lm,codes);positions,counts=target_positions(mask,codes,output)
            alignment=float((output.text_logits[0,0,mask]-bz[0,0,mask]).abs().max())
            audio_ce,audio_detail=acoustic_loss(output,codes,mask[None])
            depth=depth_checks(lm,codes,positions)
            row=dict(id=record['id'],family=record['family'],comparisons=comparisons,
                     causal_prefix_maximum_error=causal,native_training_text_alignment_max_error=alignment,
                     text_ce=float(text_ce),text_objective=detail,acoustic_ce=float(audio_ce),acoustic_objective=audio_detail,
                     native_depth_parity=depth,assistant_audio_target_counts=counts,
                     unavailable_native_delay_tail_per_head=[0,1,1,1,1,1,1,1],
                     passed=causal==0. and alignment<=.001 and depth['passed'] and all(v['maximum_absolute_error']<=.001 for v in comparisons.values()))
            result['cases'].append(row)
            (out/'numerical.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(id=row['id'],passed=row['passed'],complete=len(result['cases']))),flush=True)
            del saved,inputs,codes,mask,targets,bh,bz,hidden,logits,changed,future,output,text_ce,audio_ce
        result['reference']=dict(reference)
    unchanged=before==tensor_hashes(lm.named_parameters());peak=torch.cuda.max_memory_allocated()
    complete=len(result['cases'])==40 and len({r['id'] for r in result['cases']})==40
    result.update(complete40=complete,all_model_parameters_unchanged=unchanged,peak_allocated_bytes=peak,
                  passed=complete and unchanged and peak<80_000_000_000 and all(r['passed'] for r in result['cases']))
    (out/'numerical.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise ValueError('Complete40 native numerical gate failed')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('cache','engine-inputs','assets','base','proof','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    from evals.native_control_engine import verify
    verify(a.cache,a.engine_inputs)
    a.out.mkdir(parents=True,exist_ok=False)
    lm,parent,adapter=load_model(a)
    result=run(lm,a.engine_inputs,a.out,parent,adapter)
    print(json.dumps(dict(complete40=result['complete40'],passed=result['passed'],optimizer_updates=0,training_approved=False)))


if __name__=='__main__':main()
