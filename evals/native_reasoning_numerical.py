"""Complete32 native numerical approval at the unchanged .001/causality0 bars.

Uses the existing FP32-state/FP64-linear reference and full native depth/target
checks. Text includes every real frame; acoustic masks retain exactly upstream
span-boundary exclusions. No BF16-gradient, optimizer or behavior approval.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from evals import native_control_numerical as original
from evals import native_reasoning_engine as engine
from evals.native_reasoning_source import sha


def run(a):
    import torch
    engine.verify(SimpleNamespace(**{**vars(a),'out':a.engine_inputs}))
    a.out.mkdir(exist_ok=False)
    lm,parent,adapter=original.load_model(SimpleNamespace(**{**vars(a),'base':a.model}))
    from evals.context_training_parity import precise_native
    from evals.native_control_reference import linear_reference
    from evals.context_native_audio49_training import forward
    from evals.context_native_audio49_loss import loss as acoustic_loss
    from evals.context_native_audio49_audit import depth_checks
    from evals.context_teacher_text_balance import loss as text_loss
    from evals.context_execution import snapshot
    from evals.moshi_training_smoke import tensor_hashes
    before=tensor_hashes(lm.named_parameters());torch.cuda.reset_peak_memory_stats()
    manifest=json.loads((a.engine_inputs/'engine-inputs.json').read_text())
    result=dict(schema='dl-native-reasoning32-numerical-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        engine_report_sha256=sha(a.engine_inputs/'engine-inputs.json'),cases=[],complete32=False,passed=False,
        threshold=.001,causal_prefix_error_required=0.,optimizer_updates=0,new_checkpoint_written=False,
        training_approved=False,behavior_qualified=False,bf16_gradient_approval_required=True,
        parent=parent,adapter=adapter,execution=snapshot(lm),code_sha256=sha(Path(__file__)),
        reference_code_sha256=sha(Path('evals/native_control_reference.py')),
        original_numerical_helpers_sha256=sha(Path(original.__file__)),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        acoustic_mask_policy='Exact pinned upstream real-frame mask minus preceding frame for each provided context; no additional exclusions')
    with torch.no_grad(),precise_native(lm),linear_reference()as reference:
        for record in manifest['cases']:
            saved=torch.load(a.engine_inputs/record['id']/record['file'],weights_only=True,map_location='cpu')
            inputs=saved['inputs'].cuda();codes=saved['codes'][None].cuda()
            mask=saved['mask'].cuda();audio_mask=saved['audio_mask'].cuda();targets=saved['targets'].cuda()
            if ((audio_mask&~mask).any()or int(audio_mask.sum())!=int(mask.sum())-record['ownership']['actual_delivered_contexts']):
                raise ValueError('Only the independently approved upstream span boundary may exclude a real acoustic target')
            bh,bz=lm.forward_codes(inputs);hidden=[];logits=[]
            with lm.streaming(1):
                for call in record['calls']:
                    i,n=call['start'],call['frames'];h,z=lm.forward_codes(inputs[:,:,i:i+n])
                    hidden.append(h.detach().clone());logits.append(z.detach().clone())
            comparisons=dict(hidden=original.differences(bh,torch.cat(hidden,dim=1)),
                text_logits=original.differences(bz,torch.cat(logits,dim=2)))
            changed=inputs.clone();changed[:,0,-1]=(changed[:,0,-1]+1)%32000
            _,future=lm.forward_codes(changed);causal=float((future[:,:,:-1]-bz[:,:,:-1]).abs().max())
            text_ce,detail=text_loss(bz[0,0],targets,mask)
            output=forward(lm,codes);positions,counts=original.target_positions(audio_mask,codes,output)
            alignment=float((output.text_logits[0,0,mask]-bz[0,0,mask]).abs().max())
            audio_ce,audio_detail=acoustic_loss(output,codes,audio_mask[None])
            depth=depth_checks(lm,codes,positions)
            row=dict(id=record['id'],family=record['family'],comparisons=comparisons,
                causal_prefix_maximum_error=causal,native_training_text_alignment_max_error=alignment,
                text_ce=float(text_ce),text_objective=detail,acoustic_ce=float(audio_ce),acoustic_objective=audio_detail,
                native_depth_parity=depth,assistant_audio_target_counts=counts,
                real_text_targets=int(mask.sum()),real_acoustic_mask_frames=int(audio_mask.sum()),
                pre_span_acoustic_loss_exclusions=record['ownership']['actual_delivered_contexts'],
                unavailable_native_delay_tail_per_head=[0,1,1,1,1,1,1,1],
                passed=causal==0. and alignment<=.001 and depth['passed']and all(v['maximum_absolute_error']<=.001 for v in comparisons.values()))
            result['cases'].append(row);(a.out/'numerical.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(id=row['id'],passed=row['passed'],complete=len(result['cases']),expected=32)),flush=True)
            del saved,inputs,codes,mask,audio_mask,targets,bh,bz,hidden,logits,changed,future,output,text_ce,audio_ce
        result['reference']=dict(reference)
    unchanged=before==tensor_hashes(lm.named_parameters());peak=torch.cuda.max_memory_allocated()
    complete=[r['id']for r in result['cases']]==[r['id']for r in manifest['cases']]and len(result['cases'])==32
    result.update(complete32=complete,all_model_parameters_unchanged=unchanged,peak_allocated_bytes=peak,
        passed=complete and unchanged and peak<80_000_000_000 and all(r['passed']for r in result['cases']))
    (a.out/'numerical.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise ValueError('Complete32 native numerical gate failed')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('inputs','version','semantic','data','model','codec-source','interleaver-source','cache','assets','proof','engine-inputs','out'):
        p.add_argument('--'+n,type=Path,required=True)
    r=run(p.parse_args());print(json.dumps(dict(complete32=r['complete32'],passed=r['passed'],optimizer_updates=0,training_approved=False)))
