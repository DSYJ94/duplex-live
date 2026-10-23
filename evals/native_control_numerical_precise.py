"""Distinct whole40 higher-precision reference, preserving the original failure.

All old call boundaries, inputs, masks, targets, causality and depth checks run
again. The only new arithmetic is FP64 reference attention; FP64 linear casts
are cached within32GiB without changing their values. Production BF16 is unchanged.
This does not grant BF16-gradient, training, behavior or adoption approval.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from data.synthetic.native_control_render import sha256
from evals.native_control_numerical import load_model,differences,target_positions

ORIGINAL_FAILURE_SHA='ea78de49685707fdae3e5b7dbc1d92ff0ddd3860699e23f152d5b7579c300a3b'


def original_failure(path,engine,parent=None):
    report=json.loads(Path(path).read_text());current=json.loads((engine/'engine-inputs.json').read_text())
    if (sha256(path)!=ORIGINAL_FAILURE_SHA or report['schema']!='dl-native-control40-numerical-v1'
            or report['complete40'] is not True or report['passed'] is not False
            or report['all_model_parameters_unchanged'] is not True or report['threshold']!=.001
            or report['optimizer_updates']!=0 or report['new_checkpoint_written'] is not False
            or report['source_commit']!='7769e7cc06f80e609db3f8ac2611977cd0d7cb66'
            or report['code_sha256']!=sha256(Path(__file__).with_name('native_control_numerical.py'))
            or report['reference_code_sha256']!=sha256(Path(__file__).with_name('native_control_reference.py'))
            or report['engine_report_sha256']!=sha256(engine/'engine-inputs.json')
            or len(report['cases'])!=40 or [r['id'] for r in report['cases']]!=[r['id'] for r in current['cases']]
            or (parent is not None and parent!=report['parent'])):
        raise ValueError('Exact immutable original failed whole40 evidence required')
    return report


def verify_profile(report,folder,engine,parent):
    old=original_failure(folder/'approved-original-failed-numerical.json',engine,parent)
    from evals.native_control_gradient import reference_dependencies
    reference_dependencies(old['source_commit'])
    if (report['independent_reference_profile']!='fp32-state-fp64-linear-attention-fp32-output-v2'
            or report['original_failure_sha256']!=ORIGINAL_FAILURE_SHA
            or report['original_numerical_code_sha256']!=old['code_sha256']
            or report['attention_reference_code_sha256']!=sha256(Path(__file__).with_name('native_control_attention_reference.py'))
            or report['execution']!=old['execution']
            or report['input_inventory_rechecked_and_execution_source_clean'] is not True
            or report['reference']['arithmetic']!='fp32-weights-activations-fp64-linear-fp32-output'
            or report['reference']['allocation_only'] is not True
            or report['reference']['training_approved'] is not False
            or report['reference']['calls']<1 or report['reference']['cast_cache_hits']<1
            or report['reference']['maximum_cached_bytes']!=32*1024**3
            or not 0<report['reference']['peak_cached_bytes']<=32*1024**3
            or report['attention_reference']['calls']<1
            or report['attention_reference']['accumulation_dtype']!='float64'
            or report['attention_reference']['input_output_dtype']!='float32'
            or report['attention_reference']['dropout']!=0.
            or report['attention_reference']['production_or_training_execution_changed'] is not False
            or report['immutable_input_hash_reuse']['all_cached_identities_rechecked'] is not True
            or report['immutable_input_hash_reuse']['validation_results_or_model_outputs_cached'] is not False
            or report['immutable_input_hash_reuse']['unique_hashed_files']<1):
        raise ValueError('Complete explicit higher-precision reference provenance required; original failure must remain intact')
    return True


def run(lm,engine,out,parent,adapter):
    import torch
    from evals.context_training_parity import precise_native
    from evals.native_control_cached_reference import linear_reference
    from evals.native_control_attention_reference import attention_reference
    from evals.context_native_audio49_training import forward
    from evals.context_native_audio49_loss import loss as acoustic_loss
    from evals.context_native_audio49_audit import depth_checks
    from evals.context_teacher_text_balance import loss as text_loss
    from evals.context_execution import snapshot
    from evals.moshi_training_smoke import tensor_hashes
    before=tensor_hashes(lm.named_parameters());torch.cuda.reset_peak_memory_stats()
    manifest=json.loads((engine/'engine-inputs.json').read_text())
    result=dict(schema='dl-native-control40-numerical-fp64-attention-v2',created_utc=datetime.now(timezone.utc).isoformat(),
                engine_report_sha256=sha256(engine/'engine-inputs.json'),cases=[],complete40=False,passed=False,
                threshold=.001,causal_prefix_error_required=0.,optimizer_updates=0,new_checkpoint_written=False,
                training_approved=False,behavior_qualified=False,bf16_gradient_approval_required=True,
                parent=parent,adapter=adapter,execution=snapshot(lm),code_sha256=sha256(Path(__file__)),
                reference_code_sha256=sha256(Path('evals/native_control_cached_reference.py')),
                attention_reference_code_sha256=sha256(Path('evals/native_control_attention_reference.py')),
                original_numerical_code_sha256=sha256(Path('evals/native_control_numerical.py')),
                original_failure_sha256=sha256(out/'approved-original-failed-numerical.json'),
                independent_reference_profile='fp32-state-fp64-linear-attention-fp32-output-v2',
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    with torch.no_grad(),precise_native(lm),linear_reference() as reference,attention_reference() as attention:
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
        result['attention_reference']=dict(attention)
    unchanged=before==tensor_hashes(lm.named_parameters());peak=torch.cuda.max_memory_allocated()
    complete=len(result['cases'])==40 and len({r['id'] for r in result['cases']})==40
    result.update(complete40=complete,all_model_parameters_unchanged=unchanged,peak_allocated_bytes=peak,
                  passed=complete and unchanged and peak<80_000_000_000 and all(r['passed'] for r in result['cases']))
    (out/'numerical.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise ValueError('Complete40 native numerical gate failed')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('cache','engine-inputs','assets','base','proof','out','original-failure'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--deadline-utc',required=True)
    a=p.parse_args();deadline=datetime.fromisoformat(a.deadline_utc.replace('Z','+00:00')).timestamp()
    if not time.time()+60<deadline<time.time()+5*3600:raise ValueError('Finite future numerical deadline required')
    root=Path(__file__).resolve().parents[1];commit=os.environ['DL_TASK_COMMIT']
    def clean():
        if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=commit
                or subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()):
            raise ValueError('Exact clean immutable numerical source required')
    clean()
    def timeout(*_):raise TimeoutError('Complete40 reference deadline reached')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if any(x.strip().isdigit() for x in gpu.splitlines()):raise RuntimeError('GPU occupied; never duplicate active work')
    from evals.native_control_engine import verify
    from evals.immutable_file_hashes import verified_reads
    manifest=json.loads(a.assets.read_text())
    roots=[Path('/projects/u6si/duplex-live/runs'),root/'runs',a.base,Path(manifest['path'])]
    reads=None;success=False
    a.out.mkdir(parents=True,exist_ok=False)
    try:
        with verified_reads(roots) as reads:
            verify(a.cache,a.engine_inputs)
            original_failure(a.original_failure,a.engine_inputs)
            (a.out/'approved-original-failed-numerical.json').write_bytes(a.original_failure.read_bytes())
            lm,parent,adapter=load_model(a)
            original_failure(a.original_failure,a.engine_inputs,parent)
            run(lm,a.engine_inputs,a.out,parent,adapter)
        clean()
        success=True
    finally:
        report=a.out/'numerical.json'
        if reads is not None and report.exists():
            value=json.loads(report.read_text());value['immutable_input_hash_reuse']=reads.receipt()
            value['input_inventory_rechecked_and_execution_source_clean']=success
            value['passed']=value['passed'] and success
            report.write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':main()
