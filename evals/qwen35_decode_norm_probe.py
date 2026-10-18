"""Matched actual-prompt latency/parity proof for decode-only RMS normalization.

Five saved actual native proposals, two fixed repeats, unchanged greedy decoding.
No action dispatch, new labels, reasoning requests, cache replacement or training.
"""
import argparse,copy,json,os,time,statistics
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.duplex_actual_controller import capture

SOURCES=[('7520283','turn-1'),('7520293','turn-1'),('7520295','turn-2'),('7520299','turn-3'),('7520301','turn-2')]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('runs','assets','kernel-assets','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    for variable,folder in [('TORCHINDUCTOR_CACHE_DIR','private-inductor-cache'),('TRITON_CACHE_DIR','private-triton-cache'),('CUDA_CACHE_PATH','private-cuda-cache')]:
        p=a.out/folder;p.mkdir();os.environ[variable]=str(p)
    requests=[(job,task,*capture(a.runs/('duplex-live-transport-'+job),task)) for job,task in SOURCES]
    from evals.qwen35_controller import Qwen35ResponseExecutor
    from evals.qwen35_prefill_warmup import warm_long_prefill
    executor=Qwen35ResponseExecutor(a.assets,device='cuda:0',kernel_assets=a.kernel_assets,pending_speech_protocol='explicit-actions',compact_replies=True)
    warm_long_prefill(executor)
    from training.control.head_adapter import parameter_hashes
    params=dict(executor.model.named_parameters());before=parameter_hashes(params);versions={n:p._version for n,p in params.items()}
    rows=[];compiled=None
    for mode in ('original','decode-norm-compiled'):
        if mode=='decode-norm-compiled':
            from evals.qwen35_decode_norm import install
            compiled=install(executor.model)
        for job,task,payload,source in requests:
            for repeat in range(2):
                name=mode+'-'+job+'-'+task+'-'+str(repeat);start=time.monotonic()
                response=None;comparisons=None
                try:
                    response=executor(copy.deepcopy(payload));r=response['local_inference']
                    comparisons=dict(action=r['native_model_action']==source['original_native_action'],input_tokens=r['input_tokens']==source['original_input_tokens'],raw_text=r['raw_text']==source['original_raw_text'],generated_tokens=r['generated_tokens']==source['original_generated_tokens'])
                    exact=all(comparisons.values())
                    row=dict(mode=mode,job=job,task=task,repeat=repeat,completed=True,actual_native_output_exact=exact,first_token_s=r['first_token_s'],controller_s=r['wall_s'],generated_tokens=r['generated_tokens'],source=source,response=response)
                    if mode=='original' and not exact:raise ValueError('Original native proposal failed exact reconstruction; comparison invalid')
                except Exception as error:
                    row=dict(mode=mode,job=job,task=task,repeat=repeat,completed=False,error_type=type(error).__name__,message=str(error),evidence=getattr(error,'result',None),source=source,observed_response=response,comparisons=comparisons)
                    if mode=='original':
                        (a.out/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n');raise
                row['end_to_end_s']=time.monotonic()-start;rows.append(row);(a.out/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k in ('mode','job','task','repeat','completed','actual_native_output_exact','first_token_s','controller_s','message')}),flush=True)
    after=parameter_hashes(params)
    if before!=after or versions!={n:p._version for n,p in params.items()}:raise ValueError('A frozen published parameter changed')
    if compiled['decode_calls']<1 or compiled['prefill_calls']<1:raise ValueError('Actual generation did not exercise both compiled decode and original prefill paths')
    original=[r for r in rows if r['mode']=='original'];candidate=[r for r in rows if r['mode']=='decode-norm-compiled']
    accepted=all(r.get('completed') and r.get('actual_native_output_exact') for r in candidate)
    ratios=[b['controller_s']/c['controller_s'] for b,c in zip(original,candidate) if c.get('completed') and c.get('actual_native_output_exact')]
    steady=[b['controller_s']/c['controller_s'] for b,c in zip(original,candidate) if b['repeat']==1 and c.get('completed') and c.get('actual_native_output_exact')]
    median=statistics.median(steady) if steady else None
    report=dict(schema='dl-decode-norm-matched-profile-v1',component_execution_completed=len(rows)==20,source_load=executor.proof,warmup=executor.warmup,results=rows,compiled_norm=compiled,parameter_hashes_before=before,parameter_hashes_after=after,all_source_parameter_hashes_unchanged=True,parameter_versions_unchanged=True,all_candidate_native_outputs_exact=accepted,paired_speed_ratios=ratios,steady_paired_speed_ratios=steady,steady_median_speed_ratio=median,component_promotion_gate_passed=accepted and len(steady)==5 and median>=1.2,actual_external_requests=0,actions_dispatched=False,weights_changed=False,cache_replaced=False,optimizer_steps=0,do_not_train=True,physical_device_test=False,production_adopted=False,prototype_qualified=False,code_sha256=sha256(Path(__file__)))
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(completed=True,all_candidate_native_outputs_exact=accepted,paired_speed_ratios=ratios)),flush=True)

if __name__=='__main__':main()
