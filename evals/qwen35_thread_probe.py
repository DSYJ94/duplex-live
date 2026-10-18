"""Matched 2/1/2-thread latency measurement; no output or model changes."""
import argparse
import copy
import json
from pathlib import Path
import statistics
from evals.duplex_actual_controller import capture
from evals.moshi_model_download import sha256

SOURCES=[('7520283','turn-1'),('7520293','turn-1'),('7520295','turn-2'),('7520299','turn-3'),('7520301','turn-2')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','kernel-assets','runs','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(2)
    from evals.qwen35_controller import Qwen35ResponseExecutor
    from evals.qwen35_prefill_warmup import warm_long_prefill
    from training.control.head_adapter import parameter_hashes
    executor=Qwen35ResponseExecutor(a.assets,kernel_assets=a.kernel_assets,pending_speech_protocol='explicit-actions',compact_replies=True)
    warm_long_prefill(executor)
    parameters=dict(executor.model.named_parameters());before=parameter_hashes(parameters)
    versions={n:x._version for n,x in parameters.items()}
    requests=[(j,t,*capture(a.runs/('duplex-live-transport-'+j),t)) for j,t in SOURCES]
    rows=[];controls={}
    for stage,threads in [('reference',2),('one-thread',1),('return-reference',2)]:
        torch.set_num_threads(threads)
        assert torch.get_num_threads()==threads
        for job,task,payload,source in requests:
            for repeat in range(2):
                name=f'{stage}-{job}-{task}-{repeat}';row=dict(stage=stage,threads=threads,job=job,task=task,repeat=repeat,completed=False,source=source)
                try:
                    response=executor(copy.deepcopy(payload));actual=response['local_inference']
                    identity={k:actual[k] for k in ('native_model_action','input_tokens','generated_tokens','raw_text')}
                    key=(job,task)
                    if stage=='reference' and repeat==0:controls[key]=identity
                    row.update(completed=True,response=response,exact=identity==controls[key],wall_s=actual['wall_s'],first_token_s=actual['first_token_s'])
                except Exception as error:row.update(error_type=type(error).__name__,message=str(error),evidence=getattr(error,'result',None))
                rows.append(row);(a.out/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    unchanged=before==parameter_hashes(parameters) and versions=={n:x._version for n,x in parameters.items()}
    if not unchanged:raise ValueError('Frozen parameters changed')
    by={(r['stage'],r['job'],r['task'],r['repeat']):r for r in rows}
    ratios=[]
    for job,task in SOURCES:
        a0,b,a1=[by[(s,job,task,1)] for s in ('reference','one-thread','return-reference')]
        if all(r.get('completed') and r['exact'] for r in (a0,b,a1)):
            ratios.append(dict(job=job,task=task,speed_ratio=((a0['wall_s']+a1['wall_s'])/2)/b['wall_s'],reference_drift_ratio=a0['wall_s']/a1['wall_s']))
    exact=len(rows)==30 and all(r.get('completed') and r['exact'] for r in rows)
    speed=statistics.median(r['speed_ratio'] for r in ratios) if ratios else None
    stable=len(ratios)==5 and all(.9<=r['reference_drift_ratio']<=1.1 for r in ratios)
    report=dict(schema='dl-thread-scheduling-probe-v1',results=rows,paired_steady_ratios=ratios,all_outputs_exact=exact,all_parameters_and_versions_unchanged=unchanged,steady_median_speed_ratio=speed,reference_drift_within_ten_percent=stable,component_promotion_gate_passed=exact and stable and speed>=1.2,source_load=executor.proof,code_sha256=sha256(Path(__file__)),actual_external_requests=0,actions_dispatched=False,optimizer_steps=0,owner_demo_changed=False,production_adopted=False,do_not_train=True)
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','source_load')}))


if __name__=='__main__':main()
