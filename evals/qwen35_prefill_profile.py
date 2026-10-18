"""Reconstruct two actual long controller prompts and profile cold/hot replay.

Native proposals are not dispatched. No API credentials or external requests.
Private per-job compiler caches isolate cold behavior without deleting shared
caches or patching libraries. This is a performance diagnostic, not qualification.
"""
import argparse,copy,cProfile,hashlib,io,json,os,pstats,time
from pathlib import Path
from evals.moshi_model_download import sha256


def capture_request(root,combined):
    from evals.duplex_reasoning_worker import ReasoningWorld,run_reasoning_worker
    suite=json.loads((root/'report.json').read_text());case=suite['results'][0];folder=root/case['scenario'];meta=json.loads((folder/'turn-3-backend.json').read_text())
    old=meta['result']['events'][0]['local_inference'];world=ReasoningWorld({});world.tasks=copy.deepcopy(case['world_tasks'])
    world.tasks['turn-1']['cancelled']=False;world.tasks['turn-3'].update(closed=False,phase='pending')
    context=copy.deepcopy(world.tasks['turn-3']['context'])
    class Capture:
        provider_name='fixture-only-payload-capture'
        def __call__(self,payload):
            self.payload=payload
            return dict(id='capture-only',status='completed',model='fixture-only',output=[dict(type='function_call',name='respond_to_user',arguments=json.dumps(dict(route='answer',content='Capture fixture only.',consume_pending_speech_ids=[])))])
    cap=Capture();run_reasoning_worker(context,world,'turn-3','Qwen/Qwen3.5-9B',world.tasks['turn-3']['revision'],'state-aware',meta['pending_speech'],cap,resumable_speech=meta['resume_target'],bound_resume_target=True,allow_cancel_resume=combined)
    expected=old['compact_reply_action_protocol']
    if cap.payload['local_worker_state']!=expected:raise ValueError('Reconstructed state differs from the actual captured model state')
    if json.loads(cap.payload['input'])!=context:raise ValueError('Reconstructed chronological context changed')
    return cap.payload,dict(source_report_sha256=sha256(root/'report.json'),source_backend_sha256=sha256(folder/'turn-3-backend.json'),original_input_tokens=old['input_tokens'],original_native_action=old['native_model_action'],original_first_token_s=old['first_token_s'],original_wall_s=old['wall_s'],actual_state_exact=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('runs','assets','kernel-assets','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    for name,folder in [('TRITON_CACHE_DIR','private-triton-cache'),('CUDA_CACHE_PATH','private-cuda-cache')]:
        path=a.out/folder;path.mkdir();os.environ[name]=str(path)
    payloads=[(str(j),*capture_request(a.runs/('duplex-live-transport-'+str(j)),j==7519126)) for j in (7519124,7519126)]
    from evals.qwen35_controller import Qwen35ResponseExecutor
    executor=Qwen35ResponseExecutor(a.assets,device='cuda:0',kernel_assets=a.kernel_assets,pending_speech_protocol='explicit-actions',compact_replies=True)
    import torch
    rows=[]
    for job,payload,source in payloads:
        for repeat in range(2):
            name=job+'-'+str(repeat);prof=cProfile.Profile();before=time.monotonic()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as trace:
                prof.enable()
                try:response=executor(copy.deepcopy(payload))
                finally:prof.disable()
            seconds=time.monotonic()-before;prof.dump_stats(str(a.out/(name+'.pstats')));output=io.StringIO();pstats.Stats(prof,stream=output).sort_stats('cumulative').print_stats(35);(a.out/(name+'-python.txt')).write_text(output.getvalue());trace.export_chrome_trace(str(a.out/(name+'-trace.json')))
            averages=trace.key_averages();tables=dict(cpu=averages.table(sort_by='self_cpu_time_total',row_limit=25),cuda=averages.table(sort_by='self_cuda_time_total',row_limit=25));(a.out/(name+'-ops.json')).write_text(json.dumps(tables,indent=2)+'\n')
            inference=response['local_inference']
            if inference['input_tokens']!=source['original_input_tokens'] or inference['native_model_action']!=source['original_native_action']:raise ValueError('Replay did not preserve actual prompt length and native action')
            row=dict(source_job=job,repeat=repeat,source=source,payload_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),first_token_s=inference['first_token_s'],generation_wall_s=inference['wall_s'],generated_tokens=inference['generated_tokens'],profiled_end_to_end_s=seconds,actual_response=response,ops=tables)
            rows.append(row);(a.out/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n');print(json.dumps({k:row[k] for k in ('source_job','repeat','first_token_s','generation_wall_s','generated_tokens')}),flush=True)
    report=dict(schema='dl-qwen35-cold-hot-prefill-profile-v1',results=rows,component_execution_completed=True,actual_external_requests=0,native_actions_dispatched=False,weights_changed=False,private_compiler_caches=True,source_load=executor.proof,profiling_changes_measurement_overhead=True,production_adopted=False,code_sha256=sha256(Path(__file__)))
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
