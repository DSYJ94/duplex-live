"""Four-arm typed diagnostic of a captured pending-speech regression.

Same exact actual user context, pending speech and frozen weights; separately
vary spoken-output instruction and optional kernels. No reference transcript,
training or remote API. The captured request was already used for development.
"""
import argparse
import gc
import json
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.recovery_worker import RecordWorld,run_worker


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','kernel-assets','capture','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--kernel-mode',choices=('torch','fused'),required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    case=json.loads((args.capture/'report.json').read_text());source=json.loads((args.capture/'turn-2-backend.json').read_text())
    if case['recognition_wer']!=0 or source['task_id']!='turn-2' or not source['pending_speech']:raise RuntimeError('Actual exact-ASR pending-speech source missing')
    context=case['world_tasks']['turn-2']['context']
    if len(context)!=2 or any(e['role']!='user' for e in context) or context[-1]['text']!=source['request_text']:raise RuntimeError('This bounded source requires two actual user turns')
    from evals.qwen35_controller import Qwen35ResponseExecutor
    import torch
    results=[];loads=[]
    # Operator selection is cached when Transformers imports its model module.
    # Each kernel arm therefore runs in a separate fresh process.
    for fused in (args.kernel_mode=='fused',):
        executor=Qwen35ResponseExecutor(args.assets,kernel_assets=args.kernel_assets if fused else None,pending_speech_protocol='explicit-actions')
        loads.append(dict(fused=fused,proof=executor.proof))
        for spoken in (False,True):
            executor.spoken_output_contract=spoken
            world=RecordWorld(case['declared_counterfactual_world']);revision=world.reserve('turn-1',context[:1]);world.finish('turn-1',revision)
            revision=world.reserve('turn-2',context)
            row=dict(fused=fused,spoken_output_contract=spoken)
            try:
                result=run_worker(context,world,'turn-2',expected_revision=revision,model=executor.model_id,dispatch_protocol='state-aware',
                    pending_speech=source['pending_speech'],response_executor=executor,scope_read_tools=case['read_tools_scoped_to_user_identified_available_projects'])
                row.update(result=result,execution_completed=True)
            except Exception as error:row.update(execution_completed=False,error_type=type(error).__name__,message=str(error),evidence=getattr(error,'result',None))
            results.append(row);(args.out/('fused-'+str(fused)+'-spoken-'+str(spoken)+'.json')).write_text(json.dumps(row,indent=2)+'\n')
        del executor;gc.collect();torch.cuda.empty_cache()
    report=dict(schema='dl-captured-pending-four-arm-v1',results=results,loads=loads,actual_context=context,
        actual_pending_speech=source['pending_speech'],source_case_sha256=sha256(args.capture/'report.json'),source_backend_sha256=sha256(args.capture/'turn-2-backend.json'),
        development_reused=True,input_is_typed_actual_asr=True,speech_executed=False,remote_api_used=False,optimizer_steps=0,duplex_qualified=False,code_sha256=sha256(Path(__file__)))
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
