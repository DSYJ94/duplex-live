"""Typed counterfactual of actual audio-run context, with/without tool history."""
import argparse
import json
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.recovery_worker import RecordWorld,run_worker
from evals.duplex_tool_context import project_execution


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--capture',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    parent=json.loads((args.capture/'report.json').read_text());source=args.capture/'recovery-train-020'
    case=json.loads((source/'report.json').read_text());old=json.loads((source/'turn-1-backend.json').read_text())
    if (not parent['component_execution_completed'] or case['recognition_wer']!=0 or not case['all_live_user_rows_exact_at_temporal_forward']
            or old['result']['route']!='delegate' or len(old['result']['receipts'])!=1):raise RuntimeError('Missing completed actual source and executed first read')
    receipt=old['result']['receipts'][0]
    calls=[c for event in old['result']['events'] for c in event['output'] if c['type']=='function_call' and c['name']=='lookup_record']
    if len(calls)!=1:raise RuntimeError('Source actual operation ambiguous')
    events=json.loads((source/'events.json').read_text());users=[e['text'] for e in events if e['kind']=='controller_dispatched'][:2]
    if len(users)!=2:raise RuntimeError('Actual user events absent')
    from evals.qwen35_controller import Qwen35ResponseExecutor
    executor=Qwen35ResponseExecutor(args.assets,device='cuda:0',pending_speech_protocol='explicit-actions')
    args.out.mkdir(parents=True,exist_ok=False);results=[]
    for with_history in (False,True):
        world=RecordWorld(case['declared_counterfactual_world']);context=[dict(role='user',text=users[0])]
        revision=world.reserve('turn-1',context.copy());actual=world.read('turn-1',receipt['project'],receipt['field'],expected_revision=revision)
        if actual!=receipt:raise RuntimeError('Reconstructed actual read differs')
        if with_history:context+=project_execution(calls[0],actual)
        context.append(dict(role='user',text=users[1]));revision=world.reserve('turn-2',context.copy())
        if any(v in json.dumps(context) for record in world.records.values() for v in record.values()):raise RuntimeError('Private value leaked into new foreground context')
        try:
            result=run_worker(context,world,'turn-2',expected_revision=revision,model=executor.model_id,dispatch_protocol='state-aware',response_executor=executor)
            row=dict(with_actual_tool_history=with_history,execution_completed=True,route=result['route'],result=result,
                no_duplicate_read=result['route']=='answer' and not result['receipts'],private_values_withheld_from_prompt=True)
        except Exception as error:row=dict(with_actual_tool_history=with_history,execution_completed=False,message=str(error),failure=getattr(error,'result',None))
        results.append(row)
    report=dict(schema='dl-actual-tool-history-ablation-v1',results=results,load=executor.proof,source_report_sha256=sha256(args.capture/'report.json'),
        source_first_backend_sha256=sha256(source/'turn-1-backend.json'),source_events_sha256=sha256(source/'events.json'),
        text_is_actual_captured_asr_not_reference=True,input_is_typed=True,speech_executed=False,remote_api_used=False,optimizer_steps=0,duplex_qualified=False,
        code_sha256=sha256(Path(__file__)))
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
