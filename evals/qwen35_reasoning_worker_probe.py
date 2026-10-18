"""Actual typed reasoning worker and concurrent local question, before audio."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from evals.moshi_model_download import sha256
from evals.duplex_reasoning_worker import ReasoningWorld,run_reasoning_worker


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('assets','kernel-assets','out'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--foreground',choices=('ordinary','cancel'),default='ordinary')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    from evals.qwen35_controller import Qwen35ResponseExecutor
    executor=Qwen35ResponseExecutor(args.assets,kernel_assets=args.kernel_assets,pending_speech_protocol='explicit-actions')
    world=ReasoningWorld({});question='How many distinct subsets of two, three, five, seven, eleven, thirteen, seventeen, nineteen, twenty three, twenty nine, thirty one, and thirty seven sum to fifty? Each number can be used at most once.'
    context=[dict(role='user',text=question)];revision=world.reserve('background',context)
    before=time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(run_reasoning_worker,context,world,'background',executor.model_id,revision,'state-aware',[],executor)
        deadline=time.monotonic()+30
        while not future.done():
            with world.lock:running=world.tasks['background'].get('operation')=='delegate_reasoning'
            if running:break
            if time.monotonic()>deadline:raise RuntimeError('Local controller did not dispatch reasoning within30s')
            time.sleep(.01)
        if future.done():
            early=future.result();(args.out/'early-result.json').write_text(json.dumps(early,indent=2)+'\n');raise RuntimeError('Expected actual delegation was not dispatched')
        context.append(dict(role='user',text='While you work on that, why does ice float on water?' if args.foreground=='ordinary' else 'Cancel the counting task. Just tell me why a shadow moves during the day.'));revision=world.reserve('foreground',context)
        foreground_started=time.monotonic();foreground=run_reasoning_worker(context,world,'foreground',executor.model_id,revision,'state-aware',[],executor);foreground_ended=time.monotonic()
        (args.out/'foreground.json').write_text(json.dumps(foreground,indent=2)+'\n');background=future.result()
    receipt=background['receipts'][0]
    overlap=foreground_started<receipt['request_finished_monotonic'] and foreground_ended>receipt['request_started_monotonic']
    report=dict(schema='dl-actual-reasoning-worker-concurrent-typed-v1',background=background,foreground=foreground,load=executor.proof,
        foreground_mode=args.foreground,background_delivery_cancelled=not world.deliverable(receipt),actual_world_tasks=world.tasks,local_foreground_overlaps_actual_remote_request=overlap,foreground_seconds=foreground_ended-foreground_started,wall_s=time.monotonic()-before,
        expected_subset_count_for_after_execution_review=19,oracle_supplied_to_models=False,input_is_typed=True,development_only=True,
        speech_executed=False,cancellation_tested=args.foreground=='cancel',agent_actions_executed=False,duplex_qualified=False,optimizer_steps=0,code_sha256=sha256(Path(__file__)))
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
