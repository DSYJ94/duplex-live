"""Same learned tensors, function-name-only inference, full development tasks."""
import argparse
import json
from pathlib import Path
from training.control.evaluate import evaluate_suite,verify_baseline,sha
from training.control.curriculum import specifications,evaluate_scene


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','baseline','assets','kernel-assets','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    trained=json.loads((args.candidate/'report.json').read_text())
    if (not trained['component_execution_completed'] or not trained['all_source_parameter_hashes_unchanged']
            or not trained['strict_checkpoint_restore_logits_exact'] or not trained['optimizer_state_restore_exact']):raise ValueError('Candidate lacks the completed strict training/restore proof')
    scenes=[s for s in specifications() if s['split']=='heldout'];baseline,baseline_proof=verify_baseline(args.baseline,scenes,args.assets)
    from evals.qwen35_controller import Qwen35ResponseExecutor
    executor=Qwen35ResponseExecutor(args.assets,kernel_assets=args.kernel_assets,pending_speech_protocol='explicit-actions',compact_replies=True)
    from training.control.head_adapter import load_delta
    from training.control.scoped_head import FunctionNameHead,FunctionNameGenerator
    delta,metadata=load_delta(args.candidate/'candidate-adapter',sha(args.assets),executor.model.device);delta.requires_grad_(False)
    if metadata['weights_sha256']!=trained['adapter_metadata']['weights_sha256']:raise ValueError('Scoped evaluation selected different learned tensors')
    head=FunctionNameHead(executor.model.lm_head,delta);executor.model.lm_head=head;executor.generator=FunctionNameGenerator(executor.generator,head)
    if any(p.requires_grad for p in executor.model.parameters()):raise ValueError('Scoped evaluation has trainable parameters')
    # Prove wrapper-disabled complete outputs against actual frozen results.
    original=json.loads((args.baseline/'heldout/report.json').read_text());expected={r['id']:r for r in original['results']}
    head.enabled=False;parity=[]
    for kind in ('quiet','replace','first','clarify'):
        scene=next(s for s in scenes if s['kind']==kind);result,quality=evaluate_scene(scene,executor);prior=expected[scene['id']]['result']
        keys=('route','content','consume_pending_speech_ids','receipts')
        if any(result[k]!=prior[k] for k in keys):raise ValueError('Disabled scoped wrapper changed original complete worker output')
        parity.append(scene['id'])
    head.enabled=True;summary=evaluate_suite(executor,scenes,args.out/'function-name-only')
    if head.active:raise ValueError('Request-scoped correction leaked out of generation')
    import torch
    from safetensors.torch import load_file
    saved=load_file(str(args.candidate/'candidate-adapter/adapter.safetensors'))
    if any(not torch.equal(value.detach().cpu(),saved[name]) for name,value in delta.state_dict().items()):raise ValueError('Inference modified learned tensors')
    report=dict(schema='dl-controller-function-only-development-v1',component_execution_completed=True,optimizer_steps=0,
        same_trained_tensors=True,trained_source_report_sha256=sha(args.candidate/'report.json'),adapter_weights_sha256=metadata['weights_sha256'],
        disabled_wrapper_complete_worker_parity=parity,baseline=baseline,baseline_source=baseline_proof,function_only=summary,
        correction_applies_only_to_function_name=True,user_text_keyword_router=False,argument_repair=False,
        source_load=executor.proof,reused_cases_are_development=True,new_holdout_qualification=False,production_adopted=False,duplex_qualified=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(baseline=baseline,function_only=summary,production_adopted=False)),flush=True)


if __name__=='__main__':main()
