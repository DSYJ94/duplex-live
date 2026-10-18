"""Focused function-name grammar proof, preserving valid actual outputs."""
import argparse
import copy
import json
from pathlib import Path
from evals.moshi_model_download import sha256

SOURCES=[('7526367','turn-1'),('7526371','turn-3'),('7526373','turn-2'),('7526377','turn-2'),('7526379','turn-3'),('7526381','turn-1'),('7526381','turn-2'),('7526369','turn-1')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','kernel-assets','runs','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--candidate',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    from evals.duplex_actual_controller import capture
    from evals.failed_controller_capture import capture_failure
    requests=[(j+'-'+t,*capture(a.runs/('duplex-live-transport-'+j),t)) for j,t in SOURCES]
    requests.append(('closed-task',*capture_failure(a.runs/'duplex-live-transport-7525920/live-session-0/turn-3-failure.json')))
    from evals.qwen35_controller import Qwen35ResponseExecutor
    from evals.qwen35_prefill_warmup import warm_long_prefill
    from evals.nonempty_speech_contract import NonemptySpeechGenerator
    from training.control.head_adapter import parameter_hashes
    executor=Qwen35ResponseExecutor(a.assets,kernel_assets=a.kernel_assets,pending_speech_protocol='explicit-actions',compact_replies=True)
    warm_long_prefill(executor);candidate=None
    if a.candidate:
        from evals.context_candidate import attach
        candidate=attach(executor,a.candidate,a.assets,972)
    executor.generator=NonemptySpeechGenerator(executor.generator,'cancel-only')
    before=parameter_hashes(dict(executor.model.named_parameters()));rows=[]
    for stage in ('original','function-names'):
        if stage=='function-names':
            from evals.function_name_grammar import FunctionNameGenerator
            executor.generator=NonemptySpeechGenerator(FunctionNameGenerator(executor.generator.base),'cancel-only')
        for identity,payload,source in requests:
            row=dict(id=identity,stage=stage,source=source,completed=False,actual_request=payload)
            try:
                response=executor(copy.deepcopy(payload));actual=response['local_inference']
                row.update(completed=True,response=response,action=actual['native_model_action'],historical_input_length_exact=actual['input_tokens']==source['original_input_tokens'])
                if stage=='original' and not row['historical_input_length_exact']:raise ValueError('Original actual request token length differs')
            except Exception as error:row.update(error_type=type(error).__name__,message=str(error),evidence=getattr(error,'result',None))
            if stage=='original' and identity=='closed-task':
                observed=row.get('response',{}).get('local_inference') or (row.get('evidence') or {}).get('local_generation',{})
                row['historical_input_length_exact']=observed.get('input_tokens')==source['original_input_tokens']
                # Saved failure and this proof use the same frozen controller; the candidate is a separate comparison.
                if candidate is None and not row['historical_input_length_exact']:raise ValueError('Failed request reconstruction changed input length')
            rows.append(row);(a.out/(stage+'-'+identity+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    if before!=parameter_hashes(dict(executor.model.named_parameters())):raise ValueError('Inference changed model weights')
    from evals.capability_fresh import evaluate as evaluate_fresh
    fresh=evaluate_fresh(executor,a.out/'fresh-capabilities')
    report=dict(schema='dl-function-name-grammar-probe-v1',results=rows,candidate=candidate,rows=len(rows),all_parameters_unchanged=True,optimizer_steps=0,actual_external_requests=0,actions_dispatched=False,semantic_review_required=True,generated_outputs_repaired=False,owner_demo_changed=False,production_adopted=False,do_not_train=True,code_sha256=sha256(Path(__file__)),wrapper_sha256=sha256(Path('evals/function_name_grammar.py')),fresh=fresh,requests=executor.generator.base.requests)
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(rows=len(rows),semantic_review_required=True)))


if __name__=='__main__':main()
