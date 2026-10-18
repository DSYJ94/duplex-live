"""Actual frozen local model choosing speech control in explicit typed fixtures.

Background task state is a declared fixture, not an executing remote request.
No API calls, speech inference, training or whole-system qualification.
"""
import argparse
import json
from pathlib import Path
from evals.duplex_reasoning_worker import ReasoningWorld,run_reasoning_worker
from evals.moshi_model_download import sha256


def specifications():
    return [
        dict(id='stop-only',user='Stop.',expected_route='silence',expected_consumed=['explanation']),
        dict(id='withdraw-explanation',user='Do not finish that explanation. Stop talking.',expected_route='silence',expected_consumed=['explanation']),
        dict(id='stop-and-revise',user='Stop. I only want the first step in one short sentence.',expected_route='answer',expected_consumed=['explanation']),
        dict(id='cancel-calculation-preserve-speech',user='Cancel the calculation, but finish your explanation.',background=True,expected_route='cancel',expected_consumed=[]),
        dict(id='new-summary-request',user='Do not continue the long version. Give me just its main point.',expected_route='answer',expected_consumed=['explanation']),
        dict(id='new-stop-speech-preserve-calculation',user='Stop the explanation. Keep the calculation running.',background=True,expected_route='silence',expected_consumed=['explanation']),
        dict(id='new-cancel-only-calculation',user='Leave the explanation queued. Cancel only the calculation.',background=True,expected_route='cancel',expected_consumed=[]),
    ]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','kernel-assets','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    from evals.qwen35_controller import Qwen35ResponseExecutor
    executor=Qwen35ResponseExecutor(args.assets,kernel_assets=args.kernel_assets,pending_speech_protocol='explicit-actions',compact_replies=True)
    rows=[]
    def blocked_backend(question):raise RuntimeError('Typed speech-control fixture forbids any remote request')
    for spec in specifications():
        world=ReasoningWorld({},blocked_backend)
        if spec.get('background'):
            world.reserve('calculation');world.tasks['calculation'].update(operation='delegate_reasoning',phase='running')
        context=[dict(role='user',text='Explain in four short sentences how a seed grows into a plant.'),dict(role='user',text=spec['user'])]
        pending=[dict(id='explanation',request_text=context[0]['text'],text='A seed needs water, warmth, and oxygen to begin germination. The root emerges first. A shoot then grows upward. Leaves begin making food using sunlight.')]
        revision=world.reserve('current',context);row=dict(id=spec['id'],background_state_is_fixture=True)
        try:
            result=run_reasoning_worker(context,world,'current',executor.model_id,revision,'state-aware',pending,executor)
            passed=result['route']==spec['expected_route'] and result['consume_pending_speech_ids']==spec['expected_consumed']
            if spec['expected_route']=='silence':passed=passed and result['content']=='' and result['receipts']==[]
            if spec.get('background'):passed=passed and world.tasks['calculation']['cancelled']==(spec['expected_route']=='cancel')
            row.update(execution_completed=True,action_contract_passed=passed,result=result)
        except Exception as error:
            row.update(execution_completed=False,action_contract_passed=False,error_type=type(error).__name__,message=str(error),evidence=getattr(error,'result',None))
        row['world_tasks']=world.tasks;rows.append(row)
        (args.out/(spec['id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    report=dict(schema='dl-typed-speech-control-v1',results=rows,load=executor.proof,all_action_contracts_passed=all(r['action_contract_passed'] for r in rows),
        remote_api_used=False,input_is_typed=True,background_state_is_fixture=True,speech_generation_executed=False,optimizer_steps=0,
        development_diagnostic=True,duplex_qualified=False,code_sha256=sha256(Path(__file__)))
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
