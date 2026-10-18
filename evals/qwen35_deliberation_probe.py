"""Development-only native deliberation comparison on actual saved speech turns.

No media service changes, API calls, training, or physical playback acceptance.
Published tokenizer supplies the thinking delimiters. Bounded failures stay failures.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def after_thinking(tokens,closing):
    if len(closing)!=1:raise ValueError('Expected one published closing-thinking token')
    positions=[i for i,t in enumerate(tokens) if t==closing[0]]
    if len(positions)!=1:raise ValueError('Bounded native deliberation did not close exactly once')
    result=tokens[positions[0]+1:]
    if not result:raise ValueError('No final action after native deliberation')
    return result


class DeliberatingGenerator:
    def __init__(self,model,tokenizer):self.model=model;self.tokenizer=tokenizer
    def request(self,messages,tools,assistant_prefill=''):
        import torch
        from transformers.generation.streamers import BaseStreamer
        if assistant_prefill!='<tool_call>\n':raise ValueError('Diagnostic only compares the native function-call path')
        prompt=self.tokenizer.apply_chat_template(messages,tools=tools,tokenize=False,add_generation_prompt=True,enable_thinking=True)
        if not prompt.endswith('<think>\n'):raise ValueError('Published template did not open native deliberation')
        inputs=self.tokenizer(prompt,return_tensors='pt').to(self.model.device)
        if inputs['input_ids'].shape[1]>4096:raise ValueError('Diagnostic exceeded bounded context')
        clock=[];start=time.perf_counter()
        class Clock(BaseStreamer):
            first=True
            def put(self,value):
                if self.first:self.first=False;return
                clock.append(time.perf_counter()-start)
            def end(self):pass
        output=self.model.generate(**inputs,max_new_tokens=512,do_sample=False,streamer=Clock(),pad_token_id=self.tokenizer.pad_token_id,eos_token_id=self.tokenizer.eos_token_id,use_cache=True)
        generated=output[0,inputs['input_ids'].shape[1]:];torch.cuda.synchronize(self.model.device)
        ids=generated.tolist();raw=self.tokenizer.decode(ids,skip_special_tokens=False)
        try:final=after_thinking(ids,self.tokenizer.encode('</think>',add_special_tokens=False))
        except ValueError as error:
            # Preserve the actual bounded model output with no fabricated closure.
            error.generation=dict(raw_text=raw,generated_tokens=len(ids),wall_s=time.perf_counter()-start,max_new_tokens=512)
            raise
        return dict(raw_text=raw,cleaned_text=self.tokenizer.decode(final,skip_special_tokens=True).strip(),assistant_prefill='',
            requested_immediate_call_prefill_deliberately_not_applied=True,native_thinking=True,max_new_tokens=512,
            input_tokens=inputs['input_ids'].shape[1],generated_tokens=len(ids),wall_s=time.perf_counter()-start,
            first_token_s=clock[0] if clock else None,token_availability_s=clock)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','kernel-assets','runs','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    from evals.qwen35_controller import Qwen35ResponseExecutor
    from evals.duplex_reasoning_worker import ReasoningWorld,run_reasoning_worker
    executor=Qwen35ResponseExecutor(args.assets,kernel_assets=args.kernel_assets,pending_speech_protocol='explicit-actions',compact_replies=True)
    immediate=executor.generator;deliberate=DeliberatingGenerator(executor.model,executor.tokenizer);results=[]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    def blocked_backend(question):raise RuntimeError('Captured control diagnostic forbids external work')
    for label,job,expected in [('stop',7517429,'silence'),('summary',7517431,'answer'),('replacement',7517449,'answer'),('quiet',7517445,'silence')]:
        root=args.runs/('duplex-live-transport-'+str(job));suite=json.loads((root/'report.json').read_text())
        if not suite['component_execution_completed'] or len(suite['results'])!=1:raise ValueError('Capture must be complete and singular')
        case=suite['results'][0];p=root/case['scenario'];captured=json.loads((p/'turn-2-backend.json').read_text())
        if sha(p/'model.wav')!=case['audio_sha256']:raise ValueError('Changed captured audio')
        context=case['world_tasks']['turn-2']['context'];pending=captured['pending_speech']
        if not pending or context[-1]['text']!=captured['request_text']:raise ValueError('Missing actual chronological capture')
        for mode,generator in [('immediate',immediate),('native-deliberation',deliberate)]:
            executor.generator=generator;world=ReasoningWorld({},blocked_backend);revision=world.reserve('turn-2',context)
            row=dict(case=label,mode=mode,source_job=job,source_report_sha256=sha(root/'report.json'),source_backend_sha256=sha(p/'turn-2-backend.json'),expected_route=expected,
                reference_transcript_read=False,background_state_is_empty=True,capture_used_as_development=True)
            try:
                result=run_reasoning_worker(context,world,'turn-2',executor.model_id,revision,'state-aware',pending,executor)
                if result['receipts']:raise ValueError('Speech control diagnostic must not execute external work')
                row.update(execution_completed=True,result=result,route_matches=result['route']==expected)
            except Exception as error:
                row.update(execution_completed=False,error_type=type(error).__name__,message=str(error),evidence=getattr(error,'result',None),generation=getattr(error,'generation',None),route_matches=False)
            results.append(row);(args.out/(label+'-'+mode+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    report=dict(schema='dl-native-deliberation-control-diagnostic-v1',results=results,load=executor.proof,all_completed=all(r['execution_completed'] for r in results),
        model_weights_changed=False,optimizer_steps=0,speech_executed=False,production_adopted=False,duplex_qualified=False,
        developer_case_replay=True,route_match_is_not_semantic_or_intelligence_qualification=True,code_sha256=sha(Path(__file__)))
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(completed=len(results),all_completed=report['all_completed'],production_adopted=False)),flush=True)


if __name__=='__main__':main()
