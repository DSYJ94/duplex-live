"""Typed local text-controller knowledge/tool proof with actual read-only receipts.

No user audio, speech generation, native duplex claim, training or remote API.
The Qwen model selects structured actions; application guards execute them. Tool
format failures are retained rather than repaired or replaced with canned words.
"""
import argparse
import json
from pathlib import Path
import re
import time
from evals.moshi_model_download import sha256
from evals.recovery_worker import RecordWorld,project_identified_by_user,tool


def load_model(assets_path,device='cuda'):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
    from transformers.modeling_utils import no_init_weights
    assets=json.loads(assets_path.read_text());pin=Path(__file__).with_name('assets')/'qwen4b-controller-pins.json'
    if (assets['schema']!='dl-qwen4b-controller-assets-v1' or not assets['passed'] or assets['pins_sha256']!=sha256(pin)
            or assets['code_sha256']!=sha256(Path(__file__).with_name('qwen_controller_assets.py'))):raise RuntimeError('Missing exact text-model asset proof')
    folder=Path(assets['folder'])
    for row in assets['assets']:
        path=Path(row['path'])
        if path.parent!=folder or path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']:raise RuntimeError('Changed local text-model asset')
    config=AutoConfig.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    started=time.perf_counter()
    with no_init_weights():
        model=AutoModelForCausalLM.from_config(config,torch_dtype=torch.bfloat16,attn_implementation='sdpa',trust_remote_code=False)
    model.tie_weights();weights={};index=json.loads((folder/'model.safetensors.index.json').read_text())['weight_map']
    for file in sorted(set(index.values())):
        loaded=load_file(folder/file,device='cpu')
        if set(weights)&set(loaded):raise RuntimeError('Duplicated source tensor')
        weights.update(loaded)
    if set(weights)!=set(index):raise RuntimeError('Model tensor inventory changed')
    aliases={};expected=set(model.state_dict());extra=expected-set(weights)
    if extra:
        if (extra!={'lm_head.weight'} or not config.tie_word_embeddings or 'model.embed_tokens.weight' not in weights
                or model.get_input_embeddings().weight is not model.get_output_embeddings().weight):
            raise RuntimeError('Unaccounted model tensors: '+str(extra))
        weights['lm_head.weight']=weights['model.embed_tokens.weight'];aliases['lm_head.weight']='model.embed_tokens.weight'
    result=model.load_state_dict(weights,strict=True,assign=True);model.tie_weights();state=model.state_dict()
    if result.missing_keys or result.unexpected_keys or set(state)!=set(weights):raise RuntimeError('Text model did not load completely')
    for key,value in state.items():
        if value.shape!=weights[key].shape or value.dtype!=weights[key].dtype or not torch.equal(value,weights[key]):
            raise RuntimeError('Loaded text-model tensor differs from source: '+key)
    if any(p.device.type=='meta' for p in model.parameters()):raise RuntimeError('Unmaterialized model parameter')
    model.eval().requires_grad_(False).to(device);model.tie_weights();torch.cuda.synchronize(device)
    proof=dict(strict=True,all_source_tensors_exact_on_cpu_before_device_copy=True,source_tensor_count=len(index),
        model_state_tensor_count=len(state),explicit_true_tied_aliases=aliases,missing=[],unexpected=[],
        model_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        load_seconds=time.perf_counter()-started,device=str(next(model.parameters()).device),dtype=str(next(model.parameters()).dtype),
        assets_sha256=sha256(assets_path),repo=assets['repo'],revision=assets['revision'])
    return model,tokenizer,proof


def parse_action(text):
    text=text.strip();match=re.fullmatch(r'<tool_call>\s*(\{.*\})\s*</tool_call>',text,re.S)
    value=json.loads(match.group(1) if match else text)
    if set(value)!={'name','arguments'} or not isinstance(value['name'],str) or not isinstance(value['arguments'],dict):
        raise ValueError('Model did not return exactly one structured action')
    return value


class Generator:
    def __init__(self,model,tokenizer):self.model=model;self.tokenizer=tokenizer
    def request(self,messages,tools,assistant_prefill="",logits_processor=None):
        import torch
        from transformers.generation.streamers import BaseStreamer
        clock=[];start=time.perf_counter()
        class Clock(BaseStreamer):
            first=True
            def put(self,value):
                if self.first:self.first=False;return
                clock.append(time.perf_counter()-start)
            def end(self):pass
        text=self.tokenizer.apply_chat_template(messages,tools=tools,tokenize=False,add_generation_prompt=True)
        text+=assistant_prefill
        inputs=self.tokenizer(text,return_tensors='pt').to(self.model.device)
        if inputs['input_ids'].shape[1]>4096:raise RuntimeError('Bounded typed proof exceeded4096 input tokens')
        optional=dict(logits_processor=logits_processor) if logits_processor is not None else {}
        output=self.model.generate(**inputs,**optional,max_new_tokens=192,do_sample=False,streamer=Clock(),
            pad_token_id=self.tokenizer.pad_token_id,eos_token_id=self.tokenizer.eos_token_id,use_cache=True)
        generated=output[0,inputs['input_ids'].shape[1]:];torch.cuda.synchronize(self.model.device);wall=time.perf_counter()-start
        cleaned=self.tokenizer.decode(generated,skip_special_tokens=True).strip()
        return dict(raw_text=self.tokenizer.decode(generated,skip_special_tokens=False),assistant_prefill=assistant_prefill,cleaned_text=assistant_prefill+cleaned,
            input_tokens=inputs['input_ids'].shape[1],generated_tokens=len(generated),wall_s=wall,
            first_token_s=clock[0] if clock else None,token_availability_s=clock)


def functions():
    flat=[tool('respond_to_user','Answer ordinary knowledge or ask for missing identification; no external side effects.',
        {'route':{'type':'string','enum':['answer','clarify']},'content':{'type':'string'}}),
        tool('lookup_record','Read a private field only after the user identifies the project.',{'project':{'type':'string'},'field':{'type':'string'}}),
        tool('cancel_lookup','Cancel identified existing lookup IDs; preserve unrelated work.',{'task_ids':{'type':'array','items':{'type':'string'},'minItems':1}}),
        tool('replace_lookup','Correct an active lookup: atomically cancel selected old IDs and read the identified replacement.',
            {'task_ids':{'type':'array','items':{'type':'string'},'minItems':1},'project':{'type':'string'},'field':{'type':'string'}})]
    return [{'type':'function','function':{k:v for k,v in f.items() if k!='type'}} for f in flat]


def fixtures():
    return [
        dict(id='triangle',user='How many sides does a triangle have?',kind='answer',records={}),
        dict(id='pencil',user='What material in an ordinary pencil makes the mark on paper?',kind='answer',records={}),
        dict(id='lightning',user='Why do I see lightning before hearing thunder?',kind='answer',records={}),
        dict(id='missing-project',user="What's the locker label?",kind='clarify',records={'Elm North':{'locker label':'copper kite'}}),
        dict(id='private-record',user='Please check the locker label for Project Elm North.',kind='delegate',records={'Elm North':{'locker label':'copper kite'}}),
        dict(id='cancel',user='Cancel the lookup for Project Oak Central.',kind='cancel',records={'Oak Central':{'meeting room':'room seven'}},pending='Oak Central'),
        dict(id='correct',user='Actually use Project Pine Central instead. Do not read out the Oak result.',kind='delegate',
            records={'Oak Central':{'meeting room':'room seven'},'Pine Central':{'meeting room':'room twelve'}},pending='Oak Central',replacement=True),
        dict(id='negation',user='Do not look up Project Oak Central. Just tell me why a shadow moves during the day.',kind='answer',records={'Oak Central':{'meeting room':'room seven'}}),
    ]


CONTRACT_EXAMPLES = """
Illustrations of the action contract, NOT conversation history or execution receipts:
Example A: available records contain only Sample East/storage code, but the user asks
'What is the storage code?' without identifying a project. Choose respond_to_user
with route clarify and content 'Which project do you mean?' A sole available name
is still not user identification.
Example B: active task old-demo is a Sample East/storage-code lookup. The user says
'Use Project Sample West instead; do not read the Sample East result.' Choose
replace_lookup with task_ids ['old-demo'], project 'Sample West', field 'storage code'.
This retires the old work AND reads the corrected record; cancel_lookup alone would
leave the user's replacement request unfulfilled.
Example C: the same active task exists and the user says only 'Cancel that lookup.'
Choose cancel_lookup with task_ids ['old-demo']; do not invent a replacement.
Only the ACTUAL state and actual latest user below authorize this request, not these
illustrative names. Do not output a preliminary plan; invoke the selected function.
"""


def trial(generator,fixture,out,prompt_profile='base'):
    world=RecordWorld(fixture['records']);receipts=[];stages=[]
    if fixture.get('pending'):world.read('pending-oak',fixture['pending'],'meeting room')
    revision=world.reserve('current');context=[dict(role='user',text=fixture['user'])]
    available=dict(records={p:list(v) for p,v in world.records.items()},tasks={k:v for k,v in world.tasks.items() if k!='current'})
    instructions=('You control a voice assistant. Choose exactly one provided function, including respond_to_user for an ordinary answer or clarification. '
        'Return only the function call, never a preliminary plan or prose outside it. Answer simple general knowledge from your parameters. '
        'Private fields require an actual lookup; never infer a project from the available names. Ask for missing identification. '
        'Honor negation and corrections. Use replace_lookup when a pending lookup is corrected to a different identified project. '
        'Cancel only identified existing tasks, never current. Keep user-facing content within35 words. Available records contain field NAMES only; '
        'their values have not been provided. Actual client state: '+json.dumps(available))
    if prompt_profile=='contract-examples':instructions+=CONTRACT_EXAMPLES+'\nACTUAL client state: '+json.dumps(available)
    messages=[dict(role='system',content=instructions),dict(role='user',content=fixture['user'])];tools=functions()
    first_prompt=generator.tokenizer.apply_chat_template(messages,tools=tools,tokenize=False,add_generation_prompt=True)
    if any(value in first_prompt for record in world.records.values() for value in record.values()):raise RuntimeError('Private value leaked into initial prompt')
    try:
        first=generator.request(messages,tools);stages.append(first);action=parse_action(first['cleaned_text']);name=action['name'];args=action['arguments']
        if name=='respond_to_user':
            if set(args)!={'route','content'} or args['route'] not in ('answer','clarify') or not isinstance(args['content'],str) or not args['content'].strip():raise ValueError('Invalid reply action')
            route=args['route'];content=args['content']
        else:
            if name in ('lookup_record','replace_lookup') and not project_identified_by_user(context,args.get('project','')):raise ValueError('Private project not identified')
            if name=='lookup_record':
                if set(args)!={'project','field'}:raise ValueError('Unexpected lookup arguments')
                receipt=world.read('current',expected_revision=revision,**args);route='delegate'
            elif name=='cancel_lookup':
                if set(args)!={'task_ids'} or 'current' in args['task_ids']:raise ValueError('Invalid cancellation targets')
                receipt=world.cancel_many(args['task_ids']);route='cancel'
            elif name=='replace_lookup':
                if set(args)!={'task_ids','project','field'}:raise ValueError('Invalid replacement arguments')
                receipt=world.replace_lookup('current',expected_revision=revision,**args);route='delegate'
            else:raise ValueError('Unknown local model action')
            receipts.append(receipt)
            messages.extend([dict(role='assistant',content=None,tool_calls=[dict(type='function',id='local-call',function=dict(name=name,arguments=json.dumps(args)))]),
                dict(role='tool',tool_call_id='local-call',content=json.dumps(receipt)),
                dict(role='user',content='Report this actual receipt using respond_to_user. Do not execute another external operation.')])
            second=generator.request(messages,tools[:1]);stages.append(second);reply=parse_action(second['cleaned_text'])
            if reply['name']!='respond_to_user' or set(reply['arguments'])!={'route','content'} or reply['arguments']['route']!='answer' or not isinstance(reply['arguments']['content'],str) or not reply['arguments']['content'].strip():raise ValueError('Invalid actual-result reply')
            content=reply['arguments']['content']
        action_passed=route==fixture['kind']
        if fixture.get('replacement'):action_passed=action_passed and world.tasks['pending-oak']['cancelled'] and world.tasks['current'].get('project')=='Pine Central'
        record=dict(id=fixture['id'],prompt_profile=prompt_profile,execution_completed=True,route=route,content=content,expected_action_kind=fixture['kind'],action_contract_passed=action_passed,
            actual_receipts=receipts,world_tasks=world.tasks,stages=stages,semantic_review='pending')
    except Exception as error:
        record=dict(id=fixture['id'],prompt_profile=prompt_profile,execution_completed=False,error_type=type(error).__name__,message=str(error),actual_receipts=receipts,world_tasks=world.tasks,stages=stages,action_contract_passed=False)
    (out/(fixture['id']+'.json')).write_text(json.dumps(record,indent=2)+'\n');return record


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--prompt-profile',choices=('base','contract-examples'),default='base');args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    model,tokenizer,proof=load_model(args.assets);generator=Generator(model,tokenizer)
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        warm=generator.request([dict(role='user',content='Say ready.')],[])
        results=[trial(generator,row,args.out,args.prompt_profile) for row in fixtures()]
    report=dict(schema='dl-local-qwen-typed-controller-v1',load=proof,results=results,warmup=warm,
        prompt_profile=args.prompt_profile,local_text_model=True,remote_api_used=False,input_is_typed=True,reference_audio_supplied=False,
        speech_generation_executed=False,training_approved=False,optimizer_steps=0,duplex_system_qualified=False,
        native_speech_model_qualified=False,actual_tool_scope='episode-local read-only records and task lifecycle only',
        code_sha256=sha256(Path(__file__)))
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
