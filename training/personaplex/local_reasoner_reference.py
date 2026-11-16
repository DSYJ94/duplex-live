"""Frozen, genuine text-chat reference. No gold answers, tools, training or API.

This is a stronger backend reference, not evidence of native voice competence.
Only the explicitly shared public model directory and pin may be read.
"""
import hashlib,json,time
from pathlib import Path

PIN_SHA='133b06d15a838a44e77a91392ab043788aea8c51915f67891359a6fd132dfaa8'
REVISION='8f33b98937c6031bb49e713dee3e8e185aafd26f'
PUBLIC=Path('/projects/u6ada/remora/context-review-20261116-v1/shadow/models/qwen27b')
PIN=PUBLIC.parent.parent/'qwen27b-model-pin.json'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify_pin(model=PUBLIC,pin=PIN,expected=PIN_SHA,guard=lambda:None):
    model=Path(model);pin=Path(pin)
    if digest(pin)!=expected:raise ValueError('Exact shared public model pin required')
    record=json.loads(pin.read_text())
    if record['repository']!='Qwen/Qwen3.6-27B'or record['revision']!=REVISION:raise ValueError('Exact public model identity required')
    for name,sha in record['files'].items():
        guard();path=model/name
        if Path(name).is_absolute()or '..'in Path(name).parts or not path.is_file()or digest(path)!=sha:raise ValueError('Pinned public file mismatch: '+name)
    return record


def question_rows(path):
    rows=json.loads(Path(path).read_text())
    allowed={'id','kind','question','audio','rate','audio_sha256'}
    if len(rows)!=76 or len({r['id']for r in rows})!=76:raise ValueError('Complete76 distinct questions required')
    if any(set(r)!=allowed or not isinstance(r['question'],str)or not r['question'].strip()for r in rows):raise ValueError('Input-only question schema; no gold/route/scoring fields')
    return rows


def inference_schema(model,path):
    """Headers/meta only; MTP is published auxiliary, officially ignored by HF."""
    from safetensors import safe_open
    import re
    expected={k:list(v.shape)for k,v in model.state_dict().items()};actual={}
    index=json.loads((Path(path)/'model.safetensors.index.json').read_text())['weight_map']
    for file in sorted(set(index.values())):
        with safe_open(str(Path(path)/file),framework='pt',device='cpu')as f:
            for name in f.keys():actual[name]=f.get_slice(name).get_shape()
    missing=sorted(set(expected)-set(actual));extra=sorted(set(actual)-set(expected))
    wrong=sorted(k for k in expected.keys()&actual.keys()if expected[k]!=actual[k])
    ignored=sorted(getattr(model,'_keys_to_ignore_on_load_unexpected',[])or[])
    if missing or wrong or any(not k.startswith('mtp.')or not any(re.match(p,k)for p in ignored)for k in extra):raise ValueError('Incomplete native inference schema')
    return dict(inference_tensors=len(expected),published_tensors=len(actual),missing=missing,mismatched=wrong,official_auxiliary_ignored=extra,official_ignore_patterns=ignored)


def finish_record(output_ids,eos_ids,text,elapsed,input_tokens):
    ids=list(output_ids);eos=set(eos_ids if isinstance(eos_ids,(tuple,list,set))else[eos_ids])
    return dict(text=text,generated_tokens=len(ids),input_tokens=input_tokens,ended_with_EOS=bool(ids and ids[-1]in eos),generation_budget_exhausted=not bool(ids and ids[-1]in eos),wall_seconds=elapsed)


class LocalReasoner:
    def __init__(self,guard,model_path=PUBLIC,pin_path=PIN):
        import torch
        from transformers import AutoConfig,AutoModelForMultimodalLM,AutoProcessor
        if not torch.cuda.is_available()or torch.cuda.device_count()!=1:raise ValueError('One actual allocated GPU required')
        self.guard=guard;self.torch=torch;self.identity=verify_pin(model_path,pin_path,guard=guard)
        config=AutoConfig.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        with torch.device('meta'):meta=AutoModelForMultimodalLM.from_config(config,trust_remote_code=False)
        self.schema=inference_schema(meta,model_path);del meta;guard()
        self.processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForMultimodalLM.from_pretrained(model_path,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,trust_remote_code=False).to('cuda').eval()
        self.model.requires_grad_(False)

    def answer(self,question):
        self.guard();torch=self.torch
        messages=[dict(role='system',content='Answer the user question accurately and concisely. If information is missing or uncertain, say so.'),dict(role='user',content=question)]
        encoded=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False)
        size=encoded['input_ids'].shape[-1]
        if size>4096:raise ValueError('No silent input truncation')
        encoded={k:v.to('cuda')for k,v in encoded.items()};torch.cuda.synchronize();start=time.monotonic()
        with torch.inference_mode():out=self.model.generate(**encoded,max_new_tokens=512,do_sample=False)
        torch.cuda.synchronize();elapsed=time.monotonic()-start;ids=out[0,size:].tolist();self.guard()
        return finish_record(ids,self.model.generation_config.eos_token_id,self.processor.decode(ids,skip_special_tokens=True),elapsed,size)


def run(questions,out,guard,source):
    out=Path(out);out.mkdir(exist_ok=False);rows=question_rows(questions);actor=LocalReasoner(guard)
    metadata=dict(source=source,model=actor.identity['repository'],revision=REVISION,pin_sha256=PIN_SHA,schema=actor.schema,questions_sha256=digest(questions),optimizer_updates=0,external_API_calls=0,condition='actual supported pure text chat; stronger reference, not native voice')
    (out/'identity.json').write_text(json.dumps(metadata,indent=2)+'\n');results=[]
    for row in rows:
        value=dict(id=row['id'],kind=row['kind'],question=row['question'],**actor.answer(row['question']))
        (out/(row['id']+'.json')).write_text(json.dumps(value,indent=2)+'\n');results.append(value)
        print(json.dumps(dict(event='local_reasoner_reference_complete',completed=len(results),required=76)),flush=True)
    (out/'results.json').write_text(json.dumps(dict(metadata,complete76=True,results=results),indent=2)+'\n')


if __name__=='__main__':
    import argparse
    from .inference_guard import FrameGuard
    p=argparse.ArgumentParser()
    for key in ('questions','output','lease'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source',required=True);a=p.parse_args();guard=FrameGuard(a.lease)
    if guard(600)['source']!=a.source:raise ValueError('Exact actual reference source required')
    run(a.questions,a.output,guard,a.source)
