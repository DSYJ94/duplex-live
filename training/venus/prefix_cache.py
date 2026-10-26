"""One native replay of existing input features; cache the unchanged lower35 layers."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from .native_cache import load_native,training_input_proof
from .lora import native_segmented_forward,suffix_forward

def capture_prefix(llm,example):
    inputs=[];index=len(llm.model.layers)-1
    def hook(module,args,kwargs):
        data={}
        for key in ('attention_mask','position_ids','cache_position','position_embeddings'):
            value=kwargs.get(key)
            data[key]=value.detach().cpu().clone() if torch.is_tensor(value) else tuple(t.detach().cpu().clone() for t in value) if isinstance(value,tuple) else value
        inputs.append(dict(hidden=args[0].detach().cpu().clone(),kwargs=data))
    handle=llm.model.layers[index].register_forward_pre_hook(hook,with_kwargs=True)
    try:
        with torch.no_grad():reference,reference_hidden=native_segmented_forward(llm,example,next(llm.parameters()).device)
    finally:handle.remove()
    result=dict(example,prefix_inputs=inputs,prefix_layers=index,forward_path='native-frozen-prefix-v1')
    with torch.no_grad():actual,actual_hidden=suffix_forward(llm,result,next(llm.parameters()).device)
    error=float((reference-actual).abs().max());hidden_error=float((reference_hidden-actual_hidden).abs().max())
    if error!=0 or hidden_error!=0:raise ValueError('Exact native frozen-prefix parity failed: '+str((error,hidden_error)))
    return result,dict(logit_error=error,hidden_error=hidden_error,positions=len(example['positions']),passed=True)

def main(a):
    root=Path(a.cache);manifest=json.loads((root/'manifest.json').read_text());assert len(manifest['scenes'])==188 and not manifest['proof_only']
    model,_=load_native(a.model,language_only=True);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rows=[];start=time.monotonic()
    for row in manifest['scenes']:
        path=root/row['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        ex=torch.load(path,map_location='cpu',weights_only=False);result,proof=capture_prefix(model.llm,ex)
        dest=out/row['file'];torch.save(result,dest)
        rows.append(dict(row,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),native_prefix_proof=proof))
        print(json.dumps(dict(event='exact_frozen_prefix',completed=len(rows),total=188,elapsed=time.monotonic()-start,id=row['id'],proof=proof)),flush=True)
    updated=dict(manifest,scenes=rows,source_native_cache_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),prefix_layers=len(model.llm.model.layers)-1,prefix_producer=a.source)
    (out/'manifest.json').write_text(json.dumps(updated,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True);p.add_argument('--source',required=True);main(p.parse_args())
