"""One native replay of existing input features; cache the unchanged lower35 layers."""
import argparse,hashlib,json,time
import shutil,pickle
from pathlib import Path
import torch
from .native_cache import load_native,training_input_proof
from .lora import native_segmented_forward,suffix_forward

PRIOR_PRODUCER='3fc3ff4491c55edf020fc3e6ed6f23a3e3957c19'


def same_value(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same_value(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same_value(x,y) for x,y in zip(a,b))
    return a==b


def reuse_prefix(path,example,witness,producer,expected_layers=35):
    """Validate trusted producer output; never invent a fresh numerical measurement.

    The pinned producer calls capture_prefix, which raises on ANY numerical
    mismatch, before saving each file. An interrupted whole job can therefore
    leave valid completed entries. The original files/failed whole-job status
    remain intact. A corrupt/incomplete entry is diagnosed and recomputed.
    """
    if producer!=PRIOR_PRODUCER:raise ValueError('Unknown prior prefix producer')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=witness['sha256']:raise ValueError('Prior cache hash changed after verified copyback')
    result=torch.load(path,map_location='cpu',weights_only=False)
    expected=dict(example,forward_path='native-frozen-prefix-v1')
    if set(result)!=set(expected)|{'prefix_inputs','prefix_layers'} or any(not same_value(v,result[k]) for k,v in expected.items()):
        raise ValueError('Prior prefix does not match exact approved native inputs')
    if result['prefix_layers']!=expected_layers:raise ValueError('Wrong native layer count')
    if len(result['prefix_inputs'])!=len(example['segments']):raise ValueError('Incomplete prefix segments')
    for entry,segment in zip(result['prefix_inputs'],example['segments']):
        h=entry['hidden']
        if h.shape!=(1,segment['length'],example['embeddings'].shape[-1]) or not torch.isfinite(h).all():raise ValueError('Invalid prefix hidden state')
    return result,dict(passed=True,kind='inherited_pinned_producer_pre_serialization_exact_check',
        producer_source=producer,prior_file_sha256=witness['sha256'],independently_recomputed_this_process=False)

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
    witness=json.loads(Path(a.reuse_witness).read_text()) if a.reuse_witness else None
    if bool(a.reuse_cache)!=bool(witness):raise ValueError('Prior cache requires verified copyback witness')
    prior={r['file']:r for r in witness['files']} if witness else {}
    if witness:
        assert witness['producer_source']==PRIOR_PRODUCER and not witness['original_job_terminal_success']
        assert set(prior)<={r['file'] for r in manifest['scenes']} and len(prior)==len(witness['files'])
    model,_=load_native(a.model,language_only=True);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rows=[];start=time.monotonic()
    for row in manifest['scenes']:
        path=root/row['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        ex=torch.load(path,map_location='cpu',weights_only=False);dest=out/row['file'];reused=False
        if row['file'] in prior:
            old=Path(a.reuse_cache)/row['file']
            try:
                result,proof=reuse_prefix(old,ex,prior[row['file']],witness['producer_source']);reused=True
            except (ValueError,RuntimeError,EOFError,OSError,pickle.UnpicklingError,KeyError,TypeError,IndexError) as error:
                print(json.dumps(dict(event='diagnosed_invalid_partial_cache',id=row['id'],error=str(error),original_preserved=True)),flush=True)
        if reused:shutil.copyfile(old,dest)
        else:result,proof=capture_prefix(model.llm,ex);torch.save(result,dest)
        rows.append(dict(row,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),native_prefix_proof=proof))
        print(json.dumps(dict(event='reused_exact_prefix' if reused else 'exact_frozen_prefix',completed=len(rows),total=188,elapsed=time.monotonic()-start,id=row['id'],proof=proof)),flush=True)
    updated=dict(manifest,scenes=rows,source_native_cache_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),prefix_layers=len(model.llm.model.layers)-1,prefix_producer=a.source)
    (out/'manifest.json').write_text(json.dumps(updated,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True);p.add_argument('--source',required=True);p.add_argument('--reuse-cache');p.add_argument('--reuse-witness');main(p.parse_args())
