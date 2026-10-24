"""Inspect the pinned saved export without Torch or executable deserialization.

Only primitive pickle opcodes in the checkpoint's metadata prefix are parsed;
inspection ends before tensor/storage objects. Safetensors layout follows
https://github.com/safetensors/safetensors#format. No tensor values are loaded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickletools
import struct
import zipfile

from evals.native_reasoning_saved_stage_audit import PINNED,PILOT,SOURCE


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate object key')
        result[key]=value
    return result


def primitive_prefix(data):
    """A primitive-only stack interpreter, never pickle.load/loads/Unpickler."""
    if len(data)>2_000_000:raise ValueError('Bounded checkpoint index required')
    mark=object();stack=[];memo={}
    def marked():
        index=next((i for i in range(len(stack)-1,-1,-1) if stack[i] is mark),None)
        if index is None:raise ValueError('Unbalanced primitive pickle mark')
        values=stack[index+1:];del stack[index:];return values
    def assign(target,values):
        if type(target) is not dict or len(values)%2:raise ValueError('Primitive dictionary required')
        for key,value in zip(values[::2],values[1::2]):
            if type(key) is not str or key in target:raise ValueError('Unique string keys required')
            target[key]=value
    def prefix():
        if not stack or type(stack[0]) is not dict:return None
        result=dict(stack[0])
        if len(stack)>1:
            if stack[1] is mark:
                pending=stack[2:]
                if any(x is mark for x in pending):return None
                # Lists/dicts are memoized before being populated. Wait until
                # the NEXT root key, rather than returning an empty or partial
                # parameter_names list as soon as EMPTY_LIST is encountered.
                if len(pending)%2!=1 or pending[-1]!='adapters':return None
                assign(result,pending[:-1])
            elif stack[1:]!=['adapters']:return None
        else:return None
        required=['schema','metadata','completed_steps','parameter_names']
        if list(result)!=required:return None
        if (result['schema']!='dl-context-training-state-v1' or type(result['metadata']) is not dict
                or type(result['completed_steps']) is not int
                or type(result['parameter_names']) is not list
                or any(type(x) is not str for x in result['parameter_names'])):
            raise ValueError('Exact primitive training-state prefix required')
        # JSON encoding rejects cycles and nonfinite metadata; no object hooks.
        json.dumps(result,allow_nan=False)
        return {key:result[key] for key in required}
    try:
        for number,(opcode,arg,position) in enumerate(pickletools.genops(data)):
            if number>250_000:raise ValueError('Primitive index operation limit exceeded')
            name=opcode.name
            if name=='PROTO':
                if not 2<=arg<=5:raise ValueError('Unsupported checkpoint pickle protocol')
            elif name=='FRAME':pass
            elif name=='MARK':stack.append(mark)
            elif name=='EMPTY_DICT':stack.append({})
            elif name=='EMPTY_LIST':stack.append([])
            elif name in ('BINUNICODE','SHORT_BINUNICODE','BINUNICODE8','UNICODE'):
                stack.append(arg)
            elif name in ('BININT','BININT1','BININT2','LONG','LONG1','LONG4','INT'):
                stack.append(arg)
            elif name in ('FLOAT','BINFLOAT'):
                if not math.isfinite(arg):raise ValueError('Nonfinite primitive metadata')
                stack.append(arg)
            elif name=='NONE':stack.append(None)
            elif name in ('NEWTRUE','NEWFALSE'):stack.append(name=='NEWTRUE')
            elif name in ('BINPUT','LONG_BINPUT','PUT'):
                if arg in memo:raise ValueError('Rebound primitive memo')
                memo[arg]=stack[-1]
            elif name=='MEMOIZE':memo[len(memo)]=stack[-1]
            elif name in ('BINGET','LONG_BINGET','GET'):stack.append(memo[arg])
            elif name=='SETITEM':
                value=stack.pop();key=stack.pop();assign(stack[-1],[key,value])
            elif name=='SETITEMS':
                values=marked();assign(stack[-1],values)
            elif name=='APPEND':
                value=stack.pop()
                if type(stack[-1]) is not list:raise ValueError('Primitive list required')
                stack[-1].append(value)
            elif name=='APPENDS':
                values=marked()
                if type(stack[-1]) is not list:raise ValueError('Primitive list required')
                stack[-1].extend(values)
            else:
                raise ValueError('Unsupported opcode before metadata prefix: '+name)
            result=prefix()
            if result is not None:return result
    except (IndexError,KeyError,StopIteration,RecursionError) as error:
        raise ValueError('Malformed primitive metadata prefix') from error
    raise ValueError('Incomplete primitive metadata prefix')


def checkpoint_prefix(path,expected_sha256):
    if sha(path)!=expected_sha256:raise ValueError('Checkpoint bytes changed')
    with zipfile.ZipFile(path) as archive:
        entries=[r for r in archive.infolist() if r.filename.endswith('/data.pkl')]
        if len(entries)!=1 or not 0<entries[0].file_size<=2_000_000:
            raise ValueError('One bounded checkpoint index required')
        return primitive_prefix(archive.read(entries[0]))


def adapter_layout(path,expected_sha256):
    path=Path(path)
    if sha(path)!=expected_sha256:raise ValueError('Adapter bytes changed')
    with path.open('rb') as stream:
        raw=stream.read(8)
        if len(raw)!=8:raise ValueError('Missing safetensors length')
        size=struct.unpack('<Q',raw)[0]
        if not 2<=size<=1_000_000 or size+8>path.stat().st_size:
            raise ValueError('Bounded safetensors header required')
        raw=stream.read(size)
    if not raw.startswith(b'{'):raise ValueError('JSON object header required')
    header=json.loads(raw,object_pairs_hook=unique_object)
    if type(header) is not dict:raise ValueError('Tensor map required')
    tensors={};ranges=[];total=0
    for name,row in header.items():
        if name=='__metadata__':
            if type(row) is not dict or any(type(k) is not str or type(v) is not str for k,v in row.items()):
                raise ValueError('String-only safetensors metadata required')
            continue
        if type(row) is not dict or set(row)!={'dtype','shape','data_offsets'} or row['dtype']!='F32':
            raise ValueError('Exact FP32 adapter layout required')
        shape=row['shape'];offsets=row['data_offsets']
        if (type(shape) is not list or len(shape)!=2 or any(type(v) is not int or v<=0 for v in shape)
                or type(offsets) is not list or len(offsets)!=2
                or any(type(v) is not int or v<0 for v in offsets)
                or offsets[1]-offsets[0]!=math.prod(shape)*4):
            raise ValueError('Invalid FP32 matrix bounds')
        tensors[name]=row;total+=math.prod(shape);ranges.append(offsets)
    cursor=0
    for start,end in sorted(ranges):
        if start!=cursor:raise ValueError('Adapter data overlap or hole')
        cursor=end
    if cursor!=path.stat().st_size-8-size:raise ValueError('Unindexed adapter bytes')
    return dict(tensors=tensors,parameter_count=total)


def inspect_export(training):
    root=Path(training)
    state=checkpoint_prefix(root/'training-state.pt',PINNED[PILOT+'/training/training-state.pt'])
    layout=adapter_layout(root/'adapters.safetensors',PINNED[PILOT+'/training/adapters.safetensors'])
    meta=state['metadata'];names=state['parameter_names'];recipe=meta['recipe']
    if (state['completed_steps']!=8 or meta['repo_commit']!=SOURCE or len(names)!=26
            or len(set(names))!=26 or set(names)!=set(layout['tensors'])
            or layout['parameter_count']!=1_894_400 or recipe['rank']!=8
            or recipe['scaling']!=2. or recipe['last_layers']!=4 or recipe['lr']!=2e-7
            or recipe['retention_kl_weight']!=1. or recipe['retention_hidden_mse_weight']!=10.):
        raise ValueError('Pinned stage source, optimizer recipe or adapter scope changed')
    a={};b={}
    for name,row in layout['tensors'].items():
        if '.lora_A.' in name and row['shape'][0]==8:a[name.split('.lora_A.')[0]]=row
        elif '.lora_B.' in name and row['shape'][1]==8:b[name.split('.lora_B.')[0]]=row
        else:raise ValueError('Unexpected adapter tensor or rank')
    if len(a)!=13 or set(a)!=set(b):raise ValueError('All13 LoRA module pairs required')
    return dict(schema='dl-native-saved-export-metadata-v1',source_commit=SOURCE,
        at_utc=datetime.now(timezone.utc).isoformat(),
        implementation_sha256=sha(Path(__file__)),
        training_state_sha256=PINNED[PILOT+'/training/training-state.pt'],
        adapter_sha256=PINNED[PILOT+'/training/adapters.safetensors'],
        metadata_sha256=hashlib.sha256(json.dumps(meta,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
        completed_stage_updates=state['completed_steps'],adapter_parameters=layout['parameter_count'],
        adapter_tensors=len(names),adapter_modules=len(a),rank=recipe['rank'],scaling=recipe['scaling'],
        effective_learning_rate=recipe['lr'],retention_kl_weight=recipe['retention_kl_weight'],
        retention_hidden_mse_weight=recipe['retention_hidden_mse_weight'],tensor_values_loaded=False,
        optimizer_state_restored=False,rng_restored=False,gpu_execution=False,
        training_or_behavior_approved=False,model_adopted=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=inspect_export(args.training)
    with args.out.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
