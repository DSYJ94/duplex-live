"""Constrain only the Qwen XML function opener to advertised names.

No request text, allowed operation, arguments or generated output is rewritten.
After the opener, original generation resumes and existing strict guards remain.
"""
from bisect import bisect_left
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re


class TokenPieceIndex:
    def __init__(self,pieces):
        normal=defaultdict(list);leading=defaultdict(list)
        for identity,piece in pieces:
            if not isinstance(identity,int) or identity<0 or not isinstance(piece,str):raise ValueError('Invalid tokenizer piece')
            normal[piece].append(identity);leading[piece.lstrip()].append(identity)
        self.maps=(dict(normal),dict(leading));self.keys=tuple(sorted(m) for m in self.maps)

    def allowed(self,openers,prefix):
        clean=prefix.lstrip()
        if any(clean.startswith(opener) for opener in openers):return None
        remaining=[opener[len(clean):] for opener in openers if opener.startswith(clean)]
        if not remaining:raise ValueError('Generated function prefix escaped its declared grammar')
        kind=1 if not clean else 0;mapping=self.maps[kind];keys=self.keys[kind];allowed=set()
        if not clean:allowed.update(mapping.get('',[]))
        for suffix in remaining:
            # Short token pieces which continue a legal opener.
            for end in range(1,len(suffix)+1):allowed.update(mapping.get(suffix[:end],[]))
            # Merged token pieces may finish the opener and also begin its body.
            offset=bisect_left(keys,suffix)
            while offset<len(keys) and keys[offset].startswith(suffix):
                allowed.update(mapping[keys[offset]]);offset+=1
        if not allowed:raise ValueError('Tokenizer cannot represent a legal function continuation')
        return sorted(allowed)


class FunctionNameProcessor:
    def __init__(self,index,tokenizer,names):
        if not names or len(names)!=len(set(names)) or any(re.fullmatch('[a-z_]+',n) is None for n in names):raise ValueError('Expected unique actual XML function names')
        self.index=index;self.tokenizer=tokenizer;self.names=sorted(names);self.openers=['<function='+n+'>' for n in self.names]
        self.prompt_length=None;self.completed=False;self.selected_function=None;self.steps=[];self.masks={}

    def __call__(self,input_ids,scores):
        import torch
        if self.completed:return scores
        if input_ids.shape[0]!=1 or scores.shape[0]!=1:raise ValueError('Function opener grammar requires single-request greedy generation')
        if self.prompt_length is None:self.prompt_length=input_ids.shape[1]
        prefix=self.tokenizer.decode(input_ids[0,self.prompt_length:].tolist(),skip_special_tokens=False,clean_up_tokenization_spaces=False)
        allowed=self.index.allowed(self.openers,prefix)
        if allowed is None:
            self.completed=True;self.selected_function=next(n for n in self.names if prefix.lstrip().startswith('<function='+n+'>'));return scores
        if prefix not in self.masks:
            if max(allowed)>=scores.shape[-1]:raise ValueError('Tokenizer token exceeds model vocabulary')
            mask=torch.ones_like(scores,dtype=torch.bool);mask[0,allowed]=False;self.masks[prefix]=mask
        constrained=scores.masked_fill(self.masks[prefix],float('-inf'))
        if not torch.isfinite(constrained).any():raise ValueError('No finite logit for a valid function continuation')
        self.steps.append(dict(prefix=prefix,allowed_tokens=len(allowed),original_top_id=int(scores.argmax(-1).item()),constrained_top_id=int(constrained.argmax(-1).item())))
        return constrained


class FunctionNameGenerator:
    def __init__(self,base):
        self.base=base;self.tokenizer=base.tokenizer
        ids=sorted(set(self.tokenizer.get_vocab().values()))
        pieces=self.tokenizer.batch_decode([[i] for i in ids],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        self.index=TokenPieceIndex(zip(ids,pieces));self.requests=[]
    def __getattr__(self,name):return getattr(self.base,name)
    def request(self,messages,tools,assistant_prefill=''):
        if assistant_prefill!='<tool_call>\n':raise ValueError('Function-name constraint requires the existing native Qwen prefix')
        names=[t['function']['name'] for t in tools];processor=FunctionNameProcessor(self.index,self.tokenizer,names)
        result=self.base.request(messages,tools,assistant_prefill=assistant_prefill,logits_processor=[processor])
        metadata=dict(scope='advertised-function-opener-only',names=sorted(names),selected_function=processor.selected_function,steps=processor.steps,body_constrained=False,outputs_rewritten=False,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self.requests.append(metadata)
        return dict(result,function_name_constraint=metadata)
