"""Bounded frozen reviewer batching; complete inputs and explicit EOS/padding."""
import time
import uuid
from .local_reasoner_reference import LocalReasoner,finish_record


def complete_ids(padded,eos,pad):
    eos=set(eos if isinstance(eos,(list,tuple,set))else[eos])
    ids=list(padded)
    for index,value in enumerate(ids):
        if value in eos:
            if any(token!=pad for token in ids[index+1:]):
                raise ValueError('Only actual generation padding may follow EOS')
            return ids[:index+1]
    return ids


def encode(processor,questions):
    if not 1<=len(questions)<=4 or any(not isinstance(q,str)or not q.strip()for q in questions):
        raise ValueError('One to four complete reviewer requests required')
    tokenizer=processor.tokenizer
    if tokenizer.pad_token_id is None:raise ValueError('Existing declared padding token required')
    tokenizer.padding_side='left'
    messages=[[dict(role='system',content='Answer the user question accurately and concisely. If information is missing or uncertain, say so.'),
               dict(role='user',content=question)]for question in questions]
    lengths=[];valid=[];original_ids=[]
    for index,message in enumerate(messages):
        item=processor.apply_chat_template(message,tokenize=True,add_generation_prompt=True,
             return_dict=True,return_tensors='pt',enable_thinking=False,truncation=False)
        length=int(item['input_ids'].shape[-1]);lengths.append(length)
        original_ids.append(item['input_ids'][0].tolist())
        if length<=4096:valid.append(index)
    batch=None
    if valid:
        batch=processor.apply_chat_template([messages[index]for index in valid],tokenize=True,
             add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False,padding=True,truncation=False)
        actual=[int(value)for value in batch['attention_mask'].sum(-1)]
        if actual!=[lengths[index]for index in valid]:raise ValueError('Batching changed complete prompt lengths')
        for index,mask in enumerate(batch['attention_mask']):
            values=mask.tolist()
            if values!=sorted(values):raise ValueError('Actual left padding required')
            ids=batch['input_ids'][index].tolist()
            if [token for token,active in zip(ids,values)if active]!=original_ids[valid[index]]:
                raise ValueError('Batched conversations differ from original complete token sequences')
            if any(token!=tokenizer.pad_token_id for token,active in zip(ids,values)if not active):
                raise ValueError('Exact declared padding tokens required')
    return batch,valid,lengths


class CorpusBatchReasoner(LocalReasoner):
    def answer_batch(self,questions):
        self.guard();torch=self.torch
        encoded,valid,lengths=encode(self.processor,questions)
        results=[dict(text='',input_tokens=length,input_error='complete_input_exceeds_reviewer_context',
                      generated_tokens=0,ended_with_EOS=False,generation_budget_exhausted=False)
                 for length in lengths]
        if not valid:return results
        width=encoded['input_ids'].shape[-1];encoded={key:value.to('cuda')for key,value in encoded.items()}
        pad=self.processor.tokenizer.pad_token_id;eos=self.model.generation_config.eos_token_id
        torch.cuda.synchronize();started=time.monotonic()
        with torch.inference_mode():
            output=self.model.generate(**encoded,max_new_tokens=512,do_sample=False,pad_token_id=pad)
        torch.cuda.synchronize();elapsed=time.monotonic()-started;invocation=uuid.uuid4().hex;self.guard()
        for index,original in enumerate(valid):
            padded=output[index,width:].tolist();ids=complete_ids(padded,eos,pad)
            text=self.processor.decode(ids,skip_special_tokens=True)
            results[original]=dict(finish_record(ids,eos,text,elapsed,lengths[original]),batch_size=len(valid),
                                   batch_wall_seconds=elapsed,batch_invocation=invocation,
                                   raw_padded_generated_token_ids=padded,generated_token_ids=ids,
                                   actual_left_padding=True)
        return results
