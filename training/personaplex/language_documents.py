"""Whole-source candidate tokenization, never semantic or training admission."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(turns):
    text='';spans=[]
    if not turns or len(turns)%2:
        raise ValueError('Complete alternating source conversation required')
    for index,turn in enumerate(turns):
        role='user' if index%2==0 else 'assistant'
        if turn.get('role')!=role or not isinstance(turn.get('text'),str) or not turn['text'].strip():
            raise ValueError('Complete original source turns required')
        text+=('User:\n' if role=='user' else 'Assistant:\n')
        start=len(text);text+=turn['text'];end=len(text)
        if role=='assistant':spans.append((start,end))
        text+='\n\n'
    return text,spans


def offsets(text,piece):
    """Check actual immutable-proto surfaces; accept verified char or UTF8 offsets."""
    start,end=int(piece.begin),int(piece.end)
    if 0<=start<=end<=len(text) and text[start:end]==piece.surface:
        return start,end
    raw=text.encode('utf-8')
    try:
        if 0<=start<=end<=len(raw) and raw[start:end].decode('utf-8')==piece.surface:
            return len(raw[:start].decode('utf-8')),len(raw[:end].decode('utf-8'))
    except UnicodeDecodeError:
        pass
    raise ValueError('Tokenizer surface offsets do not match complete original text')


def compile_candidate(row,tokenizer,*,tokenizer_sha256,max_tokens):
    if row.get('admitted_for_training') is not False:
        raise ValueError('This compiler accepts unadmitted source candidates only')
    text,spans=render(row['turns'])
    encoded=tokenizer.encode(text,out_type='immutable_proto')
    tokens=[];mask=[];crossing=0
    for piece in encoded.pieces:
        begin,end=offsets(text,piece)
        inside=any(begin>=start and end<=stop and begin<end for start,stop in spans)
        overlaps=any(begin<stop and end>start for start,stop in spans)
        crossing+=int(overlaps and not inside)
        tokens.append(int(piece.id));mask.append(inside)
    # The one-shot encoding is authoritative. No separately encoded prefix may
    # change the first answer token or accidentally supervise a delimiter/user.
    if tokens!=tokenizer.encode(text,out_type=int):
        raise ValueError('Immutable tokenization differs from actual complete one-shot encoding')
    flags=[]
    if len(tokens)>max_tokens:flags.append('complete_document_exceeds_declared_context')
    if any(value in {0,1,2,3,4} for value in tokens):flags.append('reserved_or_unknown_source_token')
    if not any(mask):flags.append('no_complete_assistant_target_tokens')
    source_sha=hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    return dict(id=row['id'],source_dataset=row['source_dataset'],source_revision=row['source_revision'],
                source_licence=row['source_licence'],source_group_id=row['source_group_id'],
                source_record_ids=row['source_record_ids'],source_sha256=source_sha,
                source_content_sha256=row['content_sha256'],tokenizer_sha256=tokenizer_sha256,
                convention='literal-user-assistant-labels-v1-not-released-native-chat-template',
                text=text,answer_character_spans=spans,tokens=tokens,mask=mask,
                supervised_tokens=sum(mask),boundary_crossing_tokens_excluded=crossing,
                declared_context_limit=max_tokens,structural_flags=flags,fold=None,
                semantic_qualified=False,leakage_qualified=False,native_audio_qualified=False,
                admitted_for_training=False)


def run(inputs,output,tokenizer_path,max_tokens):
    import sentencepiece
    from evals.personaplex_train import MODEL_HASHES
    tokenizer_path=Path(tokenizer_path)
    expected=MODEL_HASHES['tokenizer_spm_32k_3.model']
    if sha(tokenizer_path)!=expected or type(max_tokens) is not int or max_tokens<1:
        raise ValueError('Exact preserved native tokenizer and declared context bound required')
    tokenizer=sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tokenizer.vocab_size()!=32000:raise ValueError('Exact native 32k vocabulary required')
    output=Path(output);output.mkdir(exist_ok=False)
    inventory=[];counts=Counter();seen=set()
    with (output/'candidates.jsonl').open('x') as stream:
        for source in inputs:
            source=Path(source);inventory.append(dict(path=str(source),sha256=sha(source)))
            for line in source.open():
                row=json.loads(line)
                if row['id'] in seen:raise ValueError('Duplicate source identity')
                seen.add(row['id'])
                value=compile_candidate(row,tokenizer,tokenizer_sha256=expected,max_tokens=max_tokens)
                stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
                counts['complete_documents']+=1;counts['tokens']+=len(value['tokens'])
                counts['supervised_tokens']+=value['supervised_tokens']
                counts['boundary_crossing_tokens_excluded']+=value['boundary_crossing_tokens_excluded']
                if not value['structural_flags']:counts['structurally_eligible_candidates']+=1
                for flag in value['structural_flags']:counts[flag]+=1
    report=dict(sources=inventory,tokenizer_sha256=expected,counts=dict(counts),
                candidates_sha256=sha(output/'candidates.jsonl'),declared_context_limit=max_tokens,
                context_bound_not_loaded_model_proof=True,optimizer_updates=0,
                semantic_qualification_performed=False,training_admitted=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--tokenizer',required=True);p.add_argument('--max-tokens',type=int,required=True)
    a=p.parse_args();run(a.inputs,a.output,a.tokenizer,a.max_tokens)
