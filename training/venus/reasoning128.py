"""Isolated text reasoning/routing comparison, not native voice qualification."""
import argparse,gc,json,re,time,subprocess
from pathlib import Path
from evals.venus_reasoning128_plan import messages
from .large_inputs import sha

PLAN_SHA='e124dd24d739d3f0d89b86918f9a8ab28c8f7dc482a218adedb9c2382852072e'


def prompt(plan,row,tokenizer=None):
    parts=messages(plan,row)
    if tokenizer is not None and getattr(tokenizer,'chat_template',None):
        return tokenizer.apply_chat_template(parts,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    # The release has no tokenizer chat_template; its source uses these ChatML delimiters.
    return ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in parts)+'<|im_start|>assistant\n<think>\n\n</think>\n\n'


def main(a):
    import torch
    from transformers import AutoTokenizer,set_seed
    from .native_cache import load_native
    from .inference import load_candidate
    from .retention_evaluate import TRAINING_ROOT,completed_training
    from .isambard_native import guarded,BASE
    from portable.venus.native_app import ensure_guard
    ensure_guard();guarded(a.lease,900)
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=a.source or json.loads(Path(a.lease).read_text())['source']!=a.source:raise ValueError('Actual source-bound reasoning evaluation required')
    planpath=Path(a.plan)
    if sha(planpath)!=PLAN_SHA:raise ValueError('Frozen whole128 reasoning plan required')
    plan=json.loads(planpath.read_text());cases=plan['cases']
    if len(cases)!=128 or len({r['id'] for r in cases})!=128:raise ValueError('No partial reasoning cohort')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    conditions=[('parent646',BASE/'native-training-resume8-v1/main/step-0120.pt','46e84c6d7b1d02ec16563c1d0e666230f8644963d3406b0f095bab97938950aa')]
    for arm in ('expanded-data','rehearsal'):
        proof=completed_training(TRAINING_ROOT,arm);conditions.append((arm,TRAINING_ROOT/arm/'step-0084.pt',proof['checkpoint_sha256']))
    tokenizer=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,local_files_only=True)
    choices=[tokenizer.encode(c,add_special_tokens=False) for c in 'ABCD']
    if any(len(x)!=1 for x in choices):raise ValueError('Single-token choice scoring required')
    summaries={}
    for condition,checkpoint,expected_sha in conditions:
        if sha(checkpoint)!=expected_sha:raise ValueError('Pinned actual checkpoint required')
        model,_=load_native(a.model,language_only=True);llm=model.llm
        profile=load_candidate(llm,checkpoint);llm.eval();rows=[];started=time.monotonic()
        for case in cases:
            ensure_guard();guarded(a.lease,120);set_seed(42)
            actual_prompt=prompt(plan,case,tokenizer)
            encoded=tokenizer(actual_prompt,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                logits=llm(**encoded).logits[0,-1].float()
                selected=logits[torch.tensor([x[0] for x in choices],device='cuda')]
                forced='ABCD'[int(selected.argmax())]
                letter_mass=float(logits.softmax(-1)[torch.tensor([x[0] for x in choices],device='cuda')].sum())
                generated=llm.generate(**encoded,max_new_tokens=24,do_sample=False,
                    eos_token_id=tokenizer.convert_tokens_to_ids('<|im_end|>'),pad_token_id=tokenizer.convert_tokens_to_ids('<|im_end|>'))
            raw=tokenizer.decode(generated[0,encoded['input_ids'].shape[-1]:],skip_special_tokens=True).strip()
            free=raw if re.fullmatch('[ABCD]',raw) else None
            rows.append(dict(id=case['id'],category=case['category'],raw_response=raw,
                free_choice=free,format_valid=free is not None,free_correct=free==case['expected'],
                constrained_choice=forced,constrained_correct=forced==case['expected'],letter_probability_mass=letter_mass,
                input_sha256=__import__('hashlib').sha256(actual_prompt.encode()).hexdigest()))
            (output/(condition+'-progress.json')).write_text(json.dumps(dict(cases=rows,required=128,complete=False),indent=2)+'\n')
        groups={k:[r for r in rows if r['category']==k] for k in sorted({r['category'] for r in rows})}
        summaries[condition]=dict(cases=128,format_valid=sum(r['format_valid'] for r in rows),
            free_correct=sum(r['free_correct'] for r in rows),constrained_correct=sum(r['constrained_correct'] for r in rows),
            per_category={k:dict(cases=len(v),free_correct=sum(r['free_correct'] for r in v),constrained_correct=sum(r['constrained_correct'] for r in v)) for k,v in groups.items()})
        (output/(condition+'.json')).write_text(json.dumps(dict(complete128=True,source=a.source,checkpoint=profile,plan_sha256=PLAN_SHA,cases=rows,summary=summaries[condition],seconds=time.monotonic()-started),indent=2)+'\n')
        del llm,model;gc.collect();torch.cuda.empty_cache()
    report=dict(complete_all3_times128=True,source=a.source,plan_sha256=PLAN_SHA,conditions=summaries,
        text_only=True,multiple_choice=True,constrained_choice_is_not_free_answer=True,
        native_audio_perception_not_tested=True,backend_execution_not_tested=True,adopted=False,
        caveat='Compare all128 including invalid formats. Constrained scores rank letters under a shared prompt; they are not free-form spoken reasoning or successful delegation.')
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','lease','source','plan','output'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
