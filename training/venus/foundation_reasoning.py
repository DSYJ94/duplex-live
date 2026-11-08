"""Add the untouched foundation once; reuse every completed candidate MC output.

The numeric control is new, uses all four conditions, and changes no weights.
This isolates text reasoning/format from audio perception and speech generation.
"""
import argparse,decimal,gc,hashlib,json,re,shutil,subprocess,time
from pathlib import Path
from .reasoning128 import PLAN_SHA,prompt
from .large_inputs import sha
from evals.venus_reasoning16_plan import messages

INPUTS=Path('docs/NATIVE-FOUNDATION-REASONING-INPUTS-2026-11-08.json')
PRIOR_SOURCE='70d39618bceb5155e4a554c679b7039a9c2a7f8e'


def numeric_score(raw,expected):
    numbers=re.findall(r'(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?!\w)',raw)
    valid=len(numbers)==1
    return dict(single_numeric_result=valid,correct=valid and decimal.Decimal(numbers[0])==decimal.Decimal(expected))


def reused_reports(root,inputs,plan,tokenizer):
    reports={}
    if set(inputs['reused_files'])!={'parent646.json','expanded-data.json','rehearsal.json','comparison.json'}:
        raise ValueError('Complete original three-condition report set required')
    for name,digest in inputs['reused_files'].items():
        if sha(root/name)!=digest:raise ValueError('Original reused result hash differs: '+name)
    expected={r['id']:hashlib.sha256(prompt(plan,r,tokenizer).encode()).hexdigest() for r in plan['cases']}
    for condition in ('parent646','expanded-data','rehearsal'):
        r=json.loads((root/(condition+'.json')).read_text())
        if not r['complete128'] or r['source']!=PRIOR_SOURCE or r['plan_sha256']!=PLAN_SHA or len(r['cases'])!=128:
            raise ValueError('Every original source-bound128 result required')
        if {x['id']:x['input_sha256'] for x in r['cases']}!=expected:raise ValueError('Reused actual prompts differ')
        reports[condition]=r
    return reports


def summary(rows):
    groups={k:[r for r in rows if r['category']==k] for k in sorted({r['category'] for r in rows})}
    return dict(cases=len(rows),format_valid=sum(r['format_valid'] for r in rows),free_correct=sum(r['free_correct'] for r in rows),
        constrained_correct=sum(r['constrained_correct'] for r in rows),per_category={k:dict(cases=len(v),free_correct=sum(r['free_correct'] for r in v),constrained_correct=sum(r['constrained_correct'] for r in v)) for k,v in groups.items()})


def main(a):
    import torch
    from transformers import AutoTokenizer,set_seed
    from .native_cache import load_native
    from .inference import load_candidate,MODEL_REVISION
    from .retention_evaluate import TRAINING_ROOT,completed_training
    from .isambard_native import guarded,BASE
    from portable.venus.native_app import ensure_guard
    ensure_guard();guarded(a.lease,600)
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=a.source or json.loads(Path(a.lease).read_text())['source']!=a.source:
        raise ValueError('Actual Git/lease source required')
    inputs=json.loads(INPUTS.read_text());planpath=Path(a.plan);numericpath=Path(a.numeric_plan)
    if sha(planpath)!=PLAN_SHA or sha(numericpath)!=inputs['numeric_plan_sha256'] or inputs['model_revision']!=MODEL_REVISION:
        raise ValueError('Exact frozen full plans and released model revision required')
    plan=json.loads(planpath.read_text());numeric=json.loads(numericpath.read_text())
    if len(plan['cases'])!=128 or len(numeric['cases'])!=16:raise ValueError('All128 plus all16 required')
    tokenizer=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,local_files_only=True)
    reused=reused_reports(TRAINING_ROOT/'reasoning128',inputs,plan,tokenizer)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'reused').mkdir()
    for name in inputs['reused_files']:shutil.copyfile(TRAINING_ROOT/'reasoning128'/name,out/'reused'/name)
    choices=[tokenizer.encode(c,add_special_tokens=False) for c in 'ABCD']
    if any(len(x)!=1 for x in choices):raise ValueError('Single-token choice scoring required')
    conditions=[('foundation',None,None),('parent646',BASE/'native-training-resume8-v1/main/step-0120.pt',inputs['parent_checkpoint_sha256'])]
    for arm in ('expanded-data','rehearsal'):
        proof=completed_training(TRAINING_ROOT,arm);conditions.append((arm,TRAINING_ROOT/arm/'step-0084.pt',proof['checkpoint_sha256']))
    numeric_summaries={};mc_summary=None
    for condition,checkpoint,digest in conditions:
        ensure_guard();guarded(a.lease,180)
        model,_=load_native(a.model,language_only=True);llm=model.llm
        if checkpoint:
            if sha(checkpoint)!=digest:raise ValueError('Preserved checkpoint differs')
            profile=load_candidate(llm,checkpoint)
        else:
            if any('lora' in n.lower() or n.endswith(('.A','.B')) for n,_ in llm.named_parameters()):raise ValueError('Untouched foundation has adapter parameters')
            profile=dict(model_revision=MODEL_REVISION,adapter=False,optimizer_updates=0)
        llm.eval();started=time.monotonic()
        def generate(actual_prompt,maximum):
            ensure_guard();guarded(a.lease,90);set_seed(42)
            encoded=tokenizer(actual_prompt,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                generated=llm.generate(**encoded,max_new_tokens=maximum,do_sample=False,eos_token_id=tokenizer.convert_tokens_to_ids('<|im_end|>'),pad_token_id=tokenizer.convert_tokens_to_ids('<|im_end|>'))
            return tokenizer.decode(generated[0,encoded['input_ids'].shape[-1]:],skip_special_tokens=True).strip()
        if condition=='foundation':
            rows=[]
            for case in plan['cases']:
                actual=prompt(plan,case,tokenizer);encoded=tokenizer(actual,return_tensors='pt').to('cuda')
                with torch.inference_mode():
                    logits=llm(**encoded).logits[0,-1].float();indices=torch.tensor([x[0] for x in choices],device='cuda')
                    forced='ABCD'[int(logits[indices].argmax())];mass=float(logits.softmax(-1)[indices].sum())
                raw=generate(actual,24);free=raw if re.fullmatch('[ABCD]',raw) else None
                rows.append(dict(id=case['id'],category=case['category'],raw_response=raw,free_choice=free,format_valid=free is not None,free_correct=free==case['expected'],constrained_choice=forced,constrained_correct=forced==case['expected'],letter_probability_mass=mass,input_sha256=hashlib.sha256(actual.encode()).hexdigest()))
                (out/'foundation-progress.json').write_text(json.dumps(dict(required=128,cases=rows))+'\n')
            mc_summary=summary(rows)
            (out/'foundation.json').write_text(json.dumps(dict(complete128=True,source=a.source,plan_sha256=PLAN_SHA,profile=profile,cases=rows,summary=mc_summary),indent=2)+'\n')
        numeric_rows=[]
        for row in numeric['cases']:
            parts=messages(numeric,row)
            if getattr(tokenizer,'chat_template',None):actual=tokenizer.apply_chat_template(parts,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            else:actual=''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in parts)+'<|im_start|>assistant\n<think>\n\n</think>\n\n'
            raw=generate(actual,64)
            numeric_rows.append(dict(id=row['id'],raw_response=raw,input_sha256=hashlib.sha256(actual.encode()).hexdigest(),**numeric_score(raw,row['expected'])))
        numeric_summaries[condition]=dict(cases=16,correct=sum(r['correct'] for r in numeric_rows),single_numeric_result=sum(r['single_numeric_result'] for r in numeric_rows))
        (out/(condition+'-numeric16.json')).write_text(json.dumps(dict(complete16=True,source=a.source,plan_sha256=sha(numericpath),profile=profile,cases=numeric_rows,summary=numeric_summaries[condition],seconds=time.monotonic()-started),indent=2)+'\n')
        print(json.dumps(dict(condition=condition,numeric=numeric_summaries[condition],foundation_MC=mc_summary if condition=='foundation' else None)),flush=True)
        del llm,model;gc.collect();torch.cuda.empty_cache()
    report=dict(complete=True,source=a.source,model_revision=MODEL_REVISION,multiple_choice={'foundation':mc_summary,**{k:r['summary'] for k,r in reused.items()}},numeric16=numeric_summaries,reused_candidate_MC_outputs=384,new_foundation_MC_outputs=128,new_numeric_outputs=64,new_optimizer_updates=0,new_API_calls=0,text_only=True,spoken_reasoning_or_backend_execution_proved=False,adopted=False)
    (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','lease','source','plan','numeric-plan','output'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
