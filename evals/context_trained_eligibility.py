"""CPU-only planning for verified trained TRAIN observations; never optimizer approval."""
import argparse
import json
from pathlib import Path
import sys
from evals.moshi_model_download import sha256


def main():
    import torch
    import sentencepiece
    from evals.context_history_validate import validate as validate_history
    from evals.context_training_data import load_training
    from evals.context_rollout_conditioning import verify
    from evals.context_future_component import plan as late_plan
    from evals.context_trained_boundary import plan as boundary_plan
    from evals.personaplex_train import MODEL_HASHES
    from evals.lychee_context import BASE_INSTRUCTIONS
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('map','data','assets','tokenizer','out'):ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--cache',type=Path,action='append',required=True)
    ap.add_argument('--regression-capture',type=Path,action='append',default=[])
    args=ap.parse_args();rows=[json.loads(l) for l in args.map.read_text().splitlines()]
    if (not rows or len({r['job'] for r in rows})!=len(rows)
            or any(r['training_approved'] is not False or r['kind'] not in ('simple','clarify','private') for r in rows)):
        raise RuntimeError('Expected distinct unapproved trained TRAIN captures')
    if sha256(args.tokenizer)!=MODEL_HASHES['tokenizer_spm_32k_3.model']:raise RuntimeError('Changed tokenizer')
    assets=json.loads(args.assets.read_text());sys.path.insert(0,assets['source'])
    from contextspan.model.sequence_convention import persona_prefix
    root=args.data.parent;first=root/('context-spanning-probe-'+rows[0]['job'])
    voice=torch.load(first/'student-voice.pt',map_location='cpu',weights_only=True)['codes']
    spm=sentencepiece.SentencePieceProcessor(str(args.tokenizer))
    examples,_=load_training(args.cache,spm,persona_prefix(BASE_INSTRUCTIONS,voice,spm),request_policy='native-prelead-epad')
    args.out.mkdir(exist_ok=False);regressions=[]
    for folder in args.regression_capture:
        from evals.context_component_kind import manifest_name
        name=manifest_name(folder,mixed=True)
        if name=='early-future.json':
            from evals.context_early_validate import validate as validate_component
        elif name=='actual-future.json':
            from evals.context_future_validate import validate as validate_component
        else:raise RuntimeError('Unsupported legacy regression kind')
        audit,_=validate_component(folder,args.data,args.assets,examples,spm,voice);regressions.append(audit)
    results=[]
    for index,row in enumerate(rows):
        history=root/('context-spanning-probe-'+row['job'])
        source=validate_history(history,args.data,args.assets,row['source_commit'])
        record=json.loads((history/'student-history.json').read_text())
        report=json.loads((history/'report.json').read_text());collector=verify(report['loading'])
        if (source['scenario']!=row['scenario'] or record['source']['kind']!=row['kind']
                or Path(collector['proof']).name!='context-training-proof-'+str(row['collector_job'])
                or not torch.equal(voice,torch.load(history/'student-voice.pt',map_location='cpu',weights_only=True)['codes'])):
            raise RuntimeError('Trained capture source, collector or voice differs from its declared map')
        item=dict(index=index,job=row['job'],scenario=row['scenario'],kind=row['kind'],teacher_id=row['teacher_id'],
            collector=collector,source_commit=row['source_commit'],raw_history_validated=True,training_approved=False)
        try:
            early=row['kind']!='private'
            plan,_=late_plan(history,args.data,examples,spm,early=early,teacher_id=row['teacher_id'])
            if early:plan,_=boundary_plan(plan,examples)
            if plan.get('conditioning_policy')!='verified-trained-rollout':raise RuntimeError('Missing verified trained conditioning')
            filename='plan-'+str(index)+'.json';path=args.out/filename;path.write_text(json.dumps(plan,indent=2)+'\n')
            item.update(eligible=True,plan_file=filename,plan_sha256=sha256(path),target_kind=plan['target_kind'])
        except RuntimeError as error:item.update(eligible=False,reason=str(error))
        results.append(item);print(json.dumps(item),flush=True)
    result=dict(schema='dl-context-trained-future-eligibility-v1',map_sha256=sha256(args.map),cases=results,
        regressions=regressions,all_regressions_passed=all(r['passed'] for r in regressions),
        eligible_count=sum(r['eligible'] for r in results),optimizer_steps=0,gold_targets_created=False,training_approved=False)
    (args.out/'eligibility.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
