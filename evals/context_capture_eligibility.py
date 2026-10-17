"""Check captured TRAIN conditioning and plan fact futures without optimization."""
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
    from evals.context_future_component import plan as fact_plan
    from evals.context_early_plan import plan as early_plan
    from evals.personaplex_train import MODEL_HASHES
    from evals.lychee_context import BASE_INSTRUCTIONS
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('map','data','assets','tokenizer','out'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--cache',type=Path,action='append',required=True)
    ap.add_argument('--regression-capture',type=Path,action='append',default=[])
    ap.add_argument('--late-diagnostics',action='store_true',
        help='Also plan private sentence corrections after actual delivered history; never approve labels')
    args = ap.parse_args()
    mapping = json.loads(args.map.read_text())
    if mapping['schema'] != 'dl-context-actual-capture-map-v1' or mapping['training_approved'] is not False:
        raise RuntimeError('Not an unapproved actual capture map')
    if sha256(args.tokenizer) != MODEL_HASHES['tokenizer_spm_32k_3.model']:
        raise RuntimeError('Changed native tokenizer')
    assets = json.loads(args.assets.read_text()); sys.path.insert(0,assets['source'])
    from contextspan.model.sequence_convention import persona_prefix
    spm = sentencepiece.SentencePieceProcessor(str(args.tokenizer))
    root = args.data.parent
    first = root/('context-spanning-probe-'+mapping['cases'][0]['job'])
    voice = torch.load(first/'student-voice.pt',map_location='cpu',weights_only=True)['codes']
    examples,_ = load_training(args.cache,spm,persona_prefix(BASE_INSTRUCTIONS,voice,spm),
        request_policy='native-prelead-epad')
    args.out.mkdir(exist_ok=False)
    regressions = []
    for folder in args.regression_capture:
        from evals.context_early_validate import validate
        recorded_voice=torch.load(folder/'voice.pt',map_location='cpu',weights_only=True)['codes']
        regression_examples,_=load_training(args.cache,spm,
            persona_prefix(BASE_INSTRUCTIONS,recorded_voice,spm),request_policy='native-prelead-epad')
        report,_ = validate(folder,args.data,args.assets,regression_examples,spm,recorded_voice)
        regressions.append(report)
    results = []
    for case in mapping['cases']:
        folder = root/('context-spanning-probe-'+case['job'])
        raw = json.loads((folder/'student-history.json').read_text())
        checked = validate_history(folder,args.data,args.assets,mapping['collector_commit'])
        expected_world = {} if case['expected_kind']=='simple' else {
            case['project']:{case['field']:case['private_value']}}
        if (checked['scenario'] != case['scenario'] or raw['source']['kind'] != case['expected_kind']
                or raw['source']['wav_sha256'] != case['source_wav_sha256']
                or raw['source']['sidecar_sha256'] != case['source_sidecar_sha256']
                or raw['source']['declared_counterfactual_world'] != expected_world):
            raise RuntimeError('Captured source/kind/world differs from the hash-bound teacher plan')
        item = {'index':case['index'],'job':case['job'],'teacher':case['teacher'],
            'scenario':case['scenario'],'kind':case['expected_kind'],
            'raw_history_validated':True,'training_approved':False}
        try:
            loading=json.loads((folder/'loading.json').read_text())
            adapter=loading.get('adapter')
            if (not adapter or not adapter['zero'] or adapter['disabled']
                    or not loading.get('layout') or loading.get('input_warmup')):
                raise RuntimeError('Capture protocol differs from the audited zero-wrapper/native-layout replay')
            if case['expected_kind'] not in ('private','clarify','simple'):
                raise RuntimeError('Cancellation needs separate independently approved action targets')
            if not torch.equal(voice,torch.load(folder/'student-voice.pt',map_location='cpu',weights_only=True)['codes']):
                raise RuntimeError('Capture uses different accepted voice codes; group separately before fitting')
            selected,_ = fact_plan(folder,args.data,examples,spm,early=True,teacher_id=case['teacher'])
            plan,_ = early_plan(selected,examples)
            name = 'plan-'+str(case['index'])+'.json'
            (args.out/name).write_text(json.dumps(plan,indent=2)+'\n')
            item.update(eligible_for_strict_early_capture=True,plan_file=name,
                plan_sha256=sha256(args.out/name),real_future_frames=plan['future_real_frames'])
        except RuntimeError as error:
            item.update(eligible_for_strict_early_capture=False,reason=str(error))
        if args.late_diagnostics and case['expected_kind']=='private':
            try:
                late,_=fact_plan(folder,args.data,examples,spm,teacher_id=case['teacher'])
                name='late-plan-'+str(case['index'])+'.json'
                (args.out/name).write_text(json.dumps(late,indent=2)+'\n')
                item.update(late_correction_diagnostic_eligible=True,late_plan_file=name,
                    late_plan_sha256=sha256(args.out/name))
            except RuntimeError as error:
                item.update(late_correction_diagnostic_eligible=False,late_diagnostic_reason=str(error))
        results.append(item)
        print(json.dumps(item),flush=True)
    report = {'schema':'dl-context-capture-eligibility-v1','capture_map_sha256':sha256(args.map),
        'cases':results,'legacy_component_regressions':regressions,'heldout_used':False,
        'training_approved':False,'optimizer_steps':0,'labels_created':False}
    (args.out/'eligibility.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'raw_histories':len(results),'eligible_early_futures':sum(
        r['eligible_for_strict_early_capture'] for r in results),'eligible_fact_futures':sum(
        r['eligible_for_strict_early_capture'] and r['kind'] in ('private','clarify') for r in results),
        'late_correction_diagnostic_candidates':sum(r.get('late_correction_diagnostic_eligible',False) for r in results),
        'legacy_regressions_passed':len(regressions),'training_approved':False}),flush=True)


if __name__ == '__main__':main()
