"""Independent actual-capability profile test on frozen native workers."""
import argparse
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('assets', 'kernel-assets', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    from evals.qwen35_controller import Qwen35ResponseExecutor
    from evals.qwen35_prefill_warmup import warm_long_prefill
    from evals.application_capabilities import ApplicationCapabilityExecutor
    from training.control.head_adapter import parameter_hashes
    from training.control.capability_curriculum import specifications
    from training.control.capability_evaluate import evaluate
    from evals.capability_fresh import evaluate as fresh
    executor = Qwen35ResponseExecutor(a.assets, kernel_assets=a.kernel_assets,
        pending_speech_protocol='explicit-actions', compact_replies=True)
    warm_long_prefill(executor); before = parameter_hashes(dict(executor.model.named_parameters()))
    scenes = [s for s in specifications() if s['split']=='heldout']; summaries = {}
    for mode in ('original', 'application-profile'):
        current = executor if mode == 'original' else ApplicationCapabilityExecutor(executor)
        summaries[mode] = dict(capabilities=evaluate(current, scenes, a.out/(mode+'-capabilities')),
            fresh=fresh(current, a.out/(mode+'-fresh')))
        if mode != 'original':
            from training.control.context_evaluate import evaluate as retention
            from training.control.paired_curriculum import specifications as paired
            summaries[mode]['retention'] = retention(current, [s for s in paired() if s['split']=='heldout'], a.out/'retention')
            (a.out/'requests.json').write_text(json.dumps(current.requests, indent=2)+'\n')
        print(json.dumps(dict(mode=mode, summary=summaries[mode])), flush=True)
    if before != parameter_hashes(dict(executor.model.named_parameters())):
        raise ValueError('Inference changed source parameters')
    report = dict(schema='dl-application-capability-probe-v1', summaries=summaries, strict_source=executor.proof,
        all_parameters_unchanged=True, optimizer_steps=0, actual_external_requests=0,
        outputs_repaired=False, actions_forced=False, semantic_review_required=True,
        do_not_train=True, overall_qualified=False, production_adopted=False, owner_demo_changed=False,
        wrapper_sha256=sha256(Path('evals/application_capabilities.py')), code_sha256=sha256(Path(__file__)))
    (a.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
