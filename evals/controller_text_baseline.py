"""Precompute actual original-model results while independent TRAIN labels collect."""
import argparse
import json
from pathlib import Path
from training.control.curriculum import specifications
from training.control.evaluate import evaluate_suite,BEHAVIOR_FILES,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','kernel-assets','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    heldout=[s for s in specifications() if s['split']=='heldout'];scene_path=args.out/'scenes.json';scene_path.write_text(json.dumps(heldout,indent=2)+'\n')
    from evals.qwen35_controller import Qwen35ResponseExecutor
    executor=Qwen35ResponseExecutor(args.assets,kernel_assets=args.kernel_assets,pending_speech_protocol='explicit-actions',compact_replies=True)
    summary=evaluate_suite(executor,heldout,args.out/'heldout')
    report=dict(schema='dl-controller-text-frozen-baseline-v1',component_execution_completed=True,optimizer_steps=0,load=executor.proof,summary=summary,
        assets_sha256=sha(args.assets),scenes_sha256=sha(scene_path),behavior_files={p:sha(p) for p in BEHAVIOR_FILES},
        evaluation_report_sha256=sha(args.out/'heldout/report.json'),row_sha256={s['id']:sha(args.out/'heldout'/(s['id']+'.json')) for s in heldout},
        synthetic_text_only=True,production_adopted=False,duplex_qualified=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(summary=summary,optimizer_steps=0)),flush=True)


if __name__=='__main__':main()
