"""Actual frozen/adapted worker generations on declared text fixtures."""
import hashlib
import json
from pathlib import Path
import time
from training.control.curriculum import evaluate_scene

BEHAVIOR_FILES=['training/control/curriculum.py','training/control/evaluate.py','evals/qwen35_controller.py','evals/qwen_response_executor.py','evals/qwen_controller_probe.py','evals/duplex_reasoning_worker.py','evals/duplex_speech_context.py']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def evaluate_suite(executor,scenes,folder):
    folder=Path(folder);folder.mkdir();results=[]
    for scene in scenes:
        row=dict(id=scene['id'],kind=scene['kind'],family=scene['family'],passed=False,synthetic_text_fixture=True,
            scene_sha256=hashlib.sha256(json.dumps(scene,sort_keys=True).encode()).hexdigest())
        before=time.monotonic()
        try:
            result,quality=evaluate_scene(scene,executor);row.update(result=result,quality=quality,passed=quality['passed'])
        except Exception as error:row.update(error_type=type(error).__name__,message=str(error),evidence=getattr(error,'result',None))
        row['wall_s']=time.monotonic()-before;results.append(row);(folder/(scene['id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    summary=dict(cases=len(results),passed=sum(r['passed'] for r in results),by_kind={kind:dict(total=sum(r['kind']==kind for r in results),passed=sum(r['kind']==kind and r['passed'] for r in results)) for kind in sorted({r['kind'] for r in results})})
    (folder/'report.json').write_text(json.dumps(dict(summary=summary,results=results),indent=2)+'\n');print(json.dumps(dict(stage=folder.name,summary=summary)),flush=True)
    return summary


def verify_baseline(folder,scenes,assets):
    folder=Path(folder);report=json.loads((folder/'report.json').read_text())
    if (report['schema']!='dl-controller-text-frozen-baseline-v1' or report['optimizer_steps']!=0 or not report['component_execution_completed']
            or report['assets_sha256']!=sha(assets) or report['behavior_files']!={p:sha(p) for p in BEHAVIOR_FILES}
            or not report['load']['strict'] or report['load']['trainable_parameters']!=0):raise ValueError('Frozen baseline source/model proof differs')
    if json.loads((folder/'scenes.json').read_text())!=scenes or sha(folder/'scenes.json')!=report['scenes_sha256']:raise ValueError('Frozen baseline held-out scenes differ')
    if sha(folder/'heldout/report.json')!=report['evaluation_report_sha256']:raise ValueError('Changed frozen baseline evaluation')
    evaluation=json.loads((folder/'heldout/report.json').read_text())
    if len(evaluation['results'])!=len(scenes) or {r['id'] for r in evaluation['results']}!={s['id'] for s in scenes}:raise ValueError('Incomplete baseline full-task inventory')
    by_id={s['id']:s for s in scenes}
    for row in evaluation['results']:
        file=folder/'heldout'/(row['id']+'.json')
        if (sha(file)!=report['row_sha256'][row['id']] or json.loads(file.read_text())!=row
                or row['scene_sha256']!=hashlib.sha256(json.dumps(by_id[row['id']],sort_keys=True).encode()).hexdigest()):raise ValueError('Changed frozen baseline row or input')
    return evaluation['summary'],dict(folder=str(folder),report_sha256=sha(folder/'report.json'),all_source_rows_exact=True)
