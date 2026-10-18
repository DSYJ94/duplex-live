"""Full fixture workers for capability holdouts; semantic review stays explicit."""
import json
from pathlib import Path
from training.control.capability_curriculum import world_and_worker,assess


def evaluate(executor,scenes,folder):
    folder=Path(folder);folder.mkdir(exist_ok=False);rows=[]
    if len(scenes)!=42 or any(s['split']!='heldout' for s in scenes):raise ValueError('Requires the complete42 capability holdouts')
    for scene in scenes:
        row=dict(id=scene['id'],kind=scene['kind'],scene=scene,passed=False,semantic_review_required=scene['kind'] in ('unsupported-read','unsupported-send','finished-cancel','record-clarify'))
        try:
            context,world,worker=world_and_worker(scene,executor.model_id)
            result=worker(context,world,'latest',response_executor=executor);quality=assess(scene,result,world)
            row.update(result=result,quality=quality,passed=quality['passed'])
        except Exception as error:row.update(error_type=type(error).__name__,message=str(error),evidence=getattr(error,'result',None))
        rows.append(row);(folder/(scene['id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    summary=dict(cases=len(rows),mechanics_passed=sum(r['passed'] for r in rows),by_kind={k:dict(total=sum(r['kind']==k for r in rows),passed=sum(r['kind']==k and r['passed'] for r in rows)) for k in sorted({r['kind'] for r in rows})},semantic_review_pending=True,actual_external_requests=0,private_reads_are_executed_synthetic_fixture_reads=True,native_duplex_qualified=False)
    (folder/'report.json').write_text(json.dumps(dict(summary=summary,results=rows),indent=2)+'\n');return summary
