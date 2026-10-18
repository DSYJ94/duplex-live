"""Full fixed heldout workers for the shared-profile state experiment."""
import json
from pathlib import Path
from training.control.state_curriculum import specifications,run,assess


def evaluate(executor,folder):
    folder=Path(folder);folder.mkdir(exist_ok=False);rows=[]
    scenes=[s for s in specifications() if s['split']=='heldout']
    if len(scenes)!=204:raise ValueError('Requires all204 declared heldout scenes')
    for scene in scenes:
        row=dict(id=scene['id'],corpus=scene['corpus'],kind=scene['kind'],scene=scene,passed=False)
        try:
            result,world=run(scene,None,executor.model_id,response_executor=executor);quality=assess(scene,result,world)
            row.update(result=result,quality=quality,actual_world_after=world.tasks,passed=quality['passed'])
        except Exception as error:row.update(error_type=type(error).__name__,message=str(error),evidence=getattr(error,'result',None))
        rows.append(row);(folder/(scene['id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    summary=dict(cases=len(rows),mechanics_passed=sum(r['passed'] for r in rows),by_corpus={k:dict(cases=sum(r['corpus']==k for r in rows),passed=sum(r['corpus']==k and r['passed'] for r in rows)) for k in sorted({r['corpus'] for r in rows})},semantic_review_pending=True,actual_external_requests=0,heldout_optimizer_rows=0,whole_system_qualified=False)
    (folder/'report.json').write_text(json.dumps(dict(summary=summary,results=rows),indent=2)+'\n');print(json.dumps(summary),flush=True)
    return summary


def retention(before,after):
    read=lambda p:json.loads((Path(p)/'report.json').read_text())
    a,b=read(before),read(after);aa={r['id']:r for r in a['results']};bb={r['id']:r for r in b['results']}
    if len(aa)!=204 or set(aa)!=set(bb):raise ValueError('Changed heldout inventory')
    lost=[k for k,x in aa.items() if x['passed'] and not bb[k]['passed']]
    return dict(all_baseline_passes_retained=not lost,lost_passes=lost,baseline=a['summary'],candidate=b['summary'])
