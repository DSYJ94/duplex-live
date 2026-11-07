"""Summarise actual job allocations without double-counting Slurm job steps.

Allocated GPU-hours are not a substitute for the portal's billed node-hours.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path


def summarize(text):
    jobs=[];seen=set()
    for line in text.splitlines():
        if not line.strip():continue
        values=line.split('|')
        if len(values)==9 and values[-1]=='':values.pop()
        if len(values)!=8:raise ValueError('Complete eight-column sacct rows required')
        ident,name,state,exit_code,elapsed,tres,start,end=values
        if '.' in ident:continue
        if ident in seen:raise ValueError('Duplicate top-level Slurm job')
        seen.add(ident)
        allocation=dict(x.split('=',1) for x in tres.split(',') if '=' in x)
        seconds=int(elapsed);gpus=int(allocation.get('gres/gpu','0'))
        if seconds<0 or gpus<0:raise ValueError('Nonnegative actual allocation required')
        jobs.append(dict(job_id=ident,name=name,state=state,exit_code=exit_code,
            elapsed_seconds=seconds,allocated_gpus=gpus,allocated_gpu_seconds=seconds*gpus,
            allocated_gpu_hours=seconds*gpus/3600,alloc_tres=tres,start=start,end=end))
    return dict(jobs=jobs,top_level_jobs=len(jobs),
        allocated_gpu_hours=sum(r['allocated_gpu_seconds'] for r in jobs)/3600,
        failed_or_cancelled_gpu_hours=sum(r['allocated_gpu_seconds'] for r in jobs
            if r['state'].startswith(('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY')))/3600,
        official_portal_charged_node_hours=None,allocation_remaining_not_inferred=True,
        scope='Actual sacct account/user/date filters recorded by caller; running rows are snapshots')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--since',required=True);p.add_argument('--account',required=True);p.add_argument('--user',required=True)
    a=p.parse_args();result=summarize(a.input.read_text())
    result.update(observed_at=datetime.now(timezone.utc).isoformat(),since=a.since,account=a.account,user=a.user)
    with a.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='jobs'}))
