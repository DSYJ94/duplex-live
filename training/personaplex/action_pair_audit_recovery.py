"""Audit the completed96 only: no engine, generation, fit or optimizer load."""
import argparse,json
from pathlib import Path
from .expanded_guard import PAIR_RECOVERY_AUTH,sha
from .inference_guard import FrameGuard
from .action_pair_recovery import ORIGIN
from . import matched_evaluation as evaluation


def run(a):
    guard=FrameGuard(a.lease);lease=guard(600)
    if lease.get('paired_recovery_sha256')!=sha(PAIR_RECOVERY_AUTH)or lease['source']!=a.source:
        raise ValueError('Exact separately bounded audit recovery scope required')
    if sha(a.plan)!=lease['action_plan_sha256']:raise ValueError('Admitted recovery plan changed')
    plan=json.loads(a.plan.read_text());a.trainer_source=plan['selection']['trainer_source'];a.capacity='expanded105M';a.base='personaplex'
    root=a.root
    for cohort,count,key in (('control32',32,'control32_sha256'),('dev64',64,'dev64_sha256')):
        path=a.inputs/('control-cases.json'if count==32 else'dev-cases.json')
        if sha(path)!=plan[key]:raise ValueError('Exact original complete inputs required')
        rows=json.loads(path.read_text());a.root=root/'readout/evaluate'/cohort
        evaluation.audit(a,rows,required=count,guard=lambda _lease,needed=0:guard(needed),
                         require_spoken_content=True,generation_source=ORIGIN)
        report=json.loads((a.root/'personaplex-expanded105M/report.json').read_text())
        if not report['complete']or report['source']!=ORIGIN or report['audit_source']!=a.source or len(report['cases'])!=count:
            raise ValueError('Complete reused generation and actual new audit provenance required')
    with (root/'readout/evaluate/terminal.json').open('x')as f:
        json.dump(dict(passed=True,source=a.source,generation_source=ORIGIN,outputs=96,
                       native_generations_repeated=0,optimizer_updates=0,actual_CUDA=True,
                       product_accepted=False),f,indent=2);f.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','inputs','plan','lease'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source',required=True);run(p.parse_args())
