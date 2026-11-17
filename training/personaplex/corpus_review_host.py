"""One frozen public-corpus review after completed native collection/backups."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from .expanded_guard import AUTH,CORPUS_REVIEW_AUTH,CORPUS_REPAIR_AUTH,ACCOUNT,KEY,sha,parse_job,stamp,validate_job
from .expanded_evaluation_host import parent_provider
from .expanded_launch import container_command
from .native_feature_host import archive
from .local_reference_host import RUNTIME
from .local_reasoner_reference import PUBLIC,PIN,PIN_SHA,REVISION

BASE=Path('/projects/u6ada/duplex-live')
PARENT=BASE/'native-TRAIN-failure-20261117-v1/job-8553179/host-terminal.json'
INPUTS=[dict(path=str(BASE/'native-broad-corpus-staging-20261117/candidates-lexical-v1'/name/'candidates.jsonl'),sha256=digest)
        for name,digest in [('OpenAssistant--oasst1','0f19a2dbcdf4ceca07e40ce592aa0753cf22713ead1337b556eda291290137f0'),
                            ('databricks--databricks-dolly-15k','9791cdb7c25aad8aafdd37cfc7fdfa6386bfe1c9c7c271e76b9a07d693e40d10')]]


def prepare(root,source):
    for entry in INPUTS:
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Exact whole public source candidates required')
    if sha(PIN)!=PIN_SHA:raise ValueError('Exact approved read-only public reviewer pin required')
    root.mkdir(exist_ok=False)
    plan=dict(source=source,inputs=INPUTS,required_candidates=17276,world_size=4,
              reviewer_pin_sha256=PIN_SHA,reviewer_revision=REVISION,
              optimizer_updates=0,external_API_calls=0,training_admitted=False)
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')


def require_parent():
    actual=parent_provider('8553179')
    if actual.get('JobState')!='COMPLETED' or actual.get('ExitCode')!='0:0':
        raise ValueError('Never overlap or interrupt valid native collection')
    terminal=json.loads(PARENT.read_text())
    result=terminal.get('results',{})
    if (terminal.get('success') is not True or result.get('complete') is not True
            or result.get('actual_outputs')!=1008 or result.get('all_result_hashes_verified') is not True
            or set(terminal.get('backups',{}))!={'0','1','2','3'}):
        raise ValueError('Complete1008 native parent and all independent backups required')
    for receipt in terminal['backups'].values():
        if (receipt.get('passed') is not True or receipt.get('every_file_sha256_verified') is not True
                or sha(Path(receipt['destination'])/'archive-inventory.json')!=receipt['inventory_sha256']):
            raise ValueError('Exact successful independent parent archive required')
    return sha(PARENT)


def complete(root,plan):
    rows=[]
    for entry in plan['inputs']:
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Immutable corpus source changed')
        rows.extend(json.loads(line)for line in Path(entry['path']).open())
    ordered=sorted(row['id']for row in rows);seen=[];processes=[];devices=[]
    if len(rows)!=17276 or len(set(ordered))!=17276:raise ValueError('Whole distinct17276 source cohort required')
    from .corpus_semantic_review import fingerprint,parse
    counts={key:Counter()for key in ('review_status','domain','task','verdict','answerability','support')}
    source_rows={row['id']:row for row in rows}
    for rank in range(4):
        folder=root/('rank-%02d'%rank);terminal=json.loads((folder/'terminal.json').read_text())
        execution=terminal.get('execution') or {};results=terminal.get('results',[])
        if (terminal.get('complete') is not True or terminal.get('source')!=plan['source']
                or terminal.get('rank')!=rank or terminal.get('world_size')!=4
                or terminal.get('whole_cohort_sha256')!=fingerprint(rows)
                or [row['id']for row in results]!=ordered[rank::4]
                or terminal.get('required')!=len(ordered[rank::4])
                or execution.get('actual_CUDA') is not True or execution.get('visible_device_count')!=1
                or terminal.get('model_identity',{}).get('revision')!=REVISION
                or terminal.get('model_identity',{}).get('repository')!='Qwen/Qwen3.6-27B'
                or terminal.get('optimizer_updates')!=0 or terminal.get('external_API_calls')!=0):
            raise ValueError('Every disjoint actual frozen reviewer must complete its whole shard')
        processes.append((execution['pid_namespace'],execution['pid'],execution['start_ticks']));devices.append(execution['selected_device'])
        for entry in results:
            name=entry['file']
            if Path(name).name!=name or sha(folder/name)!=entry['sha256']:raise ValueError('Every original review hash required')
            value=json.loads((folder/name).read_text())
            if (value['id']!=entry['id'] or value['source_sha256']!=fingerprint(source_rows[entry['id']])
                    or value['review_status']!=entry['status'] or value.get('training_admitted') is not False):
                raise ValueError('No duplicated, substituted or automatically admitted review')
            counts['review_status'][value['review_status']]+=1
            if value['review_status']=='structured_review_complete':
                checked=parse(value['generation']['text'],source_rows[entry['id']])
                if checked!=value['review']:raise ValueError('Raw review generation and parsed evidence differ')
                for key in counts:
                    if key!='review_status':counts[key][checked[key]]+=1
            seen.append(entry['id'])
    if sorted(seen)!=ordered or len(set(processes))!=4 or len(set(devices))!=4:
        raise ValueError('Whole17276 cohort and distinct actual reviewer processes/devices required')
    return dict(complete=True,required=17276,actual_reviews=len(seen),all_review_hashes_verified=True,
                counts={key:dict(value)for key,value in counts.items()},
                training_admitted=False,semantic_qualification_performed=False,optimizer_updates=0)


def main(a):
    if subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()!=a.source:
        raise ValueError('Exact committed immutable source required')
    if a.prepare:
        if a.repair:
            from .corpus_review_repair import prepare as repair_prepare
            if a.batch_readiness is None:raise ValueError('Actual batch-tokenizer readiness path required')
            repair_prepare(a.root,a.backup,a.source,INPUTS,a.batch_readiness)
        else:prepare(a.root,a.source)
        return
    lock=(a.root/'.controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if any(a.root.glob('job-*')):raise ValueError('One actual review job; no automatic retries')
    parent_sha=require_parent();plan=json.loads((a.root/'plan.json').read_text())
    if plan.get('source')!=a.source or plan.get('inputs')!=INPUTS or sha(PIN)!=PIN_SHA:
        raise ValueError('Exact prepared public review plan/pin required')
    if bool(a.repair)!=('repair_scope_sha256' in plan):raise ValueError('Explicit exact repair phase required')
    padding=0
    if a.repair:
        from .corpus_review_repair import validate as validate_repair
        padding=validate_repair(plan)
    job=os.environ['SLURM_JOB_ID']
    actual=parse_job(subprocess.check_output(['scontrol','show','job','-o',job],universal_newlines=True,timeout=30))
    raw=subprocess.check_output(['sacct','-A',ACCOUNT,'-S','2026-10-30','-X','-n','-P','-o',
        'JobIDRaw,JobName%100,State,ElapsedRaw,AllocTRES,ReqTRES,TimelimitRaw'],universal_newlines=True,timeout=30)
    prior=reserved=0
    for line in raw.splitlines():
        fields=line.split('|')
        if len(fields)<7 or fields[0]==job:continue
        ident,name,state,elapsed,allocated,requested,limit=fields[:7]
        tres=dict(v.split('=',1)for v in (allocated or requested).split(',')if '='in v)
        count=int(tres.get('gres/gpu',0));used=int(elapsed);prior+=count*used
        if state in ('RUNNING','PENDING','CONFIGURING'):
            if name.startswith('duplex-') and count:raise ValueError('Another voice allocation is active')
            if not limit.isdigit():raise ValueError('Bound every concurrent reservation')
            reserved+=count*max(0,int(limit)*60-used)
    prior+=padding
    quota=subprocess.check_output(['quota','-w','-v'],universal_newlines=True,timeout=20)
    q=next(row.split()for row in quota.splitlines()if row.strip().startswith('data1.vastp2.isambard.ac.uk:/p2/home '))
    if (int(q[3])-int(q[1]))*1024<2*1024**3:raise ValueError('Independent review backup needs2GiB headroom')
    lease=dict(provider='Slurm',resource_key=KEY,account=ACCOUNT,authorization_sha256=sha(AUTH),
        corpus_review_authorization_sha256=sha(CORPUS_REVIEW_AUTH),corpus_review_parent_job='8553179',
        corpus_review_plan_sha256=sha(a.root/'plan.json'),GPU_count=4,job_id=job,node=actual['NodeList'],source=a.source,
        start_unix=stamp(actual['StartTime']),stop_unix=stamp(actual['EndTime']),
        prior_award_GPU_seconds=prior,other_reserved_GPU_seconds=reserved,parent_terminal_sha256=parent_sha)
    if a.repair:lease['corpus_review_repair_sha256']=sha(CORPUS_REPAIR_AUTH)
    validate_job(actual,lease,time.time());folder=a.root/('job-'+job);folder.mkdir(exist_ok=False)
    a.backup.mkdir(parents=True,exist_ok=True);path=folder/'lease.json';path.write_text(json.dumps(lease,indent=2)+'\n')
    (folder/'provider-admission.txt').write_text(raw+'\n'+quota)
    env=dict(os.environ,DUPLEX_LIVE_GUARD_FILE=str(folder/'guard.json'),DUPLEX_LIVE_LEASE_FILE=str(path))
    watch=subprocess.Popen(['python3','-u','-m','training.personaplex.expanded_guard','--lease',str(path),'--output',str(folder/'guard.json')],env=env)
    try:
        deadline=time.time()+180
        while not(folder/'guard.json').exists():
            if watch.poll()is not None or time.time()>deadline:raise RuntimeError('Fresh actual provider guard unavailable')
            time.sleep(2)
        inside={key:env[key]for key in ('DUPLEX_LIVE_GUARD_FILE','DUPLEX_LIVE_LEASE_FILE','SLURM_JOB_ID','SLURM_JOB_ACCOUNT')}
        inside.update(PYTHONPATH=str(RUNTIME/'pylibs')+':'+str(Path.cwd()),USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',GIT_OPTIONAL_LOCKS='0')
        cmd=container_command(Path.cwd(),gpu=True,environment=inside,backup=a.backup)
        selected=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if (len(selected.split(','))!=4 or len(set(selected.split(',')))!=4
                or cmd.count('CUDA_VISIBLE_DEVICES=0,1,2,3')!=1):
            raise ValueError('Four distinct actual scheduler-selected devices required')
        cmd[cmd.index('CUDA_VISIBLE_DEVICES=0,1,2,3')]='CUDA_VISIBLE_DEVICES='+selected
        position=cmd.index('--pwd');cmd[position:position]=['--bind',str(PUBLIC)+':'+str(PUBLIC)+':ro','--bind',str(PIN)+':'+str(PIN)+':ro']
        cmd+=['torch.distributed.run','--standalone','--nproc_per_node=4','--module','training.personaplex.expanded_gpu',
              'training.personaplex.corpus_semantic_review','--plan',str(a.root/'plan.json'),'--output',str(a.root),
              '--lease',str(path),'--source',a.source]
        subprocess.run(cmd,env=env,check=True,timeout=max(1,lease['stop_unix']-time.time()-240))
        result=complete(a.root,plan)
        backups={str(rank):archive(a.root/('rank-%02d'%rank),a.backup/('job-'+job)/('rank-%02d'%rank))for rank in range(4)}
        (folder/'host-terminal.json').write_text(json.dumps(dict(success=True,source=a.source,results=result,backups=backups,training_admitted=False),indent=2)+'\n')
    finally:watch.terminate();watch.wait(timeout=30)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ('root','backup'):p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--source',required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--repair',action='store_true')
    p.add_argument('--batch-readiness',type=Path)
    main(p.parse_args())
