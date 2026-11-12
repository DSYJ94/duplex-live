"""One bounded evaluation-only recovery. No optimizer path or training import execution."""
import argparse
import json
import os
import shlex
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from . import matched_guard as g, matched_resume as resume, paced_health_service as health
from . import evaluation_recovery_policy as policy
from .matched_interrupted import DURABLE
from .matched_job import write, visible_pair
from .matched_paths import command, BASE
from .matched_finish_submit import parent, backup_proofs
from .matched_generation import completed_states
from .matched_evaluation_reuse import materialize, verify

TRAIN=DURABLE/'runs/rpc-recovery-8317068'
FAILED=DURABLE/'runs/matched-evaluation-8352856'
PINS=DURABLE/'runs/evaluation-reuse-pin-a5c1deb'
CLAIM=DURABLE/'evaluation-recovery-submission-v2'
PRODUCER='10e7e0664e47c1222e0492a8367b3c000ed5789d'
OLD='63f121fc876394c8d63a38e22535b6a2496015f8'
OUTPUT=DURABLE/'runs/evaluation-recovery-v2'
COUNTS={'personaplex-baseline':32,'context-spanning-baseline':32,
        'personaplex-narrow':6,'context-spanning-narrow':6}


def accounting(text, ordered, current='0'):
    rows=[line for line in text.splitlines() if line.split('|')[0] not in ('8352856','8365873',current)]
    prior=resume.allocation_carry('\n'.join(rows),ordered,'0')
    failed=[line.split('|') for line in text.splitlines() if line.split('|')[0]=='8352856']
    if len(failed)!=1:raise ValueError('Actual failed evaluation allocation required')
    f=failed[0];tres=dict(x.split('=',1) for x in f[6].split(',') if '=' in x)
    if (f[2]!=g.ACCOUNT or f[3]!='FAILED' or f[4]!='1:0' or not f[5].isdigit()
            or tres.get('gres/gpu')!='2' or prior!=263445 or int(f[5])!=2980):
        raise ValueError('All previous allocation clocks must remain exact')
    first=[line.split('|') for line in text.splitlines() if line.split('|')[0]=='8365873']
    if (len(first)!=1 or first[0][2:6]!=[g.ACCOUNT,'FAILED','1:0','580']
            or dict(x.split('=',1) for x in first[0][6].split(',') if '=' in x).get('gres/gpu')!='2'):
        raise ValueError('Actual failed copy-runtime allocation must remain counted')
    return prior+2*(int(f[5])+580)


def records():
    pinned=json.loads(Path('docs/NATIVE-MOSHI-EVALUATION-GUARD-RECOVERY-2026-11-12.json').read_text())['reuse_implementation']['actual_pin_summary']
    index=json.loads(Path('docs/NATIVE-MOSHI-EVALUATION-REUSE-INPUTS-2026-11-12.json').read_text())
    if set(pinned['conditions'])!=set(COUNTS):raise ValueError('All76 preserved cases required')
    result={}
    for name,count in COUNTS.items():
        row=pinned['conditions'][name];path=PINS/(name+'.json')
        if str(path)!=row['record'] or g.sha(path)!=row['sha256']:raise ValueError('Original reuse pin changed')
        value=json.loads(path.read_text())
        if (value['parent']!=str(FAILED/name) or value['original_source']!=OLD
                or len(value['cases'])!=count or value['full_audit']!=(count==32)):
            raise ValueError('Only actual completed outputs may be reused')
        for kind in ('profile','results','report'):
            expected=index['conditions'][name].get(kind+'_sha256')
            if expected and value['files'][kind+'.json']['sha256']!=expected:
                raise ValueError('Pinned historical metadata differs')
        result[name]=value
    return result


def immutable_math(source):
    for file in resume.CONTROL+(str(g.AUTH),str(g.EXTENSION)):
        if file=='training/personaplex/matched_guard.py':continue
        original=subprocess.check_output(['git','show',PRODUCER+':'+file])
        if original!=Path(file).read_bytes():raise ValueError('Training/math/input bytes differ: '+file)
    if subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()!=source:
        raise ValueError('Exact controller Git source required')
    if subprocess.check_output(['git','status','--porcelain'],universal_newlines=True).strip():
        raise ValueError('Clean deployed source required')


def preflight(source, current='0'):
    grant=policy.authority();immutable_math(source)
    ordered,digest,complete=parent(TRAIN)
    if not complete:raise ValueError('No training or partial trained state allowed')
    terminal=json.loads((FAILED/'execution-8352856/terminal.json').read_text())
    if terminal['actual_exit_code']!=1 or terminal['source']!=OLD:raise ValueError('Actual failed parent required')
    text=subprocess.check_output(['sacct','-n','-X','-P','-A',g.ACCOUNT,'--starttime=2026-11-08',
        '--format=JobIDRaw,JobName%100,Account,State,ExitCode,ElapsedRaw,AllocTRES'],universal_newlines=True,timeout=30)
    prior=accounting(text,ordered,current)
    if prior+2*13800>112*3600:raise ValueError('No aggregate budget reset')
    return grant,ordered[-1]['lease'],prior,records()


def submit(source):
    grant,old,prior,reuse=preflight(source)
    queue=subprocess.check_output(['squeue','-h','-A',g.ACCOUNT,'-o','%i'],universal_newlines=True,timeout=20)
    if queue.strip():raise ValueError('No overlapping project allocation')
    backups=backup_proofs(TRAIN,PRODUCER)
    # Admission before GPU allocation, one actual pass over reused bytes.
    for record in reuse.values():verify(record['parent'],record['files'])
    from .matched_rpc_job import prepare_mount_target
    prepare_mount_target(Path.cwd())
    native=['/usr/bin/python3','-B','-u','-m','training.personaplex.evaluation_recovery','--mode','run','--source',source]
    cmd=['sbatch','--parsable','--account='+g.ACCOUNT,'--partition=workq','--nodes=1','--gpus=2',
         '--cpus-per-task=16','--mem=200G','--time=230','--no-requeue','--export=NONE',
         '--job-name=duplex-matched-eval-recovery-v1','--chdir='+str(Path.cwd()),
         '--output='+str(DURABLE/'slurm-eval-recovery-%j.log'),
         '--wrap=ulimit -c 0\nexec '+' '.join(shlex.quote(x) for x in native)]
    CLAIM.mkdir(exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)
    for name,record in reuse.items():materialize(record,OUTPUT/name)
    write(OUTPUT/'staging-complete.json',dict(source=source,reused_outputs=76,models_loaded=0))
    from .matched_paths import command as container_command
    os.environ['APPTAINER_BINDPATH']=str(DURABLE)+':'+str(DURABLE)
    subprocess.run(container_command(Path.cwd(),gpu=None)+[
        'training.personaplex.evaluation_recovery','--mode','staged-check','--source',source],check=True)
    write(CLAIM/'command.json',dict(source=source,authorization_sha256=grant,prior_GPU_seconds=prior,
        maximum_seconds=13800,optimizer_updates=0,command=cmd,backups=backups,reuse_counts=COUNTS))
    try:response=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=60)
    except subprocess.TimeoutExpired:
        write(CLAIM/'unknown.json',dict(state='UNKNOWN_DO_NOT_RESUBMIT'));raise
    write(CLAIM/'response.json',dict(actual_exit=response.returncode,stdout=response.stdout,stderr=response.stderr))
    job=response.stdout.strip().split(';')[0]
    if response.returncode or not job.isdigit():raise RuntimeError('Submission failed/ambiguous; no automatic retry')
    write(CLAIM/'job.json',dict(job_id=job,source=source));print(job,flush=True)


def run(source):
    grant,old,prior,reuse=preflight(source,os.environ['SLURM_JOB_ID'])
    from .matched_rpc_job import startup_job
    job=startup_job(os.environ['SLURM_JOB_ID'])
    lease=dict(old,source=source,controller_source=source,trainer_source=PRODUCER,job_id=job['JobId'],
        node=job['NodeList'],start_unix=g.stamp(job['StartTime']),stop_unix=g.stamp(job['EndTime']),
        prior_GPU_seconds=prior,maximum_seconds=13800,continuation_interval=12,optimizer_updates=0)
    lease[policy.KEY]=grant
    g.validate_job(job,lease,time.time())
    if lease['stop_unix']-time.time()<10800:raise ValueError('Complete remaining comparison reserve required')
    selected=completed_states(TRAIN,PRODUCER)
    staging=json.loads((OUTPUT/'staging-complete.json').read_text())
    if staging!={'source':source,'reused_outputs':76,'models_loaded':0}:
        raise ValueError('Actual complete CPU staging required before GPU execution')
    execution=OUTPUT/('execution-'+lease['job_id']);execution.mkdir()
    write(execution/'lease.json',lease)
    write(OUTPUT/'candidate-selection.json',dict(states=selected,trainer_source=PRODUCER,source=source))
    services=[];control=None;code=1;errors=[]
    try:
        watcher,services,control=health.start(lease,execution)
        def ensure():
            if watcher.poll() is not None:raise RuntimeError('Independent recovery guard stopped')
            g.ensure(lease,json.loads((control/'guard.json').read_text()))
        # Baseline outputs/audits retain their ORIGINAL producer and bytes.
        for name in ('personaplex-baseline','context-spanning-baseline'):
            from .matched_evaluation_reuse import prepared
            ensure();prepared(reuse[name],OUTPUT/name)
        devices=dict(zip(('personaplex','context-spanning'),visible_pair(os.environ['CUDA_VISIBLE_DEVICES'])))
        def worker(base):
            try:
                for capacity in ('narrow','broad'):
                    for stage in ('collect','audit'):
                        ensure();name=base+'-'+capacity+'-'+stage
                        env=dict(SLURM_JOB_ID=lease['job_id'],SLURM_JOB_ACCOUNT=g.ACCOUNT,
                            DUPLEX_LIVE_LEASE_FILE=str(execution/'lease.json'),
                            DUPLEX_LIVE_GUARD_FILE=str(control/'guard.json'),CUBLAS_WORKSPACE_CONFIG=':4096:8',
                            DUPLEX_NATIVE_EXECUTION_ID=uuid.uuid4().hex,HF_HOME=str(BASE/'hf-cache'))
                        cmd=command(Path.cwd(),gpu=devices[base],environment=env)+['training.personaplex.matched_evaluation',
                            '--source',source,'--trainer-source',PRODUCER,'--training-root',str(TRAIN),
                            '--root',str(OUTPUT),'--lease',str(execution/'lease.json'),
                            '--base',base,'--capacity',capacity,'--stage',stage]
                        if capacity=='narrow' and stage=='collect':
                            record=PINS/(base+'-narrow.json')
                            cmd+=['--reuse-record',str(record),'--reuse-record-sha',g.sha(record)]
                        with (execution/(name+'.log')).open('x') as log:
                            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                            write(execution/(base+'-current.json'),dict(stage=name,pid=child.pid,source=source))
                            try:
                                while child.poll() is None:ensure();time.sleep(5)
                            finally:
                                if child.poll() is None:
                                    os.killpg(child.pid,signal.SIGTERM)
                                    try:child.wait(timeout=60)
                                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                                write(execution/(name+'-terminal.json'),dict(actual_exit_code=child.returncode,source=source))
                        if child.returncode:raise RuntimeError('Actual evaluation child failed: '+name)
            except BaseException as error:errors.append(dict(base=base,error=repr(error)))
        threads=[threading.Thread(target=worker,args=(base,)) for base in devices]
        for thread in threads:thread.start()
        for thread in threads:thread.join()
        if errors:raise RuntimeError('Recovery incomplete: '+repr(errors))
        reports={}
        for base in devices:
            for capacity in ('baseline','narrow','broad'):
                name=base+'-'+capacity;path=OUTPUT/name/'report.json';value=json.loads(path.read_text())
                expected=OLD if capacity=='baseline' else source
                if (value.get('complete') is not True or len(value.get('cases',[]))!=32
                        or value['source']!=expected or value['trainer_source']!=PRODUCER):
                    raise ValueError('All six complete actual reports required')
                reports[name]=dict(path=str(path),sha256=g.sha(path),original_producer=value['source'],summary=value['summary'])
        write(OUTPUT/'comparison.json',dict(source=source,trainer_source=PRODUCER,reports=reports,
            complete_six_times32=True,reused_outputs=76,new_outputs=116,optimizer_updates=0,
            external_API_calls=0,connected_backend_tested=False,
            independent_reasoning_semantics_and_listening_still_required=True,adopted=False));code=0
    finally:
        if control is not None:health.stop(services,control,execution)
        write(execution/'terminal.json',dict(source=source,job_id=lease['job_id'],actual_exit_code=code,errors=errors,ended_unix=time.time()))


def staged_check(source):
    if json.loads((OUTPUT/'staging-complete.json').read_text())!={'source':source,'reused_outputs':76,'models_loaded':0}:
        raise ValueError('Exact prepared stage required')
    from .matched_evaluation_reuse import prepared
    for name,record in records().items():prepared(record,OUTPUT/name)
    result=dict(source=source,actual_container_staging_passed=True,reused_outputs=76,
                models_loaded=0,optimizer_updates=0,rsync_required_in_container=False)
    write(OUTPUT/'container-staging-proof.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('submit','run','staged-check'),required=True);parser.add_argument('--source',required=True)
    args=parser.parse_args();dict(submit=submit,run=run,**{'staged-check':staged_check})[args.mode](args.source)
