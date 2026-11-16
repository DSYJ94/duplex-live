"""One complete post-training comparison, no optimizer or new model API calls.

CPU input preparation is safe during training. GPU execution requires the
successful provider-COMPLETED196 parent and exact final restore/backup proofs.
"""
import argparse,json,os,subprocess,time,shutil,fcntl
from pathlib import Path
from .expanded_launch import container_command
from .expanded_guard import AUTH,ACCOUNT,KEY,sha,stamp,parse_job,validate_job
EVAL_AUTH=Path('docs/NATIVE-EXPANDED1072-EVALUATION-2026-11-16.json')


def evaluation_command(work,*,gpu,environment,backup,parent_backup):
    command=container_command(work,gpu=gpu,environment=environment,backup=backup)
    position=command.index('--pwd')
    command[position:position]=['--bind',str(parent_backup)+':'+str(parent_backup)+':ro']
    return command


def prepare(root,data,parent,source):
    from .expanded_evaluation import dev_cases
    from training.venus.foundation_native import cases
    root=Path(root);data=Path(data);parent=Path(parent)
    if root.resolve()==parent.resolve()or parent.resolve()in root.resolve().parents:raise ValueError('Separate sibling evaluation output required')
    if subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()!=source:raise ValueError('Committed preparation source required')
    cache=json.loads((parent/'codec/cache.json').read_text());capture=json.loads((parent/'capture/inputs.json').read_text())
    if (cache['source_manifest_sha256']!=sha(data/'manifest.json')or cache.get('complete1072')is not True
            or capture.get('complete1072')is not True):raise ValueError('Reuse qualified complete original source/capture only')
    root.mkdir(parents=True,exist_ok=False);rows=dev_cases(data,root/'dev-inputs')
    control=cases(Path.cwd())
    if len(control)!=32 or len(rows)!=64 or {r['id']for r in rows}&{r['id']for r in control}:raise ValueError('Whole distinct32+64 comparison required')
    for name,values in [('dev-cases.json',rows),('control-cases.json',control)]:
        (root/name).write_text(json.dumps(values,indent=2)+'\n')
    semantic=json.loads((data/'qualification/plan.json').read_text())
    by={r['id']:r for r in semantic['exact_semantic_records']}
    # This file is NEVER passed to the native engine or any worker/backend.
    (root/'scoring-only-reference.json').write_text(json.dumps([by[r['id']]for r in rows],indent=2)+'\n')
    report=dict(source=source,parent_root=str(parent),data=str(data),source_manifest_sha256=sha(data/'manifest.json'),
        capture_sha256=sha(parent/'capture/inputs.json'),control32_sha256=sha(root/'control-cases.json'),
        dev64_sha256=sha(root/'dev-cases.json'),dev_response_tail_seconds=60,required_outputs=192,
        same_RP_reference_as_training=True,teacher_audio_or_answers_enter_model=False,optimizer_updates=0,
        native_reasoning_and_routing_review_required=True,connected_backend_tested=False)
    (root/'input-preparation.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main(a):
    if a.stage=='prepare':prepare(a.root,a.data,a.parent_root,a.source);return
    scope=json.loads(EVAL_AUTH.read_text())
    expected=dict(status='OWNER_DIRECTED_POST_TRAINING_EVALUATION',maximum_simultaneous_voice_GPUs=4,maximum_evaluation_job_seconds=21600,maximum_evaluation_jobs_now=1,required_outputs=192,new_optimizer_updates=0,new_external_model_API_calls=0)
    if any(scope.get(k)!=v for k,v in expected.items()):raise ValueError('Exact owner-directed evaluation scope required')
    lock=(a.root/'.evaluation.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()!=a.source:raise ValueError('Exact evaluation source required')
    if a.backup is None or a.parent_backup is None:raise ValueError('Independent original and result backup paths required')
    if shutil.which('rsync')is None:raise ValueError('Actual host rsync required before GPU model execution')
    preparation=json.loads((a.root/'input-preparation.json').read_text())
    if (preparation['source']!=a.source or preparation['parent_root']!=str(a.parent_root)or preparation['required_outputs']!=192
            or sha(a.root/'dev-cases.json')!=preparation['dev64_sha256']or sha(a.root/'control-cases.json')!=preparation['control32_sha256']):raise ValueError('Pinned complete input preparation required')
    parent_raw=subprocess.check_output(['scontrol','show','job','-o',a.parent_job],universal_newlines=True,timeout=30)
    parent_job=parse_job(parent_raw)
    if parent_job.get('JobState')!='COMPLETED':raise ValueError('Never overlap valid parent training')
    job=os.environ['SLURM_JOB_ID'];actual=parse_job(subprocess.check_output(['scontrol','show','job','-o',job],universal_newlines=True,timeout=30))
    if actual.get('NodeList')in scope.get('exclude_active_Remora_nodes',[]):raise ValueError('Distinct active Remora node must remain protected')
    if stamp(actual['EndTime'])-stamp(actual['StartTime'])>21600:raise ValueError('Evaluation bounded to six hours/four GPUs')
    accounting=subprocess.check_output(['sacct','-A',ACCOUNT,'-S','2026-10-30','-X','-n','-P','-o','JobIDRaw,JobName%100,State,ElapsedRaw,AllocTRES,ReqTRES,TimelimitRaw'],universal_newlines=True,timeout=30)
    prior=reserved=0
    for line in accounting.splitlines():
        fields=line.split('|')
        if len(fields)<7 or fields[0]==job:continue
        ident,name,state,elapsed,allocated,requested,limit=fields[:7];tres=dict(x.split('=',1)for x in (allocated or requested).split(',')if '='in x);count=int(tres.get('gres/gpu',0));used=int(elapsed);prior+=count*used
        if state in ('RUNNING','PENDING','CONFIGURING'):
            if name.startswith('duplex-')and count:raise ValueError('Another voice reservation is active; no overlapping evaluation')
            if not limit.isdigit():raise ValueError('All concurrent reservations need explicit bounds')
            reserved+=count*max(0,int(limit)*60-used)
    quota=subprocess.check_output(['quota','-w','-v'],universal_newlines=True,timeout=20);line=next(r.split()for r in quota.splitlines()if r.strip().startswith('data1.vastp2.isambard.ac.uk:/p2/home '));used,hard=int(line[1])*1024,int(line[3])*1024
    if hard-used<12*1024**3:raise ValueError('Independent result backup headroom required')
    lease=dict(evaluation_authorization_sha256=sha(EVAL_AUTH),provider='Slurm',resource_key=KEY,account=ACCOUNT,authorization_sha256=sha(AUTH),GPU_count=4,job_id=job,node=actual['NodeList'],source=a.source,start_unix=stamp(actual['StartTime']),stop_unix=stamp(actual['EndTime']),prior_award_GPU_seconds=prior,other_reserved_GPU_seconds=reserved,home_used_bytes=used,home_hard_bytes=hard)
    validate_job(actual,lease,time.time());folder=a.root/('job-'+job);folder.mkdir(exist_ok=False);a.backup.mkdir(parents=True,exist_ok=True)
    lease_path=folder/'lease.json';lease_path.write_text(json.dumps(lease,indent=2)+'\n');(folder/'parent-provider.json').write_text(json.dumps(parent_job)+'\n');(folder/'provider-admission.txt').write_text(accounting+'\n'+quota)
    environment=dict(os.environ,DUPLEX_LIVE_GUARD_FILE=str(folder/'guard.json'),DUPLEX_LIVE_LEASE_FILE=str(lease_path))
    watch=subprocess.Popen(['python3','-u','-m','training.personaplex.expanded_guard','--lease',str(lease_path),'--output',str(folder/'guard.json')],env=environment)
    try:
        deadline=time.time()+180
        while not (folder/'guard.json').exists():
            if watch.poll()is not None or time.time()>deadline:raise ValueError('Actual scheduler guard unavailable')
            time.sleep(2)
        env={k:environment[k]for k in ('DUPLEX_LIVE_GUARD_FILE','DUPLEX_LIVE_LEASE_FILE','SLURM_JOB_ID','SLURM_JOB_ACCOUNT')};env.update(OMP_NUM_THREADS=8,MKL_NUM_THREADS=8,GIT_OPTIONAL_LOCKS='0',HF_HOME='/projects/u6ada/duplex-live/hf-cache')
        common=evaluation_command(Path.cwd(),gpu=False,environment=env,backup=a.backup,parent_backup=a.parent_backup)
        # Modern CPU process verifies final checkpoint bytes/restore proofs first.
        code="import json; from pathlib import Path; from training.personaplex.expanded_evaluation import completed; from training.personaplex.foundation_context import preflight; rows=preflight(Path.cwd()); selection=completed(Path(%r),%r,json.loads(Path(%r).read_text()),Path(%r)); plan=json.loads(Path(%r).read_text()); assert len(rows)==32; plan.update(selection=selection,parent_root=%r,source=%r,public_assets_and_native_geometry_preflight_passed=True); Path(%r).write_text(json.dumps(plan,indent=2)+'\\n')"%(str(a.parent_root),a.trainer_source,str(folder/'parent-provider.json'),str(a.parent_backup),str(a.root/'input-preparation.json'),str(a.parent_root),a.source,str(folder/'evaluation-plan.json'))
        subprocess.run(common[:-1]+['-c',code],env=environment,check=True)
        (a.root/'evaluation-admission.json').write_text(json.dumps(dict(lease=lease,plan=json.loads((folder/'evaluation-plan.json').read_text()),parent_provider=parent_job,optimizer_updates=0),indent=2)+'\n')
        command=evaluation_command(Path.cwd(),gpu=True,environment=env,backup=a.backup,parent_backup=a.parent_backup)
        command+=['torch.distributed.run','--standalone','--nproc_per_node=4','--module','training.personaplex.expanded_gpu','training.personaplex.expanded_evaluation','--root',str(a.root),'--plan',str(folder/'evaluation-plan.json'),'--lease',str(lease_path),'--source',a.source]
        subprocess.run(command,env=environment,check=True)
        # Existing inventory copier hashes/copies only this evaluation output.
        from .expanded_evaluation import complete_reports
        complete_reports(a.root,a.source)
        from .expanded_evaluation_archive import archive
        receipt=archive(a.root,a.backup/('job-'+job),a.source)
        (folder/'host-terminal.json').write_text(json.dumps(dict(success=True,required_outputs=192,optimizer_updates=0,backup=receipt,semantic_review_pending=True,adopted=False),indent=2)+'\n')
    finally:watch.terminate();watch.wait(timeout=30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run'])
    for key in ('root','data','parent-root'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('backup','parent-backup'):p.add_argument('--'+key,type=Path)
    p.add_argument('--source',required=True);p.add_argument('--parent-job',default='8478956');p.add_argument('--trainer-source',default='3c6f699e85adb26ba764e473ee491c041c62f6db');main(p.parse_args())
