"""One LOCAL wait/transfer/Git-deploy/submit path for the approved derived cohort."""
import argparse,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path


def run(args):return subprocess.run(args,check=True,text=True,capture_output=True).stdout

def main():
    p=argparse.ArgumentParser()
    for n in ('plan','audio','claim'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();work=Path.cwd();plan=a.plan.resolve();audio=a.audio.resolve();a.claim.mkdir(parents=True,exist_ok=False)
    source=run(['git','rev-parse','HEAD']).strip();config=json.loads(plan.read_text())
    if not config['owner_approved'] or config['maximum_total_API_USD']!=100 or len(config['exact_speech_payloads'])!=2144:raise ValueError('Exact complete approved cohort required')
    metadata=dict(pid=os.getpid(),source=source,plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),audio=str(audio),stage='waiting_for_real_complete_collection',native_optimizer_updates=0)
    (a.claim/'controller.json').write_text(json.dumps(metadata,indent=2)+'\n')
    while not (audio/'collection-complete.json').exists():
        if not (audio/'writer.lock').exists():raise RuntimeError('Renderer stopped incomplete; preserve artifacts, do not submit')
        time.sleep(10)
    if (audio/'writer.lock').exists():raise RuntimeError('Completion appeared while writer remains active; inspect before submission')
    inventory=a.claim/'inventory.json'
    print(run([sys.executable,'-B','-m','data.synthetic.reasoning_audio_inventory','--plan',str(plan),'--audio',str(audio),'--out',str(inventory)]),flush=True)
    # rsync-ar preserves symlinks. Materialize ONLY these authorized artifacts as
    # hard links in a staging directory, avoiding Mac-only symlink targets.
    stage=a.claim/'transfer';stage.mkdir()
    for row in config['exact_speech_payloads']:
        for ext in ('.wav','.json'):os.link((audio/(row['id']+ext)).resolve(),stage/(row['id']+ext))
    remote='/projects/u6ada/duplex-live/'+audio.name;host='u6ada.aip2.isambard';bundle=a.claim/'source.bundle'
    run(['git','bundle','create',str(bundle),'main'])
    run(['ssh','-o','BatchMode=yes',host,'mkdir -p '+shlex.quote(remote+'/audio')])
    run(['rsync','-ar','--no-owner','--no-group','--exclude=session-token',str(stage)+'/',host+':'+remote+'/audio/'])
    run(['rsync','-ar','--no-owner','--no-group','--exclude=session-token',str(inventory),str(bundle),host+':'+remote+'/'])
    common='/projects/u6ada/duplex-live/source-audio1072-timestamps-v1';target='/projects/u6ada/duplex-live/source-derived1072-'+source[:7]
    deploy='git -C '+shlex.quote(common)+' fetch '+shlex.quote(remote+'/source.bundle')+' main && git -C '+shlex.quote(common)+' worktree add --detach '+shlex.quote(target)+' '+source
    print(run(['ssh','-o','BatchMode=yes',host,deploy]),flush=True)
    queue=run(['ssh','-o','BatchMode=yes',host,'squeue -u "$(id -un)" -h -o %A']).split()
    if any(not x.isdigit() for x in queue):raise ValueError('Unknown queue ownership; no submission')
    (a.claim/'queue-before-submit.json').write_text(json.dumps(dict(existing_jobs=queue,serialization='afterany_all_current_owner_jobs'))+'\n')
    wrapper=shlex.join(['/usr/bin/python3','-B','-u','-m','training.personaplex.audio1072_asr_job','--root',remote,'--plan',target+'/'+str(plan.relative_to(work)),'--complete-timestamps','--word-policy','math-spoken-v1'])
    cmd=['sbatch','--parsable','--account=brics.u6ada','--partition=workq','--gpus=1','--time=01:00:00','--cpus-per-task=8','--mem=80G','--no-requeue','--job-name=duplex-derived1072-asr','--chdir='+target,'--output='+remote+'/slurm-%j.log','--wrap='+wrapper]
    if queue:cmd+=['--dependency=afterany:'+':'.join(sorted(set(queue)))]
    (a.claim/'command.json').write_text(json.dumps(dict(source=source,command=cmd,remote_root=remote,native_optimizer_updates=0))+'\n')
    result=subprocess.run(['ssh','-o','BatchMode=yes',host,shlex.join(cmd)],text=True,capture_output=True,timeout=90)
    (a.claim/'response.json').write_text(json.dumps(dict(exit=result.returncode,stdout=result.stdout,stderr=result.stderr))+'\n')
    if result.returncode:raise RuntimeError('Submission failed; inspect actual outcome before any retry')
    job=result.stdout.strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Unknown outcome; do not resubmit')
    (a.claim/'job.json').write_text(json.dumps(dict(job_id=job,source=source,remote_root=remote))+'\n');print('SUBMITTED '+job,flush=True)


if __name__=='__main__':main()
