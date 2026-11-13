"""LOCAL single submitter: await actual complete render, inventory, transfer, audit."""
import json,os,subprocess,sys,time,shlex
from pathlib import Path


def run(args):return subprocess.run(args,check=True,text=True,capture_output=True).stdout

def main():
    work=Path.cwd();audio=work/'runs/native-reasoning-audio1072-20261113-v1'
    claim=work/'runs/audio1072-qualification-submission-v1';claim.mkdir(exist_ok=False)
    while not (audio/'collection-complete.json').exists():
        if not (audio/'writer.lock').exists():raise RuntimeError('Renderer stopped incomplete; do not submit')
        time.sleep(10)
    inventory=claim/'inventory.json'
    print(run([sys.executable,'-B','-m','data.synthetic.reasoning_audio_inventory','--plan',str(work/'docs/NATIVE-REASONING-AUDIO1072-SOURCE-PLAN-2026-11-13.json'),'--audio',str(audio),'--out',str(inventory)]),flush=True)
    source=run(['git','rev-parse','HEAD']).strip()
    remote='/projects/u6ada/duplex-live/native-reasoning-audio1072-20261113-v1'
    host='u6ada.aip2.isambard';bundle=claim/'source.bundle'
    run(['git','bundle','create',str(bundle),'main'])
    run(['rsync','-ar','--no-owner','--no-group','--exclude=session-token',str(audio)+'/',host+':'+remote+'/audio/'])
    run(['rsync','-ar','--no-owner','--no-group','--exclude=session-token',str(inventory),str(bundle),host+':'+remote+'/'])
    original='/home/u6ada/dsyj94.u6ada/duplex-native-recovery-20261109-v1/source-eval-recovery-ebd7128'
    target='/projects/u6ada/duplex-live/source-audio1072-'+source[:7]
    deploy='git -C '+original+' fetch '+remote+'/source.bundle main && git -C '+original+' worktree add --detach '+target+' '+source
    print(run(['ssh','-o','BatchMode=yes',host,deploy]),flush=True)
    cmd=['sbatch','--parsable','--account=brics.u6ada','--partition=workq','--gpus=1','--time=04:00:00','--cpus-per-task=8','--mem=80G','--no-requeue','--job-name=duplex-audio1072-asr','--dependency=afterany:8400058','--chdir='+target,'--output='+remote+'/slurm-%j.log','--wrap=/usr/bin/python3 -B -u -m training.personaplex.audio1072_asr_job']
    (claim/'command.json').write_text(json.dumps(dict(source=source,command=cmd,native_optimizer_updates=0))+'\n')
    # Ambiguous submission is never retried automatically.
    result=subprocess.run(['ssh','-o','BatchMode=yes',host,shlex.join(cmd)],text=True,capture_output=True,timeout=90)
    (claim/'response.json').write_text(json.dumps(dict(exit=result.returncode,stdout=result.stdout,stderr=result.stderr))+'\n')
    if result.returncode:raise RuntimeError('Submission failed; inspect before any retry')
    job=result.stdout.strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Unknown submission outcome; no retry')
    (claim/'job.json').write_text(json.dumps(dict(job_id=job,source=source))+'\n');print('SUBMITTED '+job,flush=True)


if __name__=='__main__':main()
