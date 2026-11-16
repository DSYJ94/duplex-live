"""One-shot successful-parent handover. Never stops a job or repeats learning."""
import json,subprocess,time,fcntl,sys
from pathlib import Path
BASE=Path('/projects/u6ada/duplex-live')
ROOT=BASE/'native-expanded1072-evaluation-v1'
SOURCE='069bc6534b794937632cbedb657c92e51c89d531'
WORK=BASE/'source-expanded-eval-069bc65'
sys.path.insert(0,str(WORK))
def emit(**v):print(json.dumps(dict(observed_unix=time.time(),**v)),flush=True)
def main():
    lock=(ROOT/'.handover-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    marker=ROOT/'submission.json'
    if marker.exists():raise ValueError('Submission marker exists; never duplicate')
    previous=None;unknown=0
    while time.time()<1794849000:
        try:
            raw=subprocess.check_output(['scontrol','show','job','-o','8478956'],universal_newlines=True,timeout=30)
            from training.personaplex.expanded_guard import parse_job
            parent=parse_job(raw);state=parent['JobState'];unknown=0
        except Exception as exc:
            unknown+=1;emit(event='provider_observation_unavailable',consecutive=unknown,error=repr(exc))
            if unknown>=3:raise RuntimeError('Observation unknown; no submission or cancellation')
            time.sleep(60);continue
        if state!=previous:emit(event='parent_provider',state=state);previous=state
        if state=='COMPLETED':break
        if state not in ('RUNNING','COMPLETING'):raise ValueError('Parent stopped without successful completion: '+state)
        time.sleep(60)
    else:raise TimeoutError('Bounded handover wait ended; training left untouched')
    provider=ROOT/'completed-parent-provider.json';provider.write_text(json.dumps(parent,indent=2)+'\n')
    from training.personaplex.expanded_launch import container_command
    backup=Path('/home/u6ada/dsyj94.u6ada/duplex-expanded1072-backups-v1')
    command=container_command(WORK,gpu=False,environment={'OMP_NUM_THREADS':'1','GIT_OPTIONAL_LOCKS':'0'},backup=backup)
    code="import json;from pathlib import Path;from training.personaplex.expanded_evaluation import completed;result=completed(Path('/projects/u6ada/duplex-live/native-expanded1072-phase-v1'),'3c6f699e85adb26ba764e473ee491c041c62f6db',json.loads(Path(%r).read_text()),Path(%r));Path(%r).write_text(json.dumps(result,indent=2)+'\\n')"%(str(provider),str(backup),str(ROOT/'verified-final-parent.json'))
    subprocess.run(command[:-1]+['-c',code],check=True)
    if any(ROOT.glob('job-*')):raise ValueError('Prior evaluation allocation exists; no duplicate')
    # The exclusive durable intent survives ambiguous sbatch/network outcomes.
    # It is never removed or automatically retried, even if sbatch fails.
    with marker.open('x')as f:json.dump(dict(status='SUBMISSION_INTENT_NO_AUTOMATIC_RETRY',source=SOURCE,parent='8478956',actual_unix=time.time()),f)
    result=subprocess.check_output(['sbatch','--parsable',str(BASE/'native-expanded1072-evaluation-job-v1.sh')],universal_newlines=True,timeout=30).strip()
    if not result.split(';')[0].isdigit():raise ValueError('Ambiguous submission result; do not resubmit: '+result)
    marker.write_text(json.dumps(dict(status='SUBMITTED_ONCE',job=result,source=SOURCE,parent='8478956',actual_unix=time.time()),indent=2)+'\n')
    emit(event='evaluation_submitted_once',job=result,new_optimizer_updates=0)
if __name__=='__main__':
    try:main()
    except Exception as exc:emit(event='handover_stopped_no_retry',error=repr(exc));raise
