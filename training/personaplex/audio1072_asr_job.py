"""One bounded independent source-audio audit; no native optimizer path."""
import argparse,json,os,subprocess,time
from pathlib import Path
from .matched_paths import command,BASE


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=BASE/'native-reasoning-audio1072-20261113-v1');a=p.parse_args()
    work=Path.cwd();root=a.root
    if root.parent!=BASE or not root.name.startswith('native-reasoning-audio1072-'):raise ValueError('Explicit approved audio artifact root required')
    execution=root/('execution-'+os.environ['SLURM_JOB_ID']);execution.mkdir(exist_ok=False)
    cmd=command(work,gpu=os.environ['CUDA_VISIBLE_DEVICES'],environment={'HF_HOME':str(BASE/'hf-cache')})+[
        'evals.native_audio1072_asr','--plan',str(work/'docs/NATIVE-REASONING-AUDIO1072-SOURCE-PLAN-2026-11-13.json'),
        '--audio',str(root/'audio'),'--inventory',str(root/'inventory.json'),'--out',str(root/'asr')]
    with (execution/'asr.log').open('x') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    (execution/'terminal.json').write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],actual_exit=result.returncode,ended_unix=time.time(),native_optimizer_updates=0))+'\n')
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
