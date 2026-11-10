"""Read-only bounded fresh-SSH probes; never submits, cancels or touches OneDrive.

An authenticated old socket is not evidence of fresh monitoring. One failed or
timed-out probe is surfaced immediately; no silent persistent-transport wait.
This cannot alert while the Mac/app itself is asleep or disconnected.
"""
import argparse,json,subprocess,time
from pathlib import Path

REMOTE='''import json,subprocess,time
from pathlib import Path
root=Path(ROOT);job=JOB;ex=root/('execution-'+job)
def run(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=10)
 return dict(actual_exit=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())
value=dict(actual_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),job=job,
 queue=run(['squeue','-h','-j',job,'-o','%i|%T|%M|%N']))
terminal=ex/'terminal.json'
if terminal.exists():value['host_terminal']=json.loads(terminal.read_text())
if value.get('host_terminal') or value['queue']['actual_exit'] or not value['queue']['stdout']:
 value['sacct']=run(['sacct','-n','-X','-P','-j',job,'--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'])
print(json.dumps(value),flush=True)
'''


def probe(job,root,runner=subprocess.run):
    if not str(job).isdigit():raise ValueError('One actual scheduler job ID required')
    path=Path(root)
    path.relative_to('/home/u6ada/dsyj94.u6ada/duplex-native-recovery-20261109-v1/runs')
    if '..' in path.parts:raise ValueError('No aliased observation root')
    script='ROOT='+repr(str(path))+';JOB='+repr(str(job))+'\n'+REMOTE
    command=['ssh','-S','none','-o','ControlMaster=no','-o','ControlPath=none','-o','BatchMode=yes',
        '-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=5',
        '-o','ServerAliveCountMax=2','u6ada.aip2.isambard','timeout','--kill-after=3s','25s','/usr/bin/python3','-B','-u','-']
    result=runner(command,input=script,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        universal_newlines=True,timeout=35)
    if result.returncode:raise RuntimeError('Fresh observation failed: '+str(result.returncode)+' '+result.stderr[-1000:])
    value=json.loads(result.stdout)
    if value.get('job')!=str(job):raise ValueError('Actual job observation identity differs')
    return value


def main(a):
    last=None
    while True:
        now=time.time()
        if last is not None and now-last>90:
            print(json.dumps(dict(event='monitoring_gap',job=a.job,seconds=now-last)),flush=True)
        try:value=probe(a.job,a.root)
        except Exception as exc:
            print(json.dumps(dict(event='MONITORING_BLOCKED',job=a.job,error=type(exc).__name__+': '+str(exc))),flush=True);raise
        print(json.dumps(value),flush=True);last=time.time()
        if a.once or value.get('host_terminal') or value['queue']['actual_exit'] or not value['queue']['stdout']:return
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',required=True);p.add_argument('--root',required=True)
    p.add_argument('--once',action='store_true');main(p.parse_args())
