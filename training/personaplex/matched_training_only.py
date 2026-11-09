"""ONE initial training-only allocation from four proven zero-update states.

No numerical admission stage exists here. Subsequent windows must use a separate
verified postupdate resume entry; this initial-only root cannot be re-entered.
Host Python3.6; workers use the pinned container and original lineage metadata.
"""
import argparse,json,os,subprocess,sys,threading,time
from datetime import datetime,timezone
from pathlib import Path
from . import matched_guard as g
from .matched_job import BASE,Children,visible_pair,write
from .matched_reuse import ORIGIN,ORIGINAL_ROOT,DATA_SHA,preflight,actual_usage
from .matched_continue_policy import PACKET_SHA

ROOT=BASE/'native-moshi-matched-training-only-v1'
HISTORY={'8216374':dict(state='COMPLETED',elapsed=327,GPUs=1),
         '8229989':dict(state='FAILED',elapsed=951,GPUs=2)}


def prior_usage(job_id):
    text=subprocess.check_output(['sacct','-n','-X','-P','-A',g.ACCOUNT,'--starttime=2026-11-08',
        '--format=JobIDRaw,JobName%100,Account,State,ElapsedRaw,AllocTRES'],universal_newlines=True,timeout=30)
    selected=[]
    for line in text.splitlines():
        fields=line.split('|')
        if len(fields)<6:continue
        ident,name,account,state,elapsed,tres=fields[:6]
        if ident==job_id:continue
        if ident in HISTORY:selected.append('|'.join([ident,account,state,elapsed,tres]))
        elif name.startswith('duplex-matched'):
            resources=dict(p.split('=',1) for p in tres.split(',') if '=' in p)
            if int(resources.get('gres/gpu','0')):raise ValueError('Another matched GPU allocation is uncounted; initial-only entry forbidden')
    return actual_usage('\n'.join(selected),HISTORY)


class TrainingChildren(Children):
    def run(self,base,capacity,stage,folder,extra=(),label=None):
        if stage=='admit':raise ValueError('Passed admission may not be repeated')
        return super().run(base,capacity,stage,folder,tuple(extra)+('--lineage-source',ORIGIN),label)


def initial_plan(original,root,execution):
    return [dict(base=row['base'],capacity=row['capacity'],checkpoint=row['original_checkpoint'],
        checkpoint_sha256=row['checkpoint_sha256'],lineage_source=ORIGIN,
        training_terminal=str(root/row['base']/row['capacity']/'train/training-terminal.json'),
        training_child_terminal=str(execution/(row['base']+'-'+row['capacity']+'-train-terminal.json')),
        final_restore=str(root/row['base']/row['capacity']/'final-restore/restore.json'),
        final_restore_terminal=str(execution/(row['base']+'-'+row['capacity']+'-final-restore-terminal.json')))
        for row in original['lineages']]


def execute(children,entries,root,packet,source):
    lock=threading.Lock()
    def restore(base):
        for row in entries:
            if row['base']!=base:continue
            folder=root/base/row['capacity'];folder.mkdir(parents=True,exist_ok=False)
            terminal=children.run(base,row['capacity'],'restore',folder/'restore',[
                '--checkpoint',row['checkpoint'],'--checkpoint-sha',row['checkpoint_sha256']])
            with lock:row.update(restore=str(folder/'restore/restore.json'),
                restore_sha256=g.sha(folder/'restore/restore.json'),restore_terminal=str(terminal))
    children.parallel(restore)
    write(packet,dict(schema='dl-moshi-training-only-joint-v1',source=source,lineage_source=ORIGIN,
        data_report_sha256=DATA_SHA,original_joint_packet_sha256=PACKET_SHA,entries=entries))
    partial=[]
    def train(base):
        for capacity in ('narrow','broad'):
            row=next(r for r in entries if (r['base'],r['capacity'])==(base,capacity));folder=root/base/capacity
            if row.get('restored_step',0)==196:continue
            children.run(base,capacity,'train',folder/'train',[
                '--checkpoint',row['checkpoint'],'--checkpoint-sha',row['checkpoint_sha256'],
                '--restore-proof',row['restore'],'--restore-proof-sha',row['restore_sha256'],
                '--joint-admission',str(packet),'--joint-admission-sha',g.sha(packet)])
            window=folder/'train/training-window-terminal.json'
            final=json.loads((window if window.exists() else Path(row['training_terminal'])).read_text())
            checkpoint=final['checkpoint'] if window.exists() else final['final_checkpoint']
            children.run(base,capacity,'restore',folder/'final-restore',[
                '--checkpoint',checkpoint,'--checkpoint-sha',final['checkpoint_sha256']],label='final-restore')
            if window.exists():
                if final['total_lineage_updates']<=row.get('restored_step',0):raise ValueError('No zero-update retry window allowed')
                with lock:partial.append(dict(base=base,capacity=capacity,**final))
                return
    children.parallel(train)
    return partial


def main(a):
    source=subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()
    if source!=a.source or a.inputs_sha!=DATA_SHA:raise ValueError('Exact deployed producer and qualified data required')
    # Refuse missing/pending authorization before any model stage or output root.
    amendment=g.sha(g.EXTENSION)
    if g.phase_limit({'continuation_authorization_sha256':amendment,'continuation_interval':1})!=112*3600:
        raise ValueError('Recorded owner approval required')
    original=preflight(ORIGINAL_ROOT,PACKET_SHA)
    job_id=os.environ['SLURM_JOB_ID'];prior=prior_usage(job_id)
    job=g.parse_job(subprocess.check_output(['scontrol','show','job','-o',job_id],universal_newlines=True,timeout=20))
    if not job.get('JobName','').startswith('duplex-matched-'):raise ValueError('Accounted matched job name required')
    seconds=g.stamp(job['EndTime'])-g.stamp(job['StartTime'])
    lease=dict(provider='Slurm',resource_key=g.KEY,account=g.ACCOUNT,
        authorization_sha256=g.sha(g.AUTH),continuation_authorization_sha256=amendment,continuation_interval=1,
        job_id=job_id,node=job['NodeList'],source=source,worker_worktree=str(Path.cwd()),controller_source=source,start_unix=g.stamp(job['StartTime']),
        stop_unix=g.stamp(job['EndTime']),maximum_seconds=int(seconds),GPU_count=2,prior_GPU_seconds=prior)
    g.validate_job(job,lease,time.time())
    ROOT.mkdir(parents=True,exist_ok=False);execution=ROOT/('execution-'+job_id);execution.mkdir()
    write(execution/'lease.json',lease);write(execution/'original-reuse.json',original)
    devices=visible_pair(os.environ['CUDA_VISIBLE_DEVICES']);watcher=None;code=1
    try:
        watcher=subprocess.Popen([sys.executable,'-B','-u','-m','training.personaplex.matched_guard',
            '--lease',str(execution/'lease.json'),'--output',str(execution/'guard.json')],
            stdout=(execution/'guard.log').open('x'),stderr=subprocess.STDOUT)
        deadline=time.time()+40
        while not (execution/'guard.json').exists():
            if watcher.poll() is not None or time.time()>deadline:raise RuntimeError('Independent guard did not start')
            time.sleep(1)
        children=TrainingChildren(Path.cwd(),source,execution,lease,watcher,
            dict(zip(('personaplex','context-spanning'),devices)),a.inputs,a.inputs_sha)
        entries=initial_plan(original,ROOT,execution)
        partial=execute(children,entries,ROOT,ROOT/'joint-admission.json',source)
        write(ROOT/'phase-result.json',dict(source=source,all_four_complete=not partial,
            partial_lineages=partial,full_native_behavior_speech_backend_live_required=True,adopted=False))
        code=0
    finally:
        if watcher is not None:
            watcher.terminate()
            try:watcher.wait(timeout=10)
            except subprocess.TimeoutExpired:watcher.kill();watcher.wait()
        write(execution/'terminal.json',dict(source=source,job_id=job_id,actual_exit_code=code,
            ended_UTC=datetime.now(timezone.utc).isoformat(),adopted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','inputs','inputs-sha'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
