"""Resume only actual successful postupdate windows, keeping worker immutable.

Host/controller source is recorded separately from the unchanged native worker.
All earlier allocations are queried live and carried; no model admission exists.
"""
import argparse,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from . import matched_guard as g
from .matched_job import BASE,visible_pair,write
from .matched_training_only import ROOT as INITIAL,HISTORY,FIRST_INTERVAL,STARTUP_AUTH,TrainingChildren,initial_plan,execute
from .matched_reuse import ORIGINAL_ROOT,DATA_SHA,preflight
from .matched_continue_policy import PACKET_SHA,read_pinned,restored_entry
from .matched_identity import DEPENDENCIES,NATIVE

CONTROL=('training/personaplex/matched_resume.py',NATIVE,'training/personaplex/matched_guard.py','training/personaplex/matched_continue_policy.py',
    'training/personaplex/matched_identity.py','training/personaplex/matched_reuse.py',
    'training/personaplex/matched_job.py','training/personaplex/matched_paths.py',
    'training/personaplex/matched_training_only.py','evals/moshi_model_download.py',str(STARTUP_AUTH))+DEPENDENCIES


def beneath(path):
    p=Path(path);p.resolve().relative_to(BASE.resolve())
    if p.is_symlink():raise ValueError('No aliased resume root')
    return p


def chain(previous,result_sha):
    roots=[];seen=set();p=beneath(previous);expected=result_sha
    while True:
        if str(p.resolve()) in seen or len(roots)>=4:raise ValueError('Repeated/too many prior continuation windows')
        seen.add(str(p.resolve()));result=read_pinned(p/'phase-result.json',expected)
        leases=list(p.glob('execution-*/lease.json'))
        if len(leases)!=1:raise ValueError('One actual execution per prior root required')
        lease=json.loads(leases[0].read_text());terminal=json.loads((leases[0].parent/'terminal.json').read_text())
        if (terminal.get('actual_exit_code')!=0 or terminal.get('source')!=lease['source']
                or terminal.get('job_id')!=lease['job_id'] or result.get('source')!=lease['source']
                or result.get('all_four_complete') is not False or not result.get('partial_lineages')):
            raise ValueError('Only successful incomplete training windows may resume')
        roots.append(dict(root=p,result=result,lease=lease))
        if p.resolve()==INITIAL.resolve():break
        p=beneath(result['previous_root']);expected=result['previous_result_sha256']
    ordered=list(reversed(roots))
    for number,row in enumerate(ordered,FIRST_INTERVAL):
        if row['lease']['continuation_interval']!=number:raise ValueError('Actual continuation interval sequence differs')
        if row['lease']['source']!=ordered[0]['lease']['source']:raise ValueError('Native worker changed across windows')
    return ordered


def allocation_carry(text,ordered,current_job):
    """Live sacct rows: JobIDRaw,JobName,Account,State,ExitCode,ElapsedRaw,AllocTRES."""
    wanted=set(HISTORY)|{r['lease']['job_id'] for r in ordered};rows={}
    for line in text.splitlines():
        f=line.split('|')
        if len(f)<7 or not f[0].isdigit():continue
        ident,name,account,state,exit_code,elapsed,tres=f[:7]
        if ident==current_job:continue
        resources=dict(x.split('=',1) for x in tres.split(',') if '=' in x)
        if ident not in wanted:
            if name.startswith('duplex-matched') and int(resources.get('gres/gpu','0')):
                raise ValueError('Uncounted matched GPU allocation')
            continue
        if ident in rows or account!=g.ACCOUNT or not elapsed.isdigit():raise ValueError('Invalid/duplicate actual scheduler row')
        rows[ident]=dict(state=state,exit=exit_code,elapsed=int(elapsed),GPUs=int(resources.get('gres/gpu','0')))
    if set(rows)!=wanted:raise ValueError('All actual phase allocations required')
    prior=0
    for ident,expected in HISTORY.items():
        row=rows[ident]
        if any(row[k]!=v for k,v in expected.items()) or row['exit']!=('0:0' if ident=='8216374' else '1:0'):raise ValueError('Pinned original allocation changed')
        prior+=row['elapsed']*row['GPUs']
    for item in ordered:
        lease=item['lease'];row=rows[lease['job_id']]
        if (row['state']!='COMPLETED' or row['exit']!='0:0' or row['GPUs']!=2
                or not 0<row['elapsed']<=lease['maximum_seconds'] or lease['prior_GPU_seconds']!=prior):
            raise ValueError('Actual successful window/time carry differs')
        prior+=row['elapsed']*2
    return prior


def verify_worker(work,worker_source,controller_source):
    work=beneath(work);current=Path.cwd()
    if subprocess.check_output(['git','-C',str(work),'rev-parse','HEAD'],universal_newlines=True).strip()!=worker_source:
        raise ValueError('Exact immutable native worker required')
    for file in CONTROL+(str(g.AUTH),str(g.EXTENSION)):
        original=subprocess.check_output(['git','-C',str(work),'show',worker_source+':'+file])
        controller=subprocess.check_output(['git','show',controller_source+':'+file])
        if original!=(work/file).read_bytes() or original!=controller or controller!=(current/file).read_bytes():
            raise ValueError('Controller/worker math, guard or policy differs: '+file)
    return work


def resume_plan(previous,original,root,execution,source):
    prior_packet=json.loads((previous/'joint-admission.json').read_text());prior=prior_packet['entries']
    if (prior_packet['source']!=source or prior_packet['data_report_sha256']!=DATA_SHA
            or prior_packet['original_joint_packet_sha256']!=PACKET_SHA):raise ValueError('Prior full matched packet changed')
    origins={(r['base'],r['capacity']):r for r in original['lineages']}
    if len(prior)!=4 or {(r['base'],r['capacity']) for r in prior}!=set(origins):raise ValueError('All four prior lineages required')
    fresh=initial_plan(original,root,execution);progress=0
    for entry in prior:
        key=(entry['base'],entry['capacity']);row=next(r for r in fresh if (r['base'],r['capacity'])==key)
        before=restored_entry(entry,source,DATA_SHA,origins[key]);selected=dict(entry)
        folder=previous/key[0]/key[1];window=folder/'train/training-window-terminal.json';complete=folder/'train/training-terminal.json'
        if complete.exists() or window.exists():
            terminal=complete if complete.exists() else window;value=json.loads(terminal.read_text())
            checkpoint=value['final_checkpoint'] if complete.exists() else value['checkpoint']
            selected.update(checkpoint=checkpoint,checkpoint_sha256=value['checkpoint_sha256'],
                restore=str(folder/'final-restore/restore.json'),restore_sha256=g.sha(folder/'final-restore/restore.json'),
                restore_terminal=entry['final_restore_terminal'],previous_training_terminal=str(terminal),
                previous_training_child_terminal=entry['training_child_terminal'])
        step=restored_entry(selected,source,DATA_SHA,origins[key])
        if step<before:raise ValueError('A saved optimizer step was discarded')
        if g.sha(selected['checkpoint'])!=selected['checkpoint_sha256']:raise ValueError('Actual resume checkpoint bytes changed')
        progress+=step-before
        row.update(checkpoint=selected['checkpoint'],checkpoint_sha256=selected['checkpoint_sha256'],restored_step=step)
        if step:
            for name in ('previous_training_terminal','previous_training_child_terminal'):row[name]=selected[name]
    if progress<=0:raise ValueError('No new actual updates: zero-update retries are forbidden')
    return fresh


def main(a):
    controller=subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()
    if controller!=a.controller_source or a.inputs_sha!=DATA_SHA:raise ValueError('Exact controller/data required')
    ordered=chain(a.previous_root,a.previous_result_sha);old=ordered[-1]['lease'];interval=old['continuation_interval']+1
    if interval>4:raise ValueError('Approved running interval count exhausted')
    amendment=g.sha(g.EXTENSION);g.phase_limit(dict(continuation_authorization_sha256=amendment,continuation_interval=interval))
    if any(item['lease']['continuation_authorization_sha256']!=amendment for item in ordered):raise ValueError('Original approval changed')
    source=old['source'];work=verify_worker(old['worker_worktree'],source,controller)
    original=preflight(ORIGINAL_ROOT,PACKET_SHA);job_id=os.environ['SLURM_JOB_ID']
    accounting=subprocess.check_output(['sacct','-n','-X','-P','-A',g.ACCOUNT,'--starttime=2026-11-08',
        '--format=JobIDRaw,JobName%100,Account,State,ExitCode,ElapsedRaw,AllocTRES'],universal_newlines=True,timeout=30)
    prior=allocation_carry(accounting,ordered,job_id)
    job=g.parse_job(subprocess.check_output(['scontrol','show','job','-o',job_id],universal_newlines=True,timeout=20))
    if not job.get('JobName','').startswith('duplex-matched-'):raise ValueError('Accounted matched job name required')
    lease=dict(provider='Slurm',resource_key=g.KEY,account=g.ACCOUNT,authorization_sha256=g.sha(g.AUTH),
        continuation_authorization_sha256=amendment,continuation_interval=interval,job_id=job_id,node=job['NodeList'],
        source=source,controller_source=controller,worker_worktree=str(work),start_unix=g.stamp(job['StartTime']),
        stop_unix=g.stamp(job['EndTime']),maximum_seconds=int(g.stamp(job['EndTime'])-g.stamp(job['StartTime'])),
        GPU_count=2,prior_GPU_seconds=prior)
    g.validate_job(job,lease,time.time());root=BASE/('native-moshi-matched-resume-'+job_id)
    root.mkdir(parents=True,exist_ok=False);execution=root/('execution-'+job_id);execution.mkdir()
    write(execution/'lease.json',lease);entries=resume_plan(Path(a.previous_root),original,root,execution,source)
    devices=visible_pair(os.environ['CUDA_VISIBLE_DEVICES']);watcher=None;code=1
    try:
        watcher=subprocess.Popen([sys.executable,'-B','-u','-m','training.personaplex.matched_guard',
            '--lease',str(execution/'lease.json'),'--output',str(execution/'guard.json')],
            stdout=(execution/'guard.log').open('x'),stderr=subprocess.STDOUT)
        deadline=time.time()+40
        while not (execution/'guard.json').exists():
            if watcher.poll() is not None or time.time()>deadline:raise RuntimeError('Independent resume guard did not start')
            time.sleep(1)
        children=TrainingChildren(work,source,execution,lease,watcher,
            dict(zip(('personaplex','context-spanning'),devices)),a.inputs,a.inputs_sha)
        partial=execute(children,entries,root,root/'joint-admission.json',source)
        write(root/'phase-result.json',dict(source=source,controller_source=controller,all_four_complete=not partial,
            partial_lineages=partial,previous_root=a.previous_root,previous_result_sha256=a.previous_result_sha,
            full_native_behavior_speech_backend_live_required=True,adopted=False));code=0
    finally:
        if watcher is not None:
            watcher.terminate()
            try:watcher.wait(timeout=10)
            except subprocess.TimeoutExpired:watcher.kill();watcher.wait()
        write(execution/'terminal.json',dict(source=source,controller_source=controller,job_id=job_id,actual_exit_code=code,
            ended_UTC=datetime.now(timezone.utc).isoformat(),adopted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('controller-source','inputs','inputs-sha','previous-root','previous-result-sha'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
