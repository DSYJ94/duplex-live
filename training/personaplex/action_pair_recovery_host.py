"""One known-cause recovery, sharing the original paired phase time bound."""
import argparse,json,os,subprocess,time
from pathlib import Path
from .expanded_guard import PAIR_RECOVERY_AUTH,sha
from .action_pair_host import admission,pair_command
from .action_pair_recovery import OLD,ORIGIN,completed_parent,clone_raw,execute_recovery
from .native_feature_host import archive


def main(a):
    if subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()!=a.source:
        raise ValueError('Exact immutable recovery source required')
    scope=json.loads(PAIR_RECOVERY_AUTH.read_text())
    raw=subprocess.check_output(['sacct','-j','8540802','-X','-n','-P','-o','JobIDRaw,State,Account,ElapsedRaw,ExitCode'],universal_newlines=True,timeout=30)
    rows=[r.split('|')for r in raw.splitlines()if r.startswith('8540802|')]
    if len(rows)!=1 or rows[0][1:5]!=['FAILED','brics.u6ada','5394','1:0']:
        raise ValueError('Exact terminal failed parent and all actual usage required')
    boundary,inventory=completed_parent(scope)
    a.root.mkdir(exist_ok=False);a.backup.mkdir(parents=True,exist_ok=True)
    clone=clone_raw(OLD/'readout/evaluate',a.root/'readout/evaluate',inventory)
    plan=json.loads((OLD/'plan.json').read_text());plan['source']=a.source
    path=a.root/'plan.json';path.write_text(json.dumps(plan,indent=2)+'\n')
    lease,provider=admission(a,path,recovery=True)
    folder=a.root/('job-'+lease['job_id']);folder.mkdir()
    lease_path=folder/'lease.json';lease_path.write_text(json.dumps(lease,indent=2)+'\n')
    (folder/'provider-admission.txt').write_text(provider+'\n'+raw)
    (folder/'preserved-parent.json').write_text(json.dumps(dict(boundary=boundary,clone=clone,generation_source=ORIGIN),indent=2)+'\n')
    env=dict(os.environ,DUPLEX_LIVE_GUARD_FILE=str(folder/'guard.json'),DUPLEX_LIVE_LEASE_FILE=str(lease_path))
    watch=subprocess.Popen(['python3','-u','-m','training.personaplex.expanded_guard','--lease',str(lease_path),'--output',str(folder/'guard.json')],env=env)
    try:
        until=time.time()+180
        while not(folder/'guard.json').exists():
            if watch.poll()is not None or time.time()>until:raise ValueError('Actual independent guard unavailable')
            time.sleep(2)
        inside={k:env[k]for k in ('DUPLEX_LIVE_GUARD_FILE','DUPLEX_LIVE_LEASE_FILE','SLURM_JOB_ID','SLURM_JOB_ACCOUNT')}
        command=pair_command(Path.cwd(),inside,a.backup,a.inputs)
        position=command.index('--pwd');command[position:position]=['--bind',str(OLD)+':'+str(OLD)+':ro']
        def run(mode,stage):
            common=['--root',str(a.root),'--lease',str(lease_path),'--plan',str(path),'--inputs',str(a.inputs),'--source',a.source]
            if stage=='audit-saved':
                module='training.personaplex.action_pair_audit_recovery';args=common
            else:
                if mode!='bias':raise ValueError('Never repeat completed readout training or generation')
                module='training.personaplex.action_pair_worker';args=[stage,'--mode','bias']+common
            subprocess.run(command+['training.personaplex.expanded_gpu',module]+args,env=env,check=True)
            result=json.loads((a.root/mode/('evaluate'if stage=='audit-saved'else stage)/'terminal.json').read_text())
            if stage=='fit':
                if result['actual_new_readout_updates']!=128 or result['backbone_updates']!=0:raise ValueError('Only first128bias updates allowed')
            elif not result['passed']or result['outputs']!=96 or result['optimizer_updates']!=0:
                raise ValueError('Complete96 audit/restore/evaluation required')
        def preserve(mode,stage,updates):
            receipt=archive(a.root/mode/stage,a.backup/('job-'+lease['job_id'])/mode/stage,optimizer_updates=updates)
            (folder/(mode+'-'+stage+'-backup.json')).write_text(json.dumps(receipt,indent=2)+'\n');return receipt
        receipts=execute_recovery(run,preserve)
        (folder/'host-terminal.json').write_text(json.dumps(dict(success=True,source=a.source,
            parent_job='8540802',parent_GPU_seconds=5394,readout_generation_source=ORIGIN,
            readout_updates_repeated=0,readout_generations_repeated=0,new_bias_updates=128,
            total_phase_head_updates=256,complete_native_outputs=192,backups=receipts,
            parent_boundary=boundary,native_backbone_updates=0,product_accepted=False),indent=2)+'\n')
    finally:watch.terminate();watch.wait(timeout=30)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','backup','inputs'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source',required=True);main(p.parse_args())
