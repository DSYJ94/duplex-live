"""Observe the SAME cache, then execute measured whole-epoch native training.

The original dispatcher is paused, never duplicated. Its independent guard and
provider STOP stay unchanged. Source is deployed through a NEW clean Git
worktree; cache and training producer identities are both explicit.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from . import controller as c

BASE=Path('runs/venus-training-20261026');PARENT=BASE/'native-v9'


class LiveParentGuard:
    def poll(self):
        try:
            g=json.loads((PARENT/'guard.json').read_text())
            return None if time.time()<min(g['expires_unix'],g['stop_unix']) else 1
        except (OSError,ValueError,KeyError):return 1


def main():
    lease=json.loads((PARENT/'lease.json').read_text());a=json.loads((PARENT/'ssh.json').read_text());pid=lease['controller_pid']
    birth=c.run(['/bin/ps','-p',str(pid),'-o','lstart=']).strip();assert birth==lease['controller_birth']
    assert 'T' in c.run(['/bin/ps','-p',str(pid),'-o','state='])
    assert str(lease['vm_id'])=='6904003500048518959' and str(lease['disk_id'])=='4375908109499109167'
    assert not c.run(['git','status','--porcelain','--','training/venus','data/venus']).strip()
    c.R=BASE/'sustain-v10';assert not c.R.exists(),'Observe same real process; never duplicate'
    c.R.mkdir();c.ssh_argv=a;c.guard=LiveParentGuard();assert c.guard.poll() is None
    c.source=c.run(['git','rev-parse','HEAD']).strip();c.remote=c.ROOT+'/duplex-venus-sustain-'+c.source[:12]
    parent_events=json.loads((PARENT/'events.json').read_text());cache_source=next(e['source'] for e in parent_events if e['event']=='warm_restart_preflight')
    c.UNITS.extend(['duplex-venus-native-proof-v8','duplex-venus-native-cache-v8'])
    finished=False
    try:
        c.record('same_native_cache_observed',cache_source=cache_source,training_source=c.source,vm_id=lease['vm_id'],native_stop_unix=lease['stop_unix'])
        # Deploy source while the unchanged cache job works. Never modify its worktree.
        c.run(['git','bundle','create',str(c.R/'source.bundle'),'HEAD'],120);c.sync(c.R/'source.bundle',c.ROOT+'/sustain-source-v10.bundle')
        c.ssh('git --git-dir='+c.ROOT+'/source.git fetch '+c.ROOT+'/sustain-source-v10.bundle HEAD && git --git-dir='+c.ROOT+'/source.git worktree add --detach '+c.remote+' '+c.source)
        c.wait_unit('duplex-venus-native-cache-v8',lease['stop_unix']-600)
        c.record('all188_native_cache_complete',cache_producer_source=cache_source)
        py=c.ROOT+'/venus-runtime-v1/venv/bin/python';env='CUBLAS_WORKSPACE_CONFIG=:4096:8 DUPLEX_LIVE_GUARD_FILE='+c.ROOT+'/venus-live-guard-v1.json DUPLEX_LIVE_LEASE_FILE='+c.ROOT+'/venus-native-gradient-lease-v8.json HF_HOME='+c.ROOT+'/venus-runtime-v1/hf-cache'
        def python(cmd):return '/usr/bin/env '+env+' '+py+' -u -m '+cmd
        args=' --model '+c.MODEL+' --cache '+c.OUT+'/native-cache-v8 --approval '+c.OUT+'/data-approval-v7.json --output '+c.OUT+'/candidate-v10 --source '+c.source+' --cache-producer '+cache_source+' --lease '+c.ROOT+'/venus-native-gradient-lease-v8.json'
        first='duplex-venus-native-numerical-start-v10';c.start_unit(first,python('training.venus.train --stage start'+args),max(1,int(lease['stop_unix']-time.time()-300)));c.wait_unit(first,lease['stop_unix']-300)
        recipe=json.loads(c.ssh('cat '+c.OUT+'/candidate-v10/recipe.json'))
        assert 2<=recipe['recipe']['epochs']<=8 and recipe['recipe']['updates']==38*recipe['recipe']['epochs']
        c.record('measured_complete_epoch_schedule',epochs=recipe['recipe']['epochs'],updates=recipe['recipe']['updates'],admission=recipe['admission'])
        sustained='duplex-venus-native-restore-sustain-v10';c.start_unit(sustained,python('training.venus.train --stage restore-continue'+args),max(1,int(lease['stop_unix']-time.time()-300)));c.wait_unit(sustained,lease['stop_unix']-300)
        terminal=json.loads(c.ssh('cat '+c.OUT+'/candidate-v10/training-terminal.json'))
        assert terminal['completed'] and terminal['actual_updates']==recipe['recipe']['updates'] and not terminal['adopted']
        finished=True;c.record('actual_complete_epoch_training_terminal',result=terminal)
    except BaseException as error:
        c.record('sustained_training_failure',error_type=type(error).__name__,error=str(error)[-2500:]);raise
    finally:
        try:
            for unit in c.UNITS:
                state=c.props(unit)
                if state['ActiveState']=='active' and state['SubState']!='exited':c.ssh('sudo systemctl stop '+unit,30)
            c.copyback()
        except Exception as error:c.record('copyback_error_disk_preserved',error=str(error)[-1500:])
        try:
            assert c.run(['/bin/ps','-p',str(pid),'-o','lstart=']).strip()==birth
            os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);c.record('original_dispatcher_retired',pid=pid)
        except ProcessLookupError:pass
        current=c.cloud('compute','instances','describe',c.NAME,'--zone='+lease['zone']);assert str(current['id'])==lease['vm_id']
        if current['status'] not in ('TERMINATED','STOPPED'):c.run(['gcloud','compute','instances','stop',c.NAME,'--zone='+lease['zone'],'--project='+c.PROJECT,'--quiet'],150)
        current=c.cloud('compute','instances','describe',c.NAME,'--zone='+lease['zone']);(c.R/'provider-final.json').write_text(json.dumps(current,indent=2)+'\n');c.record('provider_stopped',vm_id=current['id'],status=current['status'],training_complete=finished)


if __name__=='__main__':main()
