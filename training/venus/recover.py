"""Continue the SAME bounded allocation after diagnosed exact-word repairs.

No VM creation/restart, no new guard/lease, no new deadline or budget. Original
dispatch is paused; this records every actual recovery unit and preserves the
failed audits. Any failure stops billed compute after attempted full copyback.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from . import controller as c

PARENT=Path('runs/venus-training-20261026/allocation-v2')
BASE=Path('runs/venus-training-20261026')


class ExistingGuard:
    def poll(self):
        try:
            g=json.loads((PARENT/'guard.json').read_text())
            return None if time.time()<min(g['expires_unix'],g['stop_unix']) else 1
        except (OSError,ValueError,KeyError):return 1


def main():
    lease=json.loads((PARENT/'lease.json').read_text());a=json.loads((PARENT/'ssh.json').read_text())
    pid=lease['controller_pid'];birth=c.run(['/bin/ps','-p',str(pid),'-o','lstart=']).strip()
    assert birth==lease['controller_birth'] and 'T' in c.run(['/bin/ps','-p',str(pid),'-o','state='])
    assert str(lease['vm_id'])=='6904003500048518959' and str(lease['disk_id'])=='4375908109499109167'
    assert time.time()<lease['stop_unix']-1200
    c.R=BASE/'recovery-v7';assert not c.R.exists(),'Observe actual recovery process; never duplicate'
    c.R.mkdir();c.ssh_argv=a;c.guard=ExistingGuard();assert c.guard.poll() is None
    c.source=c.run(['git','rev-parse','HEAD']).strip();c.remote=c.ROOT+'/duplex-venus-recovery-'+c.source[:12]
    c.UNITS.extend(['duplex-venus-train-setup-v1','duplex-venus-train-fetch-v1','duplex-venus-train-asr-v1','duplex-venus-asr-repair-v5','duplex-venus-asr-repair-v6','duplex-venus-asr-repair-v7'])
    completed=False;review=None
    try:
        # Every prior GPU recognizer is terminal before another model is loaded.
        for unit in c.UNITS:
            state=c.props(unit);assert state['SubState'] in ('exited','failed')
        assert not c.ssh('nvidia-smi --query-compute-apps=pid --format=csv,noheader').strip()
        vm=c.cloud('compute','instances','describe',c.NAME,'--zone='+lease['zone']);assert str(vm['id'])==lease['vm_id'] and vm['status']=='RUNNING'
        c.record('same_live_allocation_recovery',source=c.source,vm_id=lease['vm_id'],disk_id=lease['disk_id'],unchanged_stop_unix=lease['stop_unix'],original_controller_pid=pid)
        c.run(['git','bundle','create',str(c.R/'source.bundle'),'HEAD'],120);c.sync(c.R/'source.bundle',c.ROOT+'/recovery-source.bundle')
        c.ssh('git --git-dir='+c.ROOT+'/source.git fetch '+c.ROOT+'/recovery-source.bundle HEAD && git --git-dir='+c.ROOT+'/source.git worktree add --detach '+c.remote+' '+c.source)
        c.sync(str(BASE/'timelines-v7-repaired')+'/',c.OUT+'/timelines-v7-repaired/')
        c.sync(BASE/'word-quality-complete-v7.json',c.OUT+'/word-quality-complete-v7.json')
        py=c.ROOT+'/venus-runtime-v1/venv/bin/python'
        env='DUPLEX_LIVE_GUARD_FILE='+c.ROOT+'/venus-live-guard-v1.json DUPLEX_LIVE_LEASE_FILE='+c.ROOT+'/venus-training-lease.json HF_HOME='+c.ROOT+'/venus-runtime-v1/hf-cache'
        def python(cmd):return '/usr/bin/env '+env+' '+py+' -u -m '+cmd
        proof='duplex-venus-native-proof-v7'
        c.start_unit(proof,python('training.venus.native_cache --model '+c.MODEL+' --timelines '+c.OUT+'/timelines-v7-repaired --output '+c.OUT+'/native-proof-v7 --proof-only'),600)
        prepared=BASE/'prepared-semantic-v7.json';payload=json.loads(prepared.read_text())
        assert payload['maximum_total_cost_usd']<=4 and payload['maximum_requests']==8
        quality=json.loads((BASE/'word-quality-complete-v7.json').read_text());assert quality['all_passed'] and quality['requested']==quality['passed']==159
        log=(c.R/'semantic-review.log').open('x')
        review=subprocess.Popen([sys.executable,'-u','-m','data.venus.semantic','execute','--prepared',str(prepared),'--output',str(c.R/'semantic-v7')],stdout=log,stderr=subprocess.STDOUT)
        c.record('independent_semantic_review_started',pid=review.pid,prepared_sha256=hashlib.sha256(prepared.read_bytes()).hexdigest(),scenes=224,maximum_requests=8,maximum_usd=payload['maximum_total_cost_usd'])
        c.wait_unit(proof,min(lease['stop_unix']-300,time.time()+650))
        until=time.time()+180
        while review.poll() is None:
            if time.time()>until or c.guard.poll() is not None:review.terminate();raise RuntimeError('Review blocked; stop paid compute')
            time.sleep(3)
        log.close();assert review.returncode==0,'Complete224 independent semantic review failed'
        approval=c.R/'data-approval-v7.json'
        c.run([sys.executable,'-m','data.venus.semantic','approve','--timelines',str(BASE/'timelines-v7-repaired'),'--word-quality',str(BASE/'word-quality-complete-v7.json'),'--audit',str(BASE/'source-audit-v7-repaired.json'),'--prepared',str(prepared),'--semantic-quality',str(c.R/'semantic-v7/semantic-quality.json'),'--output',str(approval)])
        c.sync(approval,c.OUT+'/data-approval-v7.json');c.record('complete_data_approved',approval_sha256=hashlib.sha256(approval.read_bytes()).hexdigest())
        cache='duplex-venus-native-cache-v7'
        c.start_unit(cache,python('training.venus.native_cache --model '+c.MODEL+' --timelines '+c.OUT+'/timelines-v7-repaired --approval '+c.OUT+'/data-approval-v7.json --output '+c.OUT+'/native-cache-v7'),max(1,int(lease['stop_unix']-time.time()-300)))
        c.wait_unit(cache,lease['stop_unix']-300)
        args=' --model '+c.MODEL+' --cache '+c.OUT+'/native-cache-v7 --approval '+c.OUT+'/data-approval-v7.json --output '+c.OUT+'/candidate-v7 --source '+c.source+' --lease '+c.ROOT+'/venus-training-lease.json'
        start='duplex-venus-numerical-start-v7';c.start_unit(start,python('training.venus.train --stage start'+args),max(1,int(lease['stop_unix']-time.time()-300)));c.wait_unit(start,lease['stop_unix']-300)
        sustain='duplex-venus-restore-sustain-v7';c.start_unit(sustain,python('training.venus.train --stage restore-continue'+args),max(1,int(lease['stop_unix']-time.time()-300)));c.wait_unit(sustain,lease['stop_unix']-300)
        completed=True;c.record('actual_training_complete',candidate_updates=304,adopted=False,behavioral_tests_required=True)
    except BaseException as error:
        c.record('recovery_failure',error_type=type(error).__name__,error=str(error)[-2500:])
        raise
    finally:
        if review and review.poll() is None:review.terminate()
        try:
            for unit in c.UNITS:
                state=c.props(unit)
                if state['ActiveState']=='active' and state['SubState']!='exited':c.ssh('sudo systemctl stop '+unit,30)
            c.copyback()
        except Exception as error:c.record('copyback_error_disk_preserved',error=str(error)[-1500:])
        # Retire only the verified paused dispatcher. Its existing guard also
        # stops this exact VM when that controller identity disappears.
        try:
            assert c.run(['/bin/ps','-p',str(pid),'-o','lstart=']).strip()==birth
            os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
            c.record('original_paused_dispatcher_retired',pid=pid)
        except ProcessLookupError:pass
        current=c.cloud('compute','instances','describe',c.NAME,'--zone='+lease['zone']);assert str(current['id'])==lease['vm_id']
        if current['status'] not in ('TERMINATED','STOPPED'):
            c.run(['gcloud','compute','instances','stop',c.NAME,'--zone='+lease['zone'],'--project='+c.PROJECT,'--quiet'],150)
        current=c.cloud('compute','instances','describe',c.NAME,'--zone='+lease['zone']);(c.R/'provider-final.json').write_text(json.dumps(current,indent=2)+'\n');c.record('provider_stopped',status=current['status'],vm_id=current['id'],training_complete=completed)


if __name__=='__main__':main()
