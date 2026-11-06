"""Exercise the complete scheduler wrapper without a model or allocation."""
import argparse,json,time
from training.venus import isambard_job as job

def test_complete_serial_evaluation_preserves_scheduler_environment(tmp_path,monkeypatch):
    real=tmp_path/'physical-project';real.mkdir()
    base=tmp_path/'base';base.symlink_to(real,target_is_directory=True);root=base/'evaluation';root.mkdir()
    monkeypatch.setattr(job,'BASE',base);monkeypatch.setattr(job,'ART',base/'artifacts')
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(job.subprocess,'check_output',lambda argv,**kwargs: 'source' if argv[0]=='git' else 'NodeList=nid1 StartTime=start EndTime=end')
    monkeypatch.setattr(job.g,'sha',lambda p:'auth');monkeypatch.setattr(job.g,'stamp',lambda s:time.time()+(7200 if s=='end' else -1))
    monkeypatch.setattr(job.g,'validate_job',lambda *a:None);monkeypatch.setattr(job.g,'ensure',lambda *a:None)
    calls=[]
    class Process:
        def __init__(self,argv,**kwargs):
            self.pid=len(calls)+100;self.returncode=0;calls.append(argv)
            if argv[0]!='apptainer':
                self.returncode=None
                (root/'execution-123/guard.json').write_text('{}')
        def poll(self):return self.returncode
        def terminate(self):self.returncode=0
        def wait(self,timeout=None):return self.returncode
    monkeypatch.setattr(job.subprocess,'Popen',Process)
    job.main(argparse.Namespace(source='source',root=str(root),worktree='source-native',stage='evaluate',epochs=2))
    native=[args for args in calls if args[0]=='apptainer']
    assert len(native)==8
    assert all('TMPDIR='+str(base/'tmp/native-123') in args for args in native)
    assert all(str(real)+':'+str(real) in args for args in native)
    assert all('SLURM_JOB_ID=123' in args and 'CUDA_VISIBLE_DEVICES=0' in args for args in native)
    assert [args[args.index('--stage')+1] for args in native]==['schema-selftest','reference','restore','reference','restore','collect','collect','combine']
    terminal=json.loads((root/'execution-123/terminal.json').read_text())
    assert terminal['actual_exit_code']==0 and terminal['new_optimizer_updates']==0
