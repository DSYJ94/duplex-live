"""Exercise the complete scheduler wrapper without a model or allocation."""
import argparse,json,time
import pytest
from training.venus import isambard_job as job

@pytest.mark.parametrize('stage,expected,fold', [('evaluate',8,False),('posttrain-evaluate',2,False),('posttrain-evaluate',2,True),('retention-source-audit',1,False),('release-evaluate',1,False),('retention-semantic',1,False)])
def test_complete_serial_evaluation_preserves_scheduler_environment(tmp_path,monkeypatch,stage,expected,fold):
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
    job.main(argparse.Namespace(source='source',root=str(root),worktree='source-native',stage=stage,epochs=2,merge_lora=fold))
    native=[args for args in calls if args[0]=='apptainer']
    assert len(native)==expected
    assert all('--containall' in args and '--cleanenv' in args for args in native)
    invocations=[next(x for x in args if x.startswith('DUPLEX_NATIVE_EXECUTION_ID=')) for args in native]
    assert len(set(invocations))==expected
    last_name={'evaluate':'combine','posttrain-evaluate':'compare','retention-source-audit':'source-audit','release-evaluate':'release-reference','retention-semantic':'retention-semantic'}[stage]
    last=json.loads((root/('execution-123/'+last_name+'-terminal.json')).read_text())
    assert last['execution_id']==invocations[-1].split('=',1)[1] and last['pid']>=100
    assert all('TMPDIR='+str(base/'tmp/native-123') in args for args in native)
    assert all(str(real)+':'+str(real) in args for args in native)
    assert all('SLURM_JOB_ID=123' in args and 'CUDA_VISIBLE_DEVICES=0' in args for args in native)
    assert all(('--merge-lora' in args)==fold for args in native)
    if fold:assert all(args[args.index('--baseline-profile')+1]=='unmerged646' for args in native)
    if stage=='retention-source-audit':
        assert native[0][native[0].index('-m')+1]=='data.venus.retention_quality'
        assert '--merge-lora' not in native[0] and '--model' not in native[0]
        assert native[0][native[0].index('--original')+1]==str(root/'original')
    elif stage=='release-evaluate':
        assert native[0][native[0].index('-m')+1]=='training.venus.foundation_compare'
        assert '--current-runtime-reference' in native[0]
        assert '--candidate-checkpoint' not in native[0] and '--private-action-repair' not in native[0]
    elif stage=='retention-semantic':
        assert native[0][native[0].index('-m')+1]=='data.venus.retention_semantics'
        assert '--checkpoint' not in native[0] and '--opal-source' not in native[0]
    else:
        assert [args[args.index('--stage')+1] for args in native]==(['schema-selftest','reference','restore','reference','restore','collect','collect','combine'] if stage=='evaluate' else ['collect','compare'])
    terminal=json.loads((root/'execution-123/terminal.json').read_text())
    assert terminal['actual_exit_code']==0 and terminal['new_optimizer_updates']==0

def test_inference_fold_cannot_be_used_in_training():
    with pytest.raises(ValueError,match='separate post-training'):
        job.main(argparse.Namespace(stage='train',merge_lora=True))
