"""Exercise the complete scheduler wrapper without a model or allocation."""
import argparse,json,time
import pytest
from training.venus import isambard_job as job

@pytest.mark.parametrize('stage,expected,fold', [('evaluate',8,False),('posttrain-evaluate',2,False),('posttrain-evaluate',2,True),('retention-source-audit',1,False),('release-evaluate',1,False),('retention-semantic',1,False),('retention-qwen',1,False),('retention-qwen-v2',1,False),('retention-cache',1,False),('retention-train',4,False),('retention-qwen-measured',1,False),('retention-evaluate',4,False),('foundation-reasoning',1,False),('foundation-native',4,False)])
def test_complete_serial_evaluation_preserves_scheduler_environment(tmp_path,monkeypatch,stage,expected,fold):
    real=tmp_path/'physical-project';real.mkdir()
    base=tmp_path/'base';base.symlink_to(real,target_is_directory=True);root=base/'evaluation';root.mkdir()
    monkeypatch.setattr(job,'BASE',base);monkeypatch.setattr(job,'ART',base/'artifacts')
    if stage in ('retention-qwen','retention-qwen-v2','retention-qwen-measured'):
        folder=base/'reviewer-models/qwen3-32b-v1';folder.mkdir(parents=True)
        (folder/'download.json').write_text(json.dumps({'download_directory':'/public/reviewer'}))
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
    job.main(argparse.Namespace(source='source',root=str(root),worktree='source-native',stage=stage,epochs=2,merge_lora=fold,semantic_sha='a'*64,semantic_root=None if stage!='retention-train' else '/review',new_cache='/new-cache',new_manifest_sha='b'*64,resume_cache='/prior-cache' if stage=='retention-cache' else None))
    native=[args for args in calls if args[0]=='apptainer']
    assert len(native)==expected
    assert all('--containall' in args and '--cleanenv' in args for args in native)
    invocations=[next(x for x in args if x.startswith('DUPLEX_NATIVE_EXECUTION_ID=')) for args in native]
    assert len(set(invocations))==expected
    last_name={'evaluate':'combine','posttrain-evaluate':'compare','retention-source-audit':'source-audit','release-evaluate':'release-reference','retention-semantic':'retention-semantic','retention-qwen':'retention-qwen','retention-qwen-v2':'retention-qwen','retention-cache':'retention-cache','retention-train':'rehearsal-restore','retention-qwen-measured':'retention-qwen','retention-evaluate':'comparison','foundation-reasoning':'foundation-reasoning','foundation-native':'audit'}[stage]
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
    elif stage=='retention-cache':
        assert native[0][native[0].index('-m')+1]=='training.venus.retention_cache'
        assert native[0][native[0].index('--semantic-sha')+1]=='a'*64
        assert '--checkpoint' not in native[0]
        assert native[0][native[0].index('--resume-cache')+1]=='/prior-cache'
    elif stage in ('retention-qwen','retention-qwen-v2','retention-qwen-measured'):
        assert native[0][native[0].index('-m')+1]=='data.venus.retention_semantics_qwen'
        assert native[0][native[0].index('--model')+1]=='/public/reviewer'
        assert native[0][native[0].index('--calibration-version')+1]==('v2' if stage in ('retention-qwen-v2','retention-qwen-measured') else 'v1')
        assert '--checkpoint' not in native[0] and '--opal-source' not in native[0]
    elif stage=='foundation-native':
        assert [x[x.index('--variant')+1] for x in native]==['venus-released-reference','venus-rp-reference','personaplex-release','audit']
        assert all(x[x.index('-m')+1]=='training.venus.foundation_native' for x in native)
        assert all(any(v.startswith('PYTHONPATH=') and 'personaplex-16419a-v1/moshi' in v for v in x) for x in native)
        assert all('--epochs' not in x and '--checkpoint' not in x for x in native)
    elif stage=='foundation-reasoning':
        assert native[0][native[0].index('-m')+1]=='training.venus.foundation_reasoning'
        assert '--candidate-checkpoint' not in native[0] and '--epochs' not in native[0]
        assert native[0][native[0].index('--numeric-plan')+1].endswith('venus-numeric16-text-20261108.json')
    elif stage=='retention-evaluate':
        assert [args[args.index('-m')+1] for args in native]==['training.venus.retention_evaluate','training.venus.retention_evaluate','training.venus.reasoning128','training.venus.retention_comparison']
        assert [args[args.index('--arm')+1] for args in native[:2]]==['expanded-data','rehearsal']
        assert native[2][native[2].index('--plan')+1].endswith('venus-reasoning128-text-20261107.json')
    elif stage=='retention-train':
        assert all(args[args.index('-m')+1]=='training.venus.retention_train' for args in native)
        assert [args[args.index('--arm')+1] for args in native]==['expanded-data','expanded-data','rehearsal','rehearsal']
        assert [('--final-restore' in args) for args in native]==[False,True,False,True]
        assert all(args[args.index('--new-manifest-sha')+1]=='b'*64 for args in native)
    else:
        assert [args[args.index('--stage')+1] for args in native]==(['schema-selftest','reference','restore','reference','restore','collect','collect','combine'] if stage=='evaluate' else ['collect','compare'])
    terminal=json.loads((root/'execution-123/terminal.json').read_text())
    assert terminal['actual_exit_code']==0 and terminal['new_optimizer_updates']==(None if stage=='retention-train' else 0)

def test_inference_fold_cannot_be_used_in_training():
    with pytest.raises(ValueError,match='separate post-training'):
        job.main(argparse.Namespace(stage='train',merge_lora=True))
