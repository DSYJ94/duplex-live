"""Exercise orchestration helpers without auth, models or provider mutations."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest
from training.venus import british_controller as b


def test_offline_schedule_is_complete_and_has_no_learning_stage():
    jobs=b.stages('actual-source','/pinned source')
    assert [x[0] for x in jobs]==['audition','main','control','combine']
    assert all('train' not in module.rsplit('.',1)[-1] for _,module,_ in jobs)
    for _,_,args in jobs:
        assert shlex.split(shlex.join(args))==args
        assert args[args.index('--source')+1]=='actual-source'
        assert args[args.index('--voice-reference-sha256')+1].startswith('552adfe')
    assert '--arm' in jobs[1][2] and jobs[1][2][-1]=='main'
    assert '--arm' in jobs[2][2] and jobs[2][2][-1]=='control'
    assert '--combine' in jobs[3][2]


def test_missing_approval_blocks_before_all_provider_calls(tmp_path,monkeypatch):
    monkeypatch.setattr(b.g,'AUTH',tmp_path/'missing-approval')
    with pytest.raises(FileNotFoundError):b.execute()


@pytest.mark.parametrize('hours,rate',[(-1,5),(2,5),(0,6)])
def test_stale_future_or_expensive_price_rejected(hours,rate):
    now=1800000000
    quote=dict(at_utc=datetime.fromtimestamp(now-hours*3600,timezone.utc).isoformat(),source='Official Cloud Billing Catalog exact Netherlands SKUs',hourly_USD=rate)
    with pytest.raises(ValueError):b.price_admission(quote,now)


def test_copyback_verifies_files_links_without_copying_checkpoint_contents(tmp_path,monkeypatch):
    local=tmp_path/'backup';records=tmp_path/'receipts';records.mkdir()
    monkeypatch.setattr(b,'BACKUP',local);monkeypatch.setattr(b,'R',records)
    payload=b'{"complete":false}'
    import hashlib
    inv=dict(files=[dict(path='native/report.json',bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())],links=[dict(path='main/step-0060.pt',target=b.OLD+'/main/step-0060.pt')])
    transfers=[]
    def run(argv,timeout=60):
        transfers.append(argv)
        if len(transfers)==1:
            p=local/'native/report.json';p.parent.mkdir(parents=True);p.write_bytes(payload)
        else:
            p=local/'main/step-0060.pt';p.parent.mkdir(parents=True);p.symlink_to(inv['links'][0]['target'])
        return ''
    c=SimpleNamespace(ssh=lambda *a:json.dumps(inv),ssh_argv=['ssh','-o','StrictHostKeyChecking=yes','user@fixture'],run=run)
    b.copyback(c,lambda:120)
    proof=json.loads((records/'partial-copyback-verified.json').read_text())
    assert proof['files']==1 and proof['links']==1 and proof['bytes']==len(payload)
    assert proof['all_hashes_verified'] and proof['all_link_targets_verified'] and not proof['qualified']
    assert all('-ar' in t and '--no-owner' in t and '--no-group' in t for t in transfers)


def test_copyback_failure_never_creates_success_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'BACKUP',tmp_path/'backup');monkeypatch.setattr(b,'R',tmp_path)
    inv=dict(files=[dict(path='report.json',bytes=1,sha256='wrong')],links=[])
    def run(*a):raise RuntimeError('actual rsync failure')
    c=SimpleNamespace(ssh=lambda *a:json.dumps(inv),ssh_argv=['ssh','fixture'],run=run)
    with pytest.raises(RuntimeError,match='rsync'):b.copyback(c,lambda:120)
    assert not (tmp_path/'copyback-verified.json').exists()


@pytest.mark.parametrize('copyback_fails',[False,True])
def test_complete_orchestration_stops_exact_gpu_and_never_hides_copyback_failure(tmp_path,monkeypatch,copyback_fails):
    # Entire provider/process boundary is scripted; no real approval or resource.
    from training.venus import controller as c,source_controller,oslogin_access,boot_hostkeys
    monkeypatch.setattr(boot_hostkeys,'verified',lambda:[dict(key='ssh-ed25519',value='TEST_PUBLIC_ONLY')])
    from portable.venus import guard as provider_guard
    for name in ('R','NAME','events','UNITS','source','remote','guard','ssh_argv'):
        monkeypatch.setattr(c,name,getattr(c,name,None),raising=False)
    root=tmp_path/'run';root.mkdir();monkeypatch.setattr(b,'R',root)
    monkeypatch.setattr(b.g,'PLAN',root/'execution.json')
    monkeypatch.setattr(b.g,'AUTH',tmp_path/'approval.json');b.g.AUTH.write_text('{}')
    monkeypatch.setattr(b.g,'BACKUP',tmp_path/'backup-proof.json');b.g.BACKUP.write_text('{}')
    monkeypatch.setattr(b.g,'authority',lambda:{'status':'TEST_ONLY'})
    monkeypatch.setattr(b.g,'complete_backup',lambda:{})
    monkeypatch.setattr(b.g,'resource',lambda lease:{})
    monkeypatch.setattr(b.recovery,'PRIOR',tmp_path)
    old=dict(id=b.g.VM_ID,name=b.NAME,status='TERMINATED',selfLink='original-link',
             lastStartTimestamp='2026-11-05T17:38:53.272+00:00',lastStopTimestamp='2026-11-06T00:10:49.641+00:00',
             creationTimestamp='original-creation',disks=[dict(source='/disks/'+b.DISK,boot=True,autoDelete=False)])
    (tmp_path/'provider-final.json').write_text(json.dumps(old))
    vm=dict(old);disk=dict(id=b.g.DISK_ID,sizeGb='250',users=['original-link'])
    now=datetime.fromisoformat('2026-11-06T02:00:01+00:00').timestamp()
    monkeypatch.setattr(b.time,'time',lambda:now)
    (root/'price.json').write_text(json.dumps(dict(at_utc=datetime.fromtimestamp(now,timezone.utc).isoformat(),source='Official Cloud Billing Catalog exact Netherlands SKUs',hourly_USD=5)))
    refs=tmp_path/'refs';refs.mkdir();monkeypatch.setattr(b,'REFS',refs)
    for name in ('rp-ben','british-alice'):(refs/(name+'-16k.wav')).write_bytes(name.encode())
    original_sha=b.recovery.sha
    monkeypatch.setattr(b.recovery,'sha',lambda path:__import__('training.venus.british_audition',fromlist=['REFS']).REFS[Path(path).name.replace('-16k.wav','')] if Path(path).parent==refs else original_sha(path))
    original_read=Path.read_text
    def read(path,*a,**kw):
        if str(path)=='runs/local-migration-20261021/.private/runpod-budget-api-key':return 'TEST_ONLY_NOT_A_CREDENTIAL'
        if str(path)=='runs/venus-large-capacity-20261104/british-native-v84/provider-final.json':return json.dumps(old)
        return original_read(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',read)
    monkeypatch.setattr(provider_guard,'bounded_runpod_read',lambda key:[])
    monkeypatch.setattr(oslogin_access,'ensure',lambda *a,**kw:None)
    monkeypatch.setattr(b.recovery,'wait_operation',lambda op,*a:op)
    commands=[];launched=[]
    def cloud(*args):
        if args[1:3]==('instances','list'):return [dict(vm)]
        if args[1:3]==('disks','describe'):return disk
        if args[1:3]==('instances','describe'):return dict(vm)
        if args[1:3]==('instances','start'):
            vm.update(status='RUNNING',lastStartTimestamp='2026-11-06T02:00:00+00:00')
            return dict(status='DONE',name='TEST_OPERATION')
        if args[1:3]==('instances','set-scheduling'):return {}
        pytest.fail('Unexpected provider call '+str(args))
    monkeypatch.setattr(c,'cloud',cloud)
    def run(argv,timeout=60):
        commands.append(argv)
        if argv[:3]==['git','rev-parse','HEAD']:return 'TEST_GIT_SOURCE'
        if argv[:2]==['/bin/ps','-p']:return 'TEST_PROCESS_BIRTH'
        if 'stop' in argv and b.NAME in argv:vm.update(status='TERMINATED',lastStopTimestamp='2026-11-06T02:10:00+00:00')
        return ''
    monkeypatch.setattr(c,'run',run)
    monkeypatch.setattr(c,'sync',lambda *a,**kw:None)
    monkeypatch.setattr(c,'ssh',lambda *a,**kw:'')
    monkeypatch.setattr(c,'props',lambda *a:dict(ActiveState='active'))
    monkeypatch.setattr(c,'start_unit',lambda name,cmd,seconds:(launched.append((name,cmd)),c.UNITS.append(name)))
    def wait_unit(name,deadline):assert (root/'both-guards-ready.json').exists()
    monkeypatch.setattr(c,'wait_unit',wait_unit)
    def ssh_ready(*a,**kw):
        c.ssh_argv=['ssh','-o','StrictHostKeyChecking=yes','TEST_HOST']
        (root/'ssh.json').write_text(json.dumps(c.ssh_argv))
    monkeypatch.setattr(source_controller,'prepare_ssh',ssh_ready)
    class Process:
        pid=987
        def __init__(self,argv,**kw):
            if 'portable.venus.guard' in argv:(root/'guard.json').write_text('{}')
        def poll(self):return None
        def wait(self,timeout):return 0
        def terminate(self):pass
    monkeypatch.setattr(b.subprocess,'Popen',Process)
    def copyback(*a,**kw):
        if copyback_fails:raise RuntimeError('ACTUAL_TEST_COPYBACK_FAILURE')
        (root/'copyback-verified.json').write_text('{"all_hashes_verified":true}')
    monkeypatch.setattr(b,'copyback',copyback)
    if copyback_fails:
        with pytest.raises(RuntimeError,match='ACTUAL_TEST_COPYBACK_FAILURE'):b.execute()
        assert not (root/'copyback-verified.json').exists()
    else:b.execute()
    assert vm['status']=='TERMINATED'
    assert len(launched)==4 and launched[-1][0].endswith('combine')
    assert all('large_train' not in command for _,command in launched)
    assert not any('delete' in argv for argv in commands)
    usage=json.loads((root/'actual-usage.json').read_text())
    assert usage['full_copyback_verified'] is (not copyback_fails)


def test_final_downloaded_reports_require_whole_cohorts_and_matched_inputs(tmp_path):
    (tmp_path/'native').mkdir();(tmp_path/'audition').mkdir()
    base=dict(complete=True,source='actual-source',new_optimizer_updates=0,word_limit=.05,peak_limit=.995,qualified=False,adopted=False)
    paired=dict(base,cases=[{}]*64);audition=dict(base,cases=[{}]*9)
    (tmp_path/'native/paired-native-report.json').write_text(json.dumps(paired))
    (tmp_path/'audition/voice-audition.json').write_text(json.dumps(audition))
    rows=[dict(condition=arm,id=str(i),input_sha256='frozen'+str(i)) for arm in ('main','control') for i in range(32)]
    p=tmp_path/'native/results.json';p.write_text(json.dumps(dict(cases=rows)))
    assert b.validate_outputs(tmp_path,'actual-source')
    rows[-1]['input_sha256']='changed';p.write_text(json.dumps(dict(cases=rows)))
    with pytest.raises(ValueError,match='paired32'):b.validate_outputs(tmp_path,'actual-source')
    with pytest.raises(ValueError,match='source-bound'):b.validate_outputs(tmp_path,'wrong-source')


def test_capacity_retry_requires_terminal_stockout_unchanged_stopped_gpu_and_spacing():
    previous=dict(id=b.g.VM_ID,status='TERMINATED',lastStartTimestamp='oldstart',lastStopTimestamp='oldstop')
    op=dict(status='DONE',endTime='2026-11-06T09:29:31+00:00',error=dict(errors=[dict(code='ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS')]))
    assert b.capacity_retry_after(op,previous,dict(previous),2)==datetime.fromisoformat(op['endTime']).timestamp()+600
    for attempt in (1,4):
        with pytest.raises(ValueError):b.capacity_retry_after(op,previous,previous,attempt)
    changed=dict(previous,lastStartTimestamp='newactualstart')
    with pytest.raises(ValueError):b.capacity_retry_after(op,previous,changed,2)
    for code in ('PERMISSION_DENIED','UNKNOWN'):
        bad=dict(op,error=dict(errors=[dict(code=code)]))
        with pytest.raises(ValueError):b.capacity_retry_after(bad,previous,previous,2)
    with pytest.raises(ValueError):b.capacity_retry_after(dict(op,status='RUNNING'),previous,previous,2)
