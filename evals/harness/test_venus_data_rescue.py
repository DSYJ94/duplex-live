import hashlib
import json

import pytest
from training.venus import data_rescue as d


def test_absent_exact_approval_prevents_all_provider_calls(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'AUTH',tmp_path/'no-owner-approval')
    monkeypatch.setattr(d.r,'cloud',lambda *a:pytest.fail('Must not reach provider'))
    with pytest.raises(FileNotFoundError):d.execute()


@pytest.mark.parametrize('path',['/absolute','../escape','native/.private/key',
    'native/owner-conversations/audio.wav','native/session-token','native/API keys.md',
    'native/core.123','native/credentials/key','native/key.pem'])
def test_private_owner_and_escape_paths_excluded(path):
    assert not d.allowed(path)


@pytest.fixture
def inventory_inputs(tmp_path,monkeypatch):
    base=tmp_path/'workspace';base.mkdir()
    monkeypatch.setattr(d,'BASE',base);monkeypatch.setattr(d,'authority',lambda:{})
    q=dict(cache_manifest_sha256=d.MANIFEST_SHA,required_cache=[])
    for i in range(268):
        p=base/'venus-training-v1/paced-prefix-cache-v24'/('scene-%03d.pt'%i)
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'exact-native-cache')
        q['required_cache'].append(dict(path=str(p.relative_to(base)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    for name in d.EXTRAS:
        top=base/name;top.mkdir(parents=True,exist_ok=True);(top/'public-native.json').write_text('{}')
    for name in d.SINGLE:
        p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    monkeypatch.setattr(d,'request',lambda *a:q)
    request=tmp_path/'request.json';request.write_text(json.dumps(q))
    return base,request,tmp_path/'inventory.json'


def test_all268_hashes_plus_named_native_files_only(inventory_inputs):
    base,request,output=inventory_inputs
    private=base/d.EXTRAS[0]/'owner-conversations';private.mkdir();(private/'private.wav').write_bytes(b'DO_NOT_COPY')
    (base/d.EXTRAS[0]/'outside-link').symlink_to('/etc/passwd')
    d.inventory(request,output);a=json.loads(output.read_text())
    cached=[x for x in a['files'] if 'paced-prefix-cache-v24/' in x['path']]
    assert len(cached)==268 and len(a['files'])==275
    assert not any('owner-conversations' in x['path'] for x in a['files'])
    assert not a['links']


def test_one_wrong_cache_hash_rejects_the_whole_selection(inventory_inputs):
    base,request,output=inventory_inputs
    (base/'venus-training-v1/paced-prefix-cache-v24/scene-267.pt').write_bytes(b'wrong')
    with pytest.raises(ValueError,match='cached file hash differs'):d.inventory(request,output)
    assert not output.exists()


def test_oversize_selection_cannot_be_accepted(inventory_inputs,monkeypatch):
    _,request,output=inventory_inputs;monkeypatch.setattr(d,'MAX_BYTES',1)
    with pytest.raises(ValueError,match='transfer cap'):d.inventory(request,output)
    assert not output.exists()


def test_exact_pinned_cache_reference_can_resolve_to_its_same_native_bytes(inventory_inputs):
    base,request,output=inventory_inputs
    original=base/'venus-training-v1/paced-prefix-cache-v24/scene-000.pt'
    target=base/'venus-training-v1/earlier-cache/scene-000.pt';target.parent.mkdir();original.rename(target)
    original.symlink_to(target)
    d.inventory(request,output);a=json.loads(output.read_text());d.admit_inventory(a,json.loads(request.read_text()))
    assert a['cache_sources'][0]['stored_path']=='venus-training-v1/earlier-cache/scene-000.pt'
    assert len(a['cache_sources'])==268
    a['cache_sources'].pop()
    with pytest.raises(ValueError,match='All268'):d.admit_inventory(a,json.loads(request.read_text()))


def test_process_probe_needs_no_docker_client_and_detects_actual_work(tmp_path):
    for pid,args in [(101,['python','-m','training.venus.large_train']),
                     (102,['/usr/bin/containerd-shim-runc-v2']),
                     (103,['dockerd'])]:
        p=tmp_path/str(pid);p.mkdir();(p/'cmdline').write_bytes('\0'.join(args).encode())
    assert d.process_probe(tmp_path)==dict(native_model_pids=[101],container_process_pids=[102])


def test_remaining_time_binds_actual_stopped_removed_first_cpu(tmp_path,monkeypatch):
    p=tmp_path/'provider-final.json'
    p.write_text(json.dumps(dict(id='3473476360381416918',name='duplex-venus-data-rescue-20261106-v1',status='TERMINATED',lastStartTimestamp='2026-11-06T16:07:30.775+00:00',lastStopTimestamp='2026-11-06T16:09:06.996+00:00')))
    (tmp_path/'original-binding-restored.json').write_text(json.dumps(dict(original_vm_id=d.r.GPU_ID,disk_id=d.r.DISK_ID,status='TERMINATED',auto_delete=False)))
    (tmp_path/'temporary-cpu-removed.json').write_text(json.dumps(dict(name='duplex-venus-data-rescue-20261106-v1',original_disk_preserved=True)))
    monkeypatch.setattr(d,'PRIOR_RESCUE',tmp_path)
    monkeypatch.setattr(d,'PRIOR_FINAL_SHA',d.r.sha(p))
    usage=d.prior_usage()
    assert usage['prior_actual_seconds']==96.221
    assert usage['prior_seconds_upper']+usage['this_native_max_seconds']+usage['shutdown_reserve_seconds']==2800
    p.write_text('{}')
    with pytest.raises(ValueError,match='terminal receipt'):d.prior_usage()


@pytest.mark.parametrize('unexpected_container',[False,True])
def test_whole_cpu_transfer_without_docker_cli_restores_binding_and_never_accepts_subset(inventory_inputs,monkeypatch,unexpected_container):
    # Execute the entire new orchestration against scripted provider boundaries.
    # This is not a live CPU/cloud or native-model qualification.
    from datetime import datetime,timezone
    from training.venus import controller as c,source_controller,oslogin_access
    base,request,inventory=inventory_inputs
    monkeypatch.setattr(d,'prior_usage',lambda:dict(combined_seconds_upper=2800))
    d.inventory(request,inventory);inv=json.loads(inventory.read_text())
    root=base.parent/'receipts';root.mkdir();monkeypatch.setattr(d,'ROOT',root)
    monkeypatch.setattr(d,'DEST',base.parent/'download')
    monkeypatch.setattr(d,'AUTH',base.parent/'approval');d.AUTH.write_text('{}')
    previous=base.parent/'previous.json';previous.write_text(json.dumps(dict(lastStartTimestamp='before',lastStopTimestamp='stopped')))
    monkeypatch.setattr(d,'PREVIOUS_GPU_RECEIPT',previous)
    now=datetime.fromisoformat('2026-11-06T17:00:00+00:00').timestamp();monkeypatch.setattr(d.time,'time',lambda:now)
    price=dict(hourly_usd=.058,source='Official Cloud Billing Catalog',machine_type='e2-standard-2',region='europe-west4',at_utc=datetime.fromtimestamp(now,timezone.utc).isoformat(),rates=dict(core=dict(usd=.025),ram=dict(usd=.001)))
    (root/'price.json').write_text(json.dumps(price))
    for name in ('R','CPU','RUN_SECONDS','ENTRYPOINT','AUTH','PROPOSAL','require_authorization'):
        monkeypatch.setattr(d.r,name,getattr(d.r,name))
    for name in ('R','NAME','ssh_argv'):monkeypatch.setattr(c,name,getattr(c,name))
    d.configure();created=[];restored=[];copied=[]
    gpu=dict(id=d.r.GPU_ID,name=d.r.GPU,status='TERMINATED',lastStartTimestamp='before',lastStopTimestamp='stopped')
    cpu=dict(id='TEST_CPU',name=d.CPU,status='RUNNING',lastStartTimestamp='2026-11-06T16:59:50+00:00')
    def cloud(*args):
        if args[1:3]==('instances','list'):return [gpu]
        if args[1:3]==('instances','describe'):return gpu if args[3]==d.r.GPU else cpu
        if args[1:3]==('disks','describe'):return dict(id=d.r.DISK_ID)
        if args[1:3]==('instances','create'):created.append(args);return dict(status='DONE',name='TEST_OPERATION')
        pytest.fail('Unexpected provider action '+str(args))
    monkeypatch.setattr(d.r,'cloud',cloud)
    monkeypatch.setattr(d.r,'validate_original',lambda *a,**kw:None)
    monkeypatch.setattr(d.r,'validate_cpu',lambda *a,**kw:None)
    monkeypatch.setattr(d.r,'wait_operation',lambda *a:None)
    monkeypatch.setattr(d.r,'restore_binding',lambda *a:restored.append(True))
    monkeypatch.setattr(oslogin_access,'ensure',lambda *a:None)
    def run(argv,timeout=60):
        if argv[:3]==['git','rev-parse','HEAD']:return 'TEST_SOURCE'
        if argv[:2]==['/bin/ps','-p']:return 'TEST_PROCESS_BIRTH'
        if argv[0]=='/opt/homebrew/bin/rsync':
            copied.append(argv)
            for row in inv['files']:
                p=d.DEST/row['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((base/row['path']).read_bytes())
        return ''
    monkeypatch.setattr(d.r,'run',run);monkeypatch.setattr(c,'sync',lambda *a,**kw:None)
    def ssh(command,*args):
        if 'docker' in command:raise RuntimeError('sudo: docker: command not found')
        if 'is-active' in command:return 'active'
        if '--probe-models' in command:return json.dumps(dict(native_model_pids=[],container_process_pids=[777] if unexpected_container else []))
        if command.startswith('cat '):return json.dumps(inv)
        return ''
    monkeypatch.setattr(c,'ssh',ssh)
    def prepare(*a,**kw):c.ssh_argv=['ssh','-o','StrictHostKeyChecking=yes','TEST_HOST']
    monkeypatch.setattr(source_controller,'prepare_ssh',prepare)
    class Process:
        pid=987
        def __init__(self,argv,**kw):
            if '--guard' in argv:(root/'guard.json').write_text(json.dumps(dict(expires_unix=now+45)))
        def poll(self):return None
        def terminate(self):pass
    monkeypatch.setattr(d.subprocess,'Popen',Process)
    if unexpected_container:
        with pytest.raises(ValueError,match='Unexpected model/container'):d.execute()
        assert not copied and not (root/'copyback-verified.json').exists()
    else:
        d.execute();proof=json.loads((root/'copyback-verified.json').read_text())
        assert proof['all_hashes_verified'] and proof['exact_old268_cache_verified']
        assert len(proof['cache_sources'])==268 and proof['new_optimizer_updates']==0 and copied
    assert len(created)==1 and restored==[True]
