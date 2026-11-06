"""Recovery safety checks; no provider calls, model imports or resource creation."""
from copy import deepcopy
import json

import pytest

from training.venus import results_recovery as r


def original():
    return dict(id=r.GPU_ID, status='TERMINATED', selfLink='original-link',
                disks=[dict(source='/disks/'+r.DISK, boot=True, autoDelete=False)])


def disk():
    return dict(id=r.DISK_ID, sizeGb='250', users=['original-link'])


def cpu():
    return dict(id='new-cpu-id', name=r.CPU, status='TERMINATED',
                machineType='/machineTypes/e2-standard-2',
                disks=[dict(source='/disks/'+r.DISK, boot=True, autoDelete=False)],
                tags=dict(items=['duplex-venus-train-bounded']),
                scheduling=dict(maxRunDuration=dict(seconds='900'), instanceTerminationAction='STOP'))


@pytest.mark.parametrize('path', ['/absolute', '../escape', 'a/../../escape', 'a\nsecret',
                                  '.private/key', 'owner-conversations/recording.wav',
                                  'a/session-token', 'API keys.md', 'a/core.999'])
def test_unapproved_paths_are_rejected(tmp_path, path):
    with pytest.raises(ValueError): r.checked_path(tmp_path, path)


def test_symlink_parent_cannot_redirect_copyback(tmp_path):
    root=tmp_path/'root'; root.mkdir()
    other=tmp_path/'other'; other.mkdir()
    (root/'a').symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes'): r.checked_path(root, 'a/file')


def test_missing_auth_fails_before_any_provider_call(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'AUTH', tmp_path/'missing-approval.json')
    monkeypatch.setattr(r, 'cloud', lambda *a: pytest.fail('Provider call before authorization'))
    with pytest.raises(FileNotFoundError): r.execute()


@pytest.mark.parametrize('fault', ['proposal_hash', 'new_GPU_intervals', 'maximum_CPU_running_seconds', 'expired'])
def test_approval_cannot_expand_or_outlive_exact_scope(tmp_path, monkeypatch, fault):
    p=json.loads(r.PROPOSAL.read_text())
    auth={k:p[k] for k in ('phase_ceiling_USD','project_ceiling_USD','recovery_maximum_USD',
        'maximum_CPU_running_intervals','maximum_CPU_running_seconds','original_GPU_id',
        'same_disk_id','new_GPU_intervals','new_optimizer_updates')}
    auth.update(status='APPROVED',human_reply_verbatim='test fixture only',
                proposal_path=str(r.PROPOSAL),proposal_sha256=r.sha(r.PROPOSAL))
    if fault=='proposal_hash':auth['proposal_sha256']='changed'
    if fault=='new_GPU_intervals':auth['new_GPU_intervals']=1
    if fault=='maximum_CPU_running_seconds':auth['maximum_CPU_running_seconds']=901
    if fault=='expired':monkeypatch.setattr(r.time,'time',lambda:r.DEADLINE)
    path=tmp_path/'approval.json';path.write_text(json.dumps(auth))
    monkeypatch.setattr(r,'AUTH',path)
    monkeypatch.setattr(r,'cloud',lambda *a:pytest.fail('Cloud call before exact scope admission'))
    with pytest.raises(ValueError):r.execute()


def test_lost_create_response_stops_only_nonce_bound_exact_cpu(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'R',tmp_path)
    (tmp_path/'claim.json').write_text('{"single_attempt":true}')
    vm=cpu();vm['status']='RUNNING'
    vm['metadata']=dict(items=[dict(key='duplex-results-recovery-claim',value=r.sha(tmp_path/'claim.json'))])
    commands=[]
    monkeypatch.setattr(r,'cloud',lambda *a:deepcopy(vm))
    def run(argv,timeout=60):commands.append(argv);vm['status']='TERMINATED';return ''
    monkeypatch.setattr(r,'run',run)
    assert r.stop_cpu()['status']=='TERMINATED'
    assert len(commands)==1 and commands[0][4]==r.CPU
    vm['metadata']['items'][0]['value']='unrelated-request'
    with pytest.raises(ValueError,match='unbound'):r.stop_cpu()
    assert len(commands)==1


def test_batches_keep_json_priority_and_no_file_larger_than_limit():
    rows=[dict(path='report.json',bytes=5),dict(path='a.wav',bytes=32*1024**2),dict(path='b.wav',bytes=1)]
    actual=list(r.batches(rows))
    assert [x['path'] for b in actual for x in b]==[x['path'] for x in rows]
    assert all(sum(x['bytes'] for x in b)<=32*1024**2 for b in actual)
    with pytest.raises(ValueError): list(r.batches([dict(bytes=32*1024**2+1)]))


@pytest.mark.parametrize('field,value', [('id','wrong'),('status','RUNNING')])
def test_original_identity_and_stopped_state_required(field,value):
    vm=original();vm[field]=value
    with pytest.raises(ValueError): r.validate_original(vm,disk(),attached=True)


def test_original_disk_requires_non_auto_delete_and_exact_users():
    vm=original();vm['disks'][0]['autoDelete']=True
    with pytest.raises(ValueError): r.validate_original(vm,disk(),attached=True)
    d=disk();d['users']=['another-machine']
    with pytest.raises(ValueError): r.validate_original(original(),d,attached=True)
    vm=original();vm['disks']=[];d=disk();d['users']=[]
    r.validate_original(vm,d,attached=False)


@pytest.mark.parametrize('change', ['GPU','extra_disk','auto_delete','deadline','service_account','wrong_tag'])
def test_cpu_admission_rejects_scope_expansion(change):
    vm=cpu()
    if change=='GPU':vm['guestAccelerators']=[dict(acceleratorCount=1)]
    if change=='extra_disk':vm['disks'].append(dict(source='new-disk'))
    if change=='auto_delete':vm['disks'][0]['autoDelete']=True
    if change=='deadline':vm['scheduling']['maxRunDuration']['seconds']='901'
    if change=='service_account':vm['serviceAccounts']=[dict(email='new-account')]
    if change=='wrong_tag':vm['tags']['items']=['http-server']
    with pytest.raises(ValueError):r.validate_cpu(vm)


def test_cleanup_restores_original_before_deleting_only_empty_cpu(tmp_path,monkeypatch):
    vm=original();vm['disks']=[]
    pd=disk();pd['users']=['cpu-link']
    temp=cpu();calls=[]
    monkeypatch.setattr(r,'R',tmp_path)
    def cloud(*args):
        if args[1:3]==('instances','list'):return [deepcopy(temp)]
        if args[1:3]==('instances','describe'):
            return deepcopy(vm if args[3]==r.GPU else temp)
        if args[1:3]==('disks','describe'):return deepcopy(pd)
        pytest.fail(str(args))
    monkeypatch.setattr(r,'cloud',cloud)
    monkeypatch.setattr(r,'stop_cpu',lambda **kw:deepcopy(temp))
    def run(argv,timeout=60):
        calls.append(argv)
        if 'detach-disk' in argv:temp['disks']=[];pd['users']=[]
        if 'attach-disk' in argv:vm['disks']=original()['disks'];pd['users']=['original-link']
        if 'delete' in argv:
            assert r.GPU not in argv and r.DISK not in argv
            assert not temp['disks'] and pd['users']==['original-link']
            assert (tmp_path/'original-binding-restored.json').exists()
        return ''
    monkeypatch.setattr(r,'run',run)
    r.restore_binding()
    assert [a[3] for a in calls if a[2]=='instances']==['detach-disk','attach-disk','set-disk-auto-delete','delete']
    assert (tmp_path/'temporary-cpu-removed.json').exists()


def test_cleanup_never_deletes_when_original_binding_cannot_be_restored(tmp_path,monkeypatch):
    vm=original();vm['disks']=[]
    pd=disk();pd['users']=['unknown-writer']
    monkeypatch.setattr(r,'R',tmp_path)
    def cloud(*args):
        if args[1:3]==('instances','list'):return []
        if args[1:3]==('instances','describe'):return vm
        return pd
    monkeypatch.setattr(r,'cloud',cloud)
    monkeypatch.setattr(r,'run',lambda *a:pytest.fail('Must not mutate unknown disk binding'))
    with pytest.raises(ValueError):r.restore_binding()
    assert not (tmp_path/'original-binding-restored.json').exists()
