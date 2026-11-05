import json,time
import pytest
from training.venus import oslogin_access as access,controller as c


def test_key_must_be_read_back_covering_whole_run(tmp_path,monkeypatch):
    pub=tmp_path/'approved.pub';pub.write_text('ssh-ed25519 PUBLIC fixture\n');monkeypatch.setattr(access,'PUBLIC_KEY',pub)
    stop=time.time()+18*3600;calls=[];keys=[]
    def cloud(*args):
        calls.append(args)
        if args[3]=='list':return keys
        keys[:]=[dict(key='fingerprint-index',value=dict(key=pub.read_text(),expirationTimeUsec=str(int((stop+3600)*1e6)),fingerprint='fixture'))]
        return {}
    monkeypatch.setattr(c,'cloud',cloud)
    result=access.ensure(stop,tmp_path)
    assert result['registration_changed'] and not result['private_key_read']
    assert [x[3] for x in calls]==['list','add','list']
    calls.clear();access.ensure(stop,tmp_path);assert [x[3] for x in calls]==['list']
    calls.clear();access.ensure(stop,tmp_path,force=True);assert [x[3] for x in calls]==['list','update','list']
    keys[0]['value']['expirationTimeUsec']='1'
    def broken(*args):return keys
    monkeypatch.setattr(c,'cloud',broken)
    with pytest.raises(ValueError,match='Read-back'):access.ensure(stop,tmp_path)


def test_status_auth_repair_retries_read_only_once(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'R',tmp_path);monkeypatch.setattr(c,'events',[])
    monkeypatch.setattr(c,'ssh_argv',['ssh','-o','ControlMaster=auto','-o','ControlPath=/old','host'])
    def denied(*args):raise RuntimeError('Permission denied (publickey).')
    monkeypatch.setattr(c,'ssh',denied);calls=[]
    monkeypatch.setattr(c,'ssh_repair',lambda:calls.append('renew-existing-key'))
    def run(args,timeout):
        calls.append(args);return 'MainPID=123\nActiveState=active\n'
    monkeypatch.setattr(c,'run',run)
    assert c.props('same-unit')['MainPID']=='123'
    assert calls[0]=='renew-existing-key' and len(calls)==2
    assert 'ControlMaster=no' in calls[1] and 'ControlPath=none' in calls[1]
    assert calls[1][-1].startswith('sudo systemctl show same-unit')
    monkeypatch.setattr(c,'ssh_repair',None)
    with pytest.raises(RuntimeError):c.props('same-unit')
