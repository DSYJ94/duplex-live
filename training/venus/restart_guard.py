"""Exact existing resource/native stop; all prior usage carried."""
import hashlib
import json
from pathlib import Path
from datetime import datetime

AUTH=Path('docs/NATIVE-VENUS-NATIVE-GRADIENT-CONTINUATION-2026-10-26.json')


def resource(lease):
    raw=AUTH.read_bytes();a=json.loads(raw);r=a['existing_resource']
    assert a['status']=='APPROVED' and a['maximum_additional_gcp_usage_usd']==35 and a['maximum_additional_gpu_seconds']==14400
    assert a['prior_new_scope_gpu_seconds_upper']==6400 and a['maximum_single_restart_seconds']==7200
    assert a['total_conservative_scope_usd']<35 and not r['new_vm_or_disk']
    assert lease['authorization_sha256']==hashlib.sha256(raw).hexdigest()
    assert lease['zone']==r['zone'] and str(lease['vm_id'])==r['vm_id'] and str(lease['disk_id'])==r['disk_id']
    return dict(name=r['name'],disk_name=r['disk_name'],zone=r['zone'],ident=r['vm_id'],disk=r['disk_id'],machine=r['machine'],authority=AUTH)


def stamp(s):return datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()


def validate(vm,instances,pods,lease,now):
    r=resource(lease);a=json.loads(AUTH.read_text());start=stamp(vm['lastStartTimestamp']);end=start+7200
    assert str(vm['id'])==r['ident'] and vm['name']==r['name'] and vm['status']=='RUNNING'
    assert vm['machineType'].endswith('/'+r['machine']) and vm['zone'].endswith('/'+r['zone'])
    assert vm['creationTimestamp']==lease['creation_timestamp'] and vm['tags']['items']==['duplex-venus-train-bounded']
    assert any(d.get('boot') and not d.get('autoDelete') and d['source'].endswith('/disks/'+r['disk_name']) for d in vm['disks'])
    s=vm['scheduling'];assert s['provisioningModel']=='FLEX_START' and int(s['maxRunDuration']['seconds'])==7200
    assert s['instanceTerminationAction']=='STOP' and not s['automaticRestart'] and start<=now<end and lease['stop_unix']==end
    assert lease['prior_new_gpu_seconds_upper']==6400 and lease['prior_gpu_seconds']==40351 and lease['prior_scope_usd']==a['prior_original_scope_cost_upper_usd'] and lease['disk_size_gib']==100
    assert lease['hourly_admission_ceiling_usd']==a['hourly_admission_ceiling_usd']==4
    assert 0<lease['verified_hourly_price_usd']<=4
    assert 6400+7200<=14400 and 40351+6400+7200<=57600 and (6400*7.25+7200*4)/3600+6+4<35
    running=[v for v in instances if v['status'] not in ('TERMINATED','STOPPED','STOPPING') and (v.get('guestAccelerators') or any('/'+m in v.get('machineType','') for m in ['a2-','a3-','a4-','g2-']))]
    assert len(running)==1 and str(running[0]['id'])==r['ident']
    assert all(p['status'] in ('EXITED','TERMINATED','STOPPED') for p in pods)
    return end
