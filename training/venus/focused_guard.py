"""New human-approved repair scope; all previous compute stays in the ledger."""
import hashlib,json
from pathlib import Path
from datetime import datetime

AUTH=Path('docs/NATIVE-VENUS-FOCUSED-REPAIR-AUTHORIZATION-2026-10-27.json')
NAME='duplex-venus-behavior-20261027-v1'
IDENT='6535540622249260120'

def resource(lease):
    raw=AUTH.read_bytes();a=json.loads(raw);r=a['resources']
    assert a['status']=='APPROVED' and a['human_approval']=='Approve $40 / four GPU hours'
    assert a['maximum_additional_gcp_usage_usd']==40 and a['maximum_additional_gpu_seconds']==14400
    assert lease['authorization_sha256']==hashlib.sha256(raw).hexdigest()
    assert lease['zone']==r['zone'] and str(lease['disk_id'])==r['same_existing_disk_id']
    assert str(lease['vm_id'])==IDENT
    return dict(name=NAME,zone=r['zone'],ident=IDENT,disk=r['same_existing_disk_id'],disk_name=r['same_existing_disk_name'],machine='a2-highgpu-1g',authority=AUTH)

def validate(vm,instances,pods,lease,now):
    r=resource(lease);start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();end=start+1300
    assert str(vm['id'])==r['ident'] and vm['name']==r['name'] and vm['status']=='RUNNING'
    assert vm['machineType'].endswith('/'+r['machine']) and vm['zone'].endswith('/'+r['zone']) and vm['creationTimestamp']==lease['creation_timestamp']
    assert vm['tags']['items']==['duplex-venus-train-bounded']
    assert any(d.get('boot') and not d.get('autoDelete') and d['source'].endswith('/disks/'+r['disk_name']) for d in vm['disks'])
    s=vm['scheduling'];assert s['provisioningModel']=='FLEX_START' and int(s['maxRunDuration']['seconds'])==1300
    assert s['instanceTerminationAction']=='STOP' and not s['automaticRestart'] and start<=now<end and lease['stop_unix']==end
    assert lease['prior_gpu_seconds']==54451 and lease['prior_new_gpu_seconds_upper']==0 and lease['disk_size_gib']==100
    assert lease['prior_scope_usd']==91.98448055555555 and 0<lease['verified_hourly_price_usd']<=4
    assert 1300<=14400 and 54451+1300<=72000 and 1300*4/3600+10+6+8<=40
    running=[v for v in instances if v['status'] not in ('TERMINATED','STOPPED','STOPPING') and (v.get('guestAccelerators') or any('/'+m in v.get('machineType','') for m in ['a2-','a3-','a4-','g2-']))]
    assert len(running)==1 and str(running[0]['id'])==r['ident'] and all(p['status'] in ('EXITED','STOPPED','TERMINATED') for p in pods)
    return end
