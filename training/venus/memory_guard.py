"""Same durable disk, sequential short STANDARD job, fully carried ledger."""
import hashlib,json
from pathlib import Path
from datetime import datetime
AUTH=Path('docs/NATIVE-VENUS-MEMORY-REPAIR-CONTINUATION-2026-10-26.json')

def resource(lease):
    raw=AUTH.read_bytes();a=json.loads(raw);r=a['resources']
    assert a['status']=='APPROVED_WITHIN_EXISTING_TRAINING_SCOPE' and a['maximum_additional_gcp_usage_usd']==35
    assert a['prior_new_scope_gpu_seconds_upper']==9500 and a['maximum_next_gpu_seconds']==4800 and a['conservative_total_scope_usd']<35
    assert lease['authorization_sha256']==hashlib.sha256(raw).hexdigest()
    assert lease['zone']==r['zone'] and str(lease['disk_id'])==r['same_disk_id']
    assert str(lease['vm_id']).isdigit() and str(lease['vm_id'])!=r['retire_only_vm_id']
    return dict(name=r['new_name'],zone=r['zone'],ident=str(lease['vm_id']),disk=r['same_disk_id'],disk_name=r['same_disk_name'],machine=r['machine'],authority=AUTH)

def stamp(s):return datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()

def validate(vm,instances,pods,lease,now):
    r=resource(lease);start=stamp(vm['lastStartTimestamp']);end=start+4800
    assert str(vm['id'])==r['ident'] and vm['name']==r['name'] and vm['status']=='RUNNING'
    assert vm['machineType'].endswith('/'+r['machine']) and vm['zone'].endswith('/'+r['zone']) and vm['creationTimestamp']==lease['creation_timestamp']
    assert vm['tags']['items']==['duplex-venus-train-bounded']
    assert any(d.get('boot') and not d.get('autoDelete') and d['source'].endswith('/disks/'+r['disk_name']) for d in vm['disks'])
    s=vm['scheduling'];assert s.get('provisioningModel','STANDARD')=='STANDARD' and int(s['maxRunDuration']['seconds'])==4800
    assert s['instanceTerminationAction']=='STOP' and not s['automaticRestart'] and start<=now<end and lease['stop_unix']==end
    assert lease['prior_new_gpu_seconds_upper']==9500 and lease['prior_gpu_seconds']==40351 and lease['prior_scope_usd']==91.98448055555555 and lease['disk_size_gib']==100
    assert 0<lease['verified_hourly_price_usd']<=4 and 9500+4800<=14400 and 40351+9500+4800<=57600
    assert (6400*7.25+3100*4+4800*4)/3600+6+4<35
    running=[v for v in instances if v['status'] not in ('TERMINATED','STOPPED','STOPPING') and (v.get('guestAccelerators') or any('/'+m in v.get('machineType','') for m in ['a2-','a3-','a4-','g2-']))]
    assert len(running)==1 and str(running[0]['id'])==r['ident'] and all(p['status'] in ('EXITED','STOPPED','TERMINATED') for p in pods)
    return end
