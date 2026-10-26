"""Apply only the exact IAP proposal after its direct-human approval is recorded."""
import json,subprocess
from pathlib import Path
from .access import approved_iap_proxy,NAME,ZONE,VM_ID,DISK_ID
PROJECT='aurium-duplex-live-20261024'
TAG='duplex-venus-iap-bounded'

def command(*args):
 p=subprocess.run(['gcloud',*args,'--project='+PROJECT,'--quiet','--format=json'],capture_output=True,text=True,timeout=120)
 if p.returncode:raise RuntimeError(p.stderr[-1800:])
 return json.loads(p.stdout) if p.stdout.strip() else None

def main():
 approved_iap_proxy()  # Before ANY provider read or mutation; file does not exist until human approval.
 r=Path('runs/venus-20261026/us-policy-iap-v3');r.mkdir(exist_ok=True)
 vm=command('compute','instances','describe',NAME,'--zone='+ZONE)
 disk=command('compute','disks','describe',NAME,'--zone='+ZONE)
 assert str(vm['id'])==VM_ID and vm['status']=='TERMINATED'
 assert str(disk['id'])==DISK_ID and int(disk['sizeGb'])==100
 assert all(v['status']=='TERMINATED' for v in command('compute','instances','list'))
 names=['duplex-venus-iap-ssh-v1','duplex-venus-iap-deny-other-v1']
 existing=command('compute','firewall-rules','list','--filter=name:duplex-venus-iap-')
 assert not existing,'Do not overwrite un-attributed existing firewall rules'
 (r/'before-access.json').write_text(json.dumps(dict(vm=vm,disk=disk),indent=2)+'\n')
 command('services','enable','iap.googleapis.com')
 command('compute','firewall-rules','create',names[0],'--network=default','--direction=INGRESS','--priority=800','--source-ranges=35.235.240.0/20','--target-tags='+TAG,'--allow=tcp:22')
 command('compute','firewall-rules','create',names[1],'--network=default','--direction=INGRESS','--priority=900','--source-ranges=0.0.0.0/0','--target-tags='+TAG,'--deny=all')
 command('compute','instances','add-tags',NAME,'--zone='+ZONE,'--tags='+TAG)
 old=vm.get('tags',{}).get('items',[])
 if old:command('compute','instances','remove-tags',NAME,'--zone='+ZONE,'--tags='+','.join(old))
 after=command('compute','instances','describe',NAME,'--zone='+ZONE)
 assert set(after['tags']['items'])=={TAG} and after['status']=='TERMINATED' and str(after['id'])==VM_ID
 rules=[command('compute','firewall-rules','describe',n) for n in names]
 (r/'access-setup-complete.json').write_text(json.dumps(dict(vm_id=VM_ID,disk_id=DISK_ID,vm_still_stopped=True,rules=rules,new_iam_grants=False),indent=2)+'\n')
 print('Exact approved IAP setup complete; GPU remains stopped.')
if __name__=='__main__':main()
