import copy,json,tempfile,unittest
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch
from training.venus import source_guard as s
from portable.venus import guard

class SourceGuardTests(unittest.TestCase):
 def setUp(self):
  p=json.loads(s.PLAN.read_text());start=datetime.fromisoformat(p['previous_provider_stop']).timestamp()+100
  self.lease=dict(resource_key=s.KEY,zone='us-central1-a',vm_id='123',disk_id='4375908109499109167',disk_size_gib=100,creation_timestamp='fixture',stop_unix=start+1200,prior_new_gpu_seconds_upper=28000,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,verified_hourly_price_usd=3.7,authorization_sha256=s.sha(s.AUTH),execution_plan_sha256=s.sha(s.PLAN),controller_pid=1,controller_birth='fixture')
  self.vm=dict(name=p['vm_name'],id='123',status='RUNNING',zone='/us-central1-a',machineType='/a2-highgpu-1g',creationTimestamp='fixture',lastStartTimestamp=datetime.fromtimestamp(start).isoformat(),tags=dict(items=['duplex-venus-train-bounded']),disks=[dict(boot=True,autoDelete=False,diskSizeGb='100',source='/disks/duplex-venus-train-20261026-v1')],scheduling=dict(provisioningModel='FLEX_START',maxRunDuration=dict(seconds=1200),instanceTerminationAction='STOP',automaticRestart=False))
  self.now=start+1
 def test_exact_resource_registration_and_budget(self):
  self.assertEqual(guard.resource(self.lease)['ident'],'123')
  self.assertEqual(guard.validate(self.vm,[self.vm],[],self.lease,self.now),self.lease['stop_unix'])
 def test_reject_storage_deadline_budget_reset_and_overlap(self):
  for name,value in [('disk_size_gib',200),('prior_new_gpu_seconds_upper',0),('stop_unix',self.lease['stop_unix']+1)]:
   l=dict(self.lease);l[name]=value
   with self.assertRaises(AssertionError):s.validate(self.vm,[self.vm],[],l,self.now)
  for bad in ('disk','duration','overlap'):
   v=copy.deepcopy(self.vm);instances=[v]
   if bad=='disk':v['disks'][0]['diskSizeGb']='200'
   if bad=='duration':v['scheduling']['maxRunDuration']['seconds']=2400
   if bad=='overlap':instances.append(dict(v,id='124'))
   with self.assertRaises(AssertionError):s.validate(v,instances,[],self.lease,self.now)
 def test_actual_mac_entrypoint_accepts_new_authority_fields(self):
  class ReachedLiveRead(Exception):pass
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'lease').write_text(json.dumps(self.lease));(root/'ssh').write_text(json.dumps(['ssh','-o','StrictHostKeyChecking=yes','fixture']));(root/'key').write_text('fixture-no-provider-call')
   args=SimpleNamespace(lease=str(root/'lease'),ssh_argv=str(root/'ssh'),runpod_key=str(root/'key'),status=str(root/'status'),stop_file=None)
   with patch.object(guard,'run',side_effect=ReachedLiveRead),patch.object(guard,'cloud',return_value=dict(id='123',status='TERMINATED')):
    with self.assertRaises(ReachedLiveRead):guard.mac(args)
