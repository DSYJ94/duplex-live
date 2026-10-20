import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from evals import native_precision_chain as chain


class PrecisionChain(unittest.TestCase):
 def receipt(self):
  return dict(source='a'*40,executor='LOCAL',optimizer_updates=0,training_extension_or_adoption_approved=False,jobs=dict(predecessor='7586717'))
 def env(self,folder):
  return patch.dict(os.environ,dict(DL_CHAIN_WATCHER_JOB='20',DL_TASK_SOURCE=str(folder),DL_TASK_COMMIT='a'*40))
 def test_predecessor_requires_every_unchanged_teacher_and_exact_parent(self):
  r=dict(schema='dl-native-bf16-stream-batch49-diagnostic-v1',all49_complete=True,all_parameters_unchanged=True,execution_policy_unchanged=True,optimizer_updates=0,training_approved=False,training_extension_or_adoption_approved=False,original_reference_approvals_unchanged=True,original_reference_threshold=.001,native520_checkpoint_sha256=chain.PARENT_SHA,cases=[dict(teacher=dict(id=str(i))) for i in range(49)])
  chain.validate_predecessor(r)
  for field,value in [('all_parameters_unchanged',False),('training_approved',True),('original_reference_threshold',.1),('optimizer_updates',1),('native520_checkpoint_sha256','bad'),('cases',r['cases'][:-1]),('cases',[r['cases'][0]]*49)]:
   bad=copy.deepcopy(r);bad[field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):chain.validate_predecessor(bad)
 def test_recovery_predecessor_binds_exact_immutable_receipt_and_job(self):
  with tempfile.TemporaryDirectory() as folder:
   base=Path(folder);tree=base/'tree';plan=tree/'evals/fixtures/native-precision16-no-tf32-plan-20261020.json';plan.parent.mkdir(parents=True);plan.write_text('{}')
   predecessor=base/'runs/native-bf16-stream49-recovery1-local-20261020.json';predecessor.parent.mkdir();predecessor.write_text(json.dumps(dict(source='b'*40,optimizer_updates=0,jobs=dict(bf16_stream49='111'))))
   r=self.receipt();r.update(plan_sha256=chain.digest(plan),predecessor_receipt_path=str(predecessor),predecessor_receipt_sha256=chain.digest(predecessor),predecessor_source='b'*40,lm_matmul_tf32=False,scheduler_watcher_job='20');r['jobs']['predecessor']='111'
   def call(args,env=None):return 'a'*40+'\n' if 'rev-parse' in args else ''
   with self.env(tree),patch.object(chain,'BASE',base),patch.object(chain,'call',side_effect=call):
    chain.validate(r)
    cpu=base/'cpu.json';pre=dict(passed=True,source='a'*40,allocation_job='20',tests_run=7,cuda_initialized=False,cpu_backend_contract_only=True)
    cpu.write_text(json.dumps(pre));r.update(cpu_api_preflight_required=True,cpu_api_preflight=dict(path=str(cpu),sha256=chain.digest(cpu)));chain.validate(r)
    pre['cuda_initialized']=True;cpu.write_text(json.dumps(pre));r['cpu_api_preflight']['sha256']=chain.digest(cpu)
    with self.assertRaises(ValueError):chain.validate(r)
    pre['cuda_initialized']=False;cpu.write_text(json.dumps(pre));r['cpu_api_preflight']['sha256']=chain.digest(cpu)
    r['jobs']['predecessor']='222'
    with self.assertRaises(ValueError):chain.validate(r)
    r['jobs']['predecessor']='111';predecessor.write_text(predecessor.read_text()+'\n')
    with self.assertRaises(ValueError):chain.validate(r)
 def test_busy_slot_does_not_record_any_submission_attempt(self):
  with tempfile.TemporaryDirectory() as folder,self.env(folder),patch.object(chain,'validate'),patch.object(chain,'evidence') as evidence,patch.object(chain,'call',return_value='20\n7586717\n'):
   r=self.receipt();p=Path(folder)/'receipt.json'
   with self.assertRaises(chain.SlotBusy):chain.submit(p,r,'predecessor')
   self.assertNotIn('submission_attempts',r);self.assertFalse(p.exists());evidence.assert_not_called()
 def test_uncertain_sbatch_is_persisted_and_never_retried(self):
  with tempfile.TemporaryDirectory() as folder,self.env(folder),patch.object(chain,'validate'),patch.object(chain,'evidence'):
   r=self.receipt();p=Path(folder)/'receipt.json';calls=[]
   def run(args,env=None):
    calls.append(args)
    if args[0]=='squeue':return '20\n'
    raise subprocess.TimeoutExpired('sbatch',90)
   with patch.object(chain,'call',side_effect=run):
    with self.assertRaises(subprocess.TimeoutExpired):chain.submit(p,r,'predecessor')
    self.assertEqual(json.loads(p.read_text())['submission_attempts']['native16']['status'],'submitting')
    with self.assertRaises(ValueError):chain.submit(p,r,'predecessor')
   self.assertEqual(sum(a[0]=='sbatch' for a in calls),1)
 def test_held_job_is_not_released_with_wrong_settings(self):
  with tempfile.TemporaryDirectory() as folder,self.env(folder),patch.object(chain,'validate'),patch.object(chain,'evidence'),patch.object(chain,'call',side_effect=['20\n','30\n','TimeLimit=00:10:00']) as call:
   r=self.receipt();p=Path(folder)/'receipt.json'
   with self.assertRaises(ValueError):chain.submit(p,r,'predecessor')
   self.assertEqual(json.loads(p.read_text())['submission_attempts']['native16']['status'],'held')
   self.assertFalse(any(a.args[0][:2]==['scontrol','release'] for a in call.call_args_list))
 def test_live_watcher_waits_busy_slot_then_submits_each_stage_once(self):
  with tempfile.TemporaryDirectory() as folder,self.env(folder),patch.object(chain,'validate'),patch.object(chain,'evidence'),patch.object(chain.time,'sleep'):
   p=Path(folder)/'receipt.json';p.write_text(json.dumps(self.receipt()));submitted=[];queues=[];observations=[]
   def run(args,env=None):
    if args[0]=='sacct':
     observations.append(True)
     return args[args.index('-j')+1]+('|TRANSIENT_UNKNOWN|0:0|\n' if len(observations)==1 else '|COMPLETED|0:0|\n')
    if args[0]=='squeue':
     queues.append(True);return '20\n7586717\n' if len(queues)==1 else '20\n'
    if args[0]=='sbatch':submitted.append(args);return '30\n' if len(submitted)==1 else '40\n'
    if args[:3]==['scontrol','show','job']:
     return 'TimeLimit=00:30:00 Reservation=interactive TresPerNode=gres/gpu:2' if args[-1]=='30' else 'TimeLimit=00:30:00 Partition=workq'
    if args[:2]==['scontrol','release']:return ''
    raise AssertionError(args)
   with patch.object(chain,'call',side_effect=run):chain.watch(p)
   r=json.loads(p.read_text());self.assertEqual(len(submitted),2);self.assertEqual(r['released'],['30','40']);self.assertEqual(r['watcher_terminal']['stage'],'speech16');self.assertEqual(r['watcher_terminal']['state'],'COMPLETED')
