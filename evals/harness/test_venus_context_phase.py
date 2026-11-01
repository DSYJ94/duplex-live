import copy
import json
from datetime import datetime
from pathlib import Path
import unittest

from training.venus.context_objective import authority,ADMISSION,sha
from training.venus.context_guard import validate,PLAN,KEY


class ContextPhaseTests(unittest.TestCase):
    def fixture(self):
        vm=json.loads(Path('runs/venus-training-20261101/source-audit-v47/provider-start.json').read_text())
        vm['scheduling']['maxRunDuration']['seconds']='4000'
        lease=json.loads(Path('runs/venus-training-20261101/source-audit-v47/lease.json').read_text())
        now=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()
        vm['lastStartTimestamp']='2026-11-02T00:00:00Z'
        now=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()
        lease.update(resource_key=KEY,authorization_sha256=sha(ADMISSION),execution_plan_sha256=sha(PLAN),prior_new_gpu_seconds_upper=28700,stop_unix=now+4000)
        return vm,lease,now

    def test_whole_epoch_and_cumulative_cost_admitted(self):
        a=authority()
        self.assertEqual((a['maximum_new_updates'],a['maximum_complete_epochs']),(60,1))
        vm,lease,now=self.fixture()
        self.assertEqual(validate(vm,[vm],[],lease,now),now+4000)

    def test_wrong_disk_deadline_overlap_or_budget_reset_rejected(self):
        for kind in ('disk','deadline','overlap','reset'):
            vm,lease,now=self.fixture();instances=[vm]
            if kind=='disk':lease['disk_id']='123'
            elif kind=='deadline':lease['stop_unix']+=1
            elif kind=='overlap':instances.append(dict(vm,id='123'))
            else:lease['prior_new_gpu_seconds_upper']=0
            with self.subTest(kind=kind),self.assertRaises(AssertionError):validate(vm,instances,[],lease,now)

    def test_no_running_runpod_or_different_provider_stop_policy(self):
        vm,lease,now=self.fixture()
        with self.assertRaises(AssertionError):validate(vm,[vm],[dict(status='RUNNING')],lease,now)
        vm['scheduling']['instanceTerminationAction']='DELETE'
        with self.assertRaises(AssertionError):validate(vm,[vm],[],lease,now)
