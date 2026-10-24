import copy
import hashlib
import json
from pathlib import Path
import unittest

from portable.context.gcp_budget_watchdog import (
    AUTH_SHA, CATALOG_SHA, DISK, PROJECT, ZONE, check_live, owned, validate_lease)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.authority = (Path(__file__).resolve().parents[2]/
            'docs/GCP-NATIVE-CONTINUATION-AUTHORIZATION-2026-10-24.json').read_bytes()
        self.destination = json.dumps(dict(schema='dl-native-gcp-destination-hashes-v1',
            inventory_sha256=CATALOG_SHA, passed=True, all_catalog_entries_verified=True,
            verified_regular_files=40132, verified_bytes=81154452891,
            verified_archival_links=8, failures=[])).encode()
        self.lease = dict(schema='dl-native-gcp-bounded-gpu-lease-v1',project=PROJECT,zone=ZONE,
            name='duplex-native-saved8-gpu-fixture', id='123',disk_name=DISK,disk_id='456',
            authorization_sha256=AUTH_SHA,optimizer_updates=0,hourly_upper_usd=5.63,
            created_utc='2026-10-25T00:00:00Z',stop_utc='2026-10-25T12:00:00Z',
            destination_report_sha256=hashlib.sha256(self.destination).hexdigest())
        base='https://www.googleapis.com/compute/v1/projects/'+PROJECT+'/zones/'+ZONE
        self.vm=dict(id='123',name=self.lease['name'],creationTimestamp=self.lease['created_utc'],
            zone=base,status='RUNNING',machineType=base+'/machineTypes/a2-ultragpu-1g',
            scheduling=dict(instanceTerminationAction='STOP',automaticRestart=False,
                            terminationTime=self.lease['stop_utc']),
            disks=[dict(type='PERSISTENT',source=base+'/disks/'+DISK,autoDelete=False,
                        boot=True,mode='READ_WRITE')],
            guestAccelerators=[dict(acceleratorCount=1,acceleratorType=base+'/acceleratorTypes/nvidia-a100-80gb')])
        self.disk=dict(id='456',name=DISK,sizeGb='250')
        self.now=1792886700.0  # 2026-10-25T00:05:00Z

    def check(self, vm=None, disk=None, instances=None, pods=None):
        return check_live(self.lease, vm or self.vm, disk or self.disk,
                          [self.vm] if instances is None else instances,
                          [] if pods is None else pods, self.now)

    def test_real_owner_scope_and_complete_destination_required(self):
        validate_lease(self.lease,self.authority,self.destination)
        with self.assertRaises(ValueError):validate_lease(self.lease,b'{}',self.destination)
        changed=json.loads(self.destination);changed['verified_regular_files']=40131
        payload=json.dumps(changed).encode()
        lease=dict(self.lease,destination_report_sha256=hashlib.sha256(payload).hexdigest())
        with self.assertRaisesRegex(ValueError,'Full real'):
            validate_lease(lease,self.authority,payload)

    def test_full_lifetime_budget_and_twelve_hour_limit_are_hard(self):
        for changes in (dict(hourly_upper_usd=8),dict(hourly_upper_usd=True),
                        dict(stop_utc='2026-10-25T12:00:01Z'),dict(optimizer_updates=8),
                        dict(project='other'),dict(disk_name='other')):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                validate_lease(dict(self.lease,**changes),self.authority,self.destination)

    def test_healthy_live_guard_counts_elapsed_cost_and_remaining_window(self):
        guard=self.check()
        self.assertTrue(guard['single_gpu_across_providers_verified'])
        self.assertEqual(guard['provider'],'gcp')
        self.assertEqual(guard['resource_id'],'123')
        self.assertLess(guard['planned_maximum_cost_usd'],guard['remaining_budget_usd'])

    def test_changed_deadline_restart_policy_or_persistent_disk_is_rejected(self):
        for change in ('deadline','restart','disk','delete','size'):
            vm=copy.deepcopy(self.vm);disk=dict(self.disk)
            if change=='deadline':vm['scheduling']['terminationTime']='2026-10-26T00:00:00Z'
            if change=='restart':vm['scheduling']['automaticRestart']=True
            if change=='disk':vm['disks'][0]['source']='unexpected'
            if change=='delete':vm['disks'][0]['autoDelete']=True
            if change=='size':disk['sizeGb']='251'
            with self.subTest(change=change),self.assertRaises(ValueError):self.check(vm,disk)

    def test_replaced_instance_never_matches_the_owned_stop_target(self):
        replacement=dict(self.vm,id='999')
        self.assertFalse(owned(replacement,self.lease))
        with self.assertRaisesRegex(ValueError,'identity changed'):self.check(replacement)

    def test_other_gpu_or_live_staging_cpu_prevents_execution(self):
        for other in (dict(self.vm,id='999',name='other'),
                      dict(self.vm,id='777',name=DISK,guestAccelerators=[])):
            with self.assertRaises(ValueError):self.check(instances=[self.vm,other])
        with self.assertRaises(ValueError):
            self.check(pods=[dict(name='duplex-native-other',status='RUNNING',computeType='GPU')])
        self.check(pods=[dict(name='duplex-native-old',status='EXITED')])


if __name__ == '__main__':
    unittest.main()
