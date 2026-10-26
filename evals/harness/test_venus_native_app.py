import asyncio
import unittest
from portable.venus.native_app import ArithmeticWorker, DelegateParser
from portable.venus.guard import validate, IDENT, PRIOR_SECONDS, PRIOR_USD


class NativeProtocolTests(unittest.IsolatedAsyncioTestCase):
    def test_split_private_request(self):
        p=DelegateParser()
        self.assertEqual(p.feed('<unit><dele'),[])
        self.assertEqual(p.feed('gate>sum318,247,569</dele'),[])
        self.assertEqual(p.feed('gate></unit>'),['sum318,247,569'])
        self.assertEqual(p.feed('<delegate>new</delegate><delegate>second</delegate>'),['new','second'])

    async def test_computed_changed_result_and_single_injection(self):
        w=ArithmeticWorker(0)
        w.submit('sum318,247,569');await asyncio.gather(*w.tasks.values())
        self.assertIn('1134',w.take_results());self.assertIsNone(w.take_results())
        w.submit('sum318,247,570');await asyncio.gather(*w.tasks.values())
        r=w.take_results();self.assertIn('1135',r);self.assertNotIn('1134',r)
        await w.close()

    async def test_cancel_actual_pending_task_suppresses_result(self):
        w=ArithmeticWorker(.05);w.submit('sum318,247,569');await asyncio.sleep(0)
        w.submit('cancel that task');await asyncio.gather(*w.tasks.values(),return_exceptions=True)
        receipt=w.take_results();self.assertIn('cancelled',receipt);self.assertNotIn('Computed sum',receipt)
        self.assertTrue(any(e['type']=='worker_cancelled' for e in w.events))
        kinds=[e['type'] for e in w.events]
        self.assertLess(kinds.index('worker_cancelled'),kinds.index('cancellation_confirmed'))
        await w.close()

    async def test_no_execution_from_arbitrary_utterances(self):
        w=ArithmeticWorker(0);p=DelegateParser()
        self.assertEqual(p.feed('Please cancel or add318 and247.'),[])
        self.assertFalse(w.tasks);await w.close()

    def test_exact_cumulative_guard_and_no_overlap(self):
        vm=dict(id=IDENT,name='duplex-native-saved8-gpu-flex90-20261025-v1',
            zone='/europe-west4-a',status='RUNNING',machineType='/a2-ultragpu-1g',
            lastStartTimestamp='2026-10-26T08:00:00Z',
            disks=[dict(source='/disks/duplex-native-stage-20261024-v1',boot=True,autoDelete=False)],
            scheduling=dict(provisioningModel='FLEX_START',maxRunDuration={'seconds':'5400'},
                instanceTerminationAction='STOP',automaticRestart=False))
        from portable.venus.guard import stamp
        start=stamp(vm['lastStartTimestamp'])
        lease=dict(stop_unix=start+5400,prior_gpu_seconds=PRIOR_SECONDS,prior_scope_usd=PRIOR_USD)
        self.assertEqual(validate(vm,[vm],[],lease,start+10),start+5400)
        with self.assertRaises(ValueError):validate(vm,[vm],[{'status':'RUNNING'}],lease,start+10)
        with self.assertRaises(ValueError):validate(vm,[vm],[],{**lease,'prior_gpu_seconds':0},start+10)
        with self.assertRaises(ValueError):validate(vm,[vm],[],lease,start+5401)

    def test_approved_zone_b_binding_requires_exact_disk_machine_and_deadline(self):
        import json,hashlib
        from pathlib import Path
        from portable.venus.guard import stamp
        raw=Path('docs/NATIVE-FOUNDATION-CAPACITY-FALLBACK-AUTHORIZATION-2026-10-26.json').read_bytes()
        start=stamp('2026-10-26T10:00:00Z')
        lease=dict(resource_key='zone-b-a10040',vm_id='1962162267540853286',authorization_sha256=hashlib.sha256(raw).hexdigest(),
            creation_timestamp='2026-10-26T09:59:30Z',stop_unix=start+14000,
            prior_gpu_seconds=PRIOR_SECONDS,prior_scope_usd=PRIOR_USD)
        vm=dict(id='1962162267540853286',name='duplex-venus-audio-20261026-b-v1',zone='/europe-west4-b',status='RUNNING',
            creationTimestamp=lease['creation_timestamp'],lastStartTimestamp='2026-10-26T10:00:00Z',
            machineType='/a2-highgpu-1g',disks=[dict(source='/disks/duplex-native-stage-20261025-b-v1',boot=True,autoDelete=False)],
            scheduling=dict(provisioningModel='STANDARD',terminationTime='2026-10-26T13:53:20Z',instanceTerminationAction='STOP',automaticRestart=False))
        self.assertEqual(validate(vm,[vm],[],lease,start+10),start+14000)
        with self.assertRaises(ValueError):validate({**vm,'id':'999'},[vm],[],lease,start+10)
        with self.assertRaises(ValueError):validate({**vm,'machineType':'/a2-highgpu-2g'},[vm],[],lease,start+10)
        newer={**vm,'id':'777','scheduling':dict(provisioningModel='FLEX_START',maxRunDuration={'seconds':'3600'},instanceTerminationAction='STOP',automaticRestart=False)}
        next_lease={**lease,'vm_id':'777','stop_unix':start+3600,'prior_new_gpu_seconds_upper':1200}
        self.assertEqual(validate(newer,[newer],[],next_lease,start+10),start+3600)
        with self.assertRaises(ValueError):validate(newer,[newer],[],{**next_lease,'prior_new_gpu_seconds_upper':0},start+10)
