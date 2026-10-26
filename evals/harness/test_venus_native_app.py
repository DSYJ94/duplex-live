import asyncio
import unittest
from portable.venus.native_app import ArithmeticWorker, DelegateParser, explicit_sum_numbers
from portable.venus.guard import validate, IDENT, PRIOR_SECONDS, PRIOR_USD


class NativeProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_recorded_english_delegate_is_computed_not_rejected(self):
        w=ArithmeticWorker(0)
        w.submit('Total three hundred eighteen, two hundred forty seven, and five hundred seventy.')
        await asyncio.gather(*w.tasks.values())
        self.assertIn('Computed sum: 1135',w.take_results())
        w.submit('Total three hundred and eighteen, two hundred and forty seven, and five hundred sixty-nine.')
        await asyncio.gather(*w.tasks.values())
        self.assertIn('Computed sum: 1134',w.take_results())
        await w.close()

    def test_ambiguous_word_numbers_and_decimals_are_not_silently_added(self):
        self.assertEqual(explicit_sum_numbers('Total one hundred, twenty, and zero.'),[100,20,0])
        with self.assertRaises(ValueError):explicit_sum_numbers('Add three hundred eighteen and two hundred forty seven')
        with self.assertRaises(ValueError):explicit_sum_numbers('Add1.5 and2.5')
        with self.assertRaises(ValueError):explicit_sum_numbers('Add one thousand, two.')
        with self.assertRaises(ValueError):explicit_sum_numbers('Add negative three, two.')
        with self.assertRaises(ValueError):explicit_sum_numbers('Sum query{"id":"w10","numbers":[318,247,570]}')

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

    def test_regional_binding_carries_budget_and_rejects_other_zones(self):
        import json,hashlib
        from pathlib import Path
        from portable.venus.guard import stamp,resource
        raw=Path('docs/NATIVE-REGIONAL-PLACEMENT-AUTHORIZATION-2026-10-26.json').read_bytes()
        start=stamp('2026-10-26T15:00:00Z')
        lease=dict(resource_key='us-a10040',vm_id='123',disk_id='456',zone='us-central1-a',
            authorization_sha256=hashlib.sha256(raw).hexdigest(),disk_size_gib=100,
            creation_timestamp='2026-10-26T14:59:30Z',stop_unix=start+7200,
            prior_new_gpu_seconds_upper=1200,prior_gpu_seconds=PRIOR_SECONDS,prior_scope_usd=PRIOR_USD)
        vm=dict(id='123',name='duplex-venus-us-20261026-v1',zone='/us-central1-a',status='RUNNING',
            creationTimestamp=lease['creation_timestamp'],lastStartTimestamp='2026-10-26T15:00:00Z',
            machineType='/a2-highgpu-1g',disks=[dict(source='/disks/duplex-venus-us-20261026-v1',boot=True,autoDelete=False)],
            scheduling=dict(provisioningModel='FLEX_START',maxRunDuration={'seconds':'7200'},instanceTerminationAction='STOP',automaticRestart=False))
        self.assertEqual(validate(vm,[vm],[],lease,start+10),start+7200)
        for bad in ({'zone':'europe-west1-b'},{'authorization_sha256':'bad'},{'prior_new_gpu_seconds_upper':0},
                    {'disk_size_gib':250},{'prior_new_gpu_seconds_upper':10000}):
            with self.assertRaises(ValueError):validate(vm,[vm],[],{**lease,**bad},start+10)

    def test_supplemental_iap_hour_never_resets_paid_history(self):
        from unittest.mock import patch
        from portable.venus.guard import stamp
        start=stamp('2026-10-27T10:00:00Z')
        b=dict(name='duplex-venus-us-20261026-v1',zone='us-central1-a',ident='5664322998932521310',
            disk='7763575121976148318',disk_name='duplex-venus-us-20261026-v1',machine='a2-highgpu-1g')
        lease=dict(resource_key='us-a10040',access_mode='iap',disk_size_gib=100,creation_timestamp='2026-10-26T16:28:00Z',
            prior_new_gpu_seconds_upper=3600,prior_gpu_seconds=PRIOR_SECONDS,prior_scope_usd=PRIOR_USD,stop_unix=start+3600)
        vm=dict(id=b['ident'],name=b['name'],zone='/us-central1-a',status='RUNNING',machineType='/a2-highgpu-1g',
            creationTimestamp=lease['creation_timestamp'],lastStartTimestamp='2026-10-27T10:00:00Z',
            tags={'items':['duplex-venus-iap-bounded']},disks=[dict(source='/disks/'+b['disk_name'],boot=True,autoDelete=False)],
            scheduling=dict(provisioningModel='FLEX_START',maxRunDuration={'seconds':'3600'},instanceTerminationAction='STOP',automaticRestart=False))
        # resource() separately requires the direct-human supplemental authority.
        with patch('portable.venus.guard.resource',return_value=b):
            self.assertEqual(validate(vm,[vm],[],lease,start+10),start+3600)
            with self.assertRaises(ValueError):validate(vm,[vm],[],{**lease,'prior_new_gpu_seconds_upper':0},start+10)
            with self.assertRaises(ValueError):validate(vm,[vm],[],{**lease,'prior_new_gpu_seconds_upper':14000},start+10)
            with self.assertRaises(ValueError):validate({**vm,'tags':{'items':['duplex-native-bounded']}},[vm],[],lease,start+10)

    def test_single_read_timeout_retries_without_reusing_a_snapshot(self):
        import subprocess
        from unittest.mock import patch
        from portable.venus.guard import bounded_cloud_read
        with patch('portable.venus.guard.cloud',side_effect=[subprocess.TimeoutExpired('gcloud',12),[{'status':'TERMINATED'}]]) as call:
            self.assertEqual(bounded_cloud_read('instances','list'),[{'status':'TERMINATED'}])
            self.assertEqual([x.kwargs['timeout'] for x in call.call_args_list],[12,8])

    def test_persistent_read_timeouts_still_fail_closed(self):
        import subprocess
        from unittest.mock import patch
        from portable.venus.guard import bounded_cloud_read
        with patch('portable.venus.guard.cloud',side_effect=subprocess.TimeoutExpired('gcloud',12)) as call:
            with self.assertRaises(subprocess.TimeoutExpired):bounded_cloud_read('instances','list')
            self.assertEqual(call.call_count,2)

    def test_actual_native_frame_markers_do_not_enter_worker_arguments(self):
        p=DelegateParser()
        raw='query{"task":"explain why<|chunk_eos|></unit><|speak|> plants need light"}'
        self.assertEqual(p.feed('<delegate>'+raw+'</delegate>'),['query{"task":"explain why plants need light"}'])
        self.assertEqual(p.last_raw_queries,[raw])

    def test_native_chunk_boundary_inside_word_preserves_integer_meaning(self):
        p=DelegateParser()
        self.assertEqual(p.feed('<delegate>Total three hun<|chunk_eos|></unit><|speak|>'),[])
        values=p.feed('dred eighteen, two hundred forty seven, and five hundred seventy.</delegate>')
        self.assertEqual(explicit_sum_numbers(values[0]),[318,247,570])
        self.assertIn('<|chunk_eos|>',p.last_raw_queries[0])
