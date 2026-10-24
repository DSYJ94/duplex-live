import unittest

from portable.context.gcp_guest_cost_watchdog import stop_reason
from portable.context.gcp_budget_watchdog import AUTH_SHA, stamp


class GuestCostGuardTests(unittest.TestCase):
    def setUp(self):
        self.lease=dict(id='123',stop_utc='2026-10-25T12:00:00Z')
        self.now=stamp('2026-10-25T00:05:00Z')
        self.guard=dict(provider='gcp',resource_id='123',authorization_sha256=AUTH_SHA,
            deadline_unix=stamp(self.lease['stop_utc']),checked_unix=self.now,
            execution_authorized=True,live_provider_deadline_verified=True,
            live_external_cost_guard_verified=True,single_gpu_across_providers_verified=True)

    def test_fresh_authorized_same_resource_continues(self):
        self.assertIsNone(stop_reason(self.guard,self.lease,self.now))

    def test_missing_stale_or_nonfinite_heartbeat_requires_shutdown(self):
        for malformed in (None, [], 'invalid'):
            self.assertEqual(stop_reason(malformed,self.lease,self.now),'control_guard_unreadable')
        for changed in (None,self.now-91,float('nan'),True,self.now+1):
            with self.subTest(checked=changed):
                self.assertEqual(stop_reason(dict(self.guard,checked_unix=changed),self.lease,self.now),
                                 'mac_guard_lost_or_stale')

    def test_changed_resource_scope_or_permission_requires_shutdown(self):
        for changes in (dict(resource_id='999'),dict(authorization_sha256='0'*64),
                        dict(deadline_unix=self.guard['deadline_unix']+1),
                        dict(live_external_cost_guard_verified=False)):
            with self.subTest(changes=changes):
                self.assertIsNotNone(stop_reason(dict(self.guard,**changes),self.lease,self.now))

    def test_native_deadline_is_enforced_even_with_a_fresh_heartbeat(self):
        end=stamp(self.lease['stop_utc'])
        self.assertEqual(stop_reason(dict(self.guard,checked_unix=end),self.lease,end),
                         'approved_native_deadline_reached')


if __name__ == '__main__':
    unittest.main()
