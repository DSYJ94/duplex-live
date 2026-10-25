import unittest
from unittest.mock import Mock
from portable.context.native_guard_startup import await_guest_guard


class StartupTests(unittest.TestCase):
    def run_guard(self, values):
        read=Mock(side_effect=values)
        result=await_guest_guard(read,'guard.service','/tmp/status.json',attempts=2,
                                 pause=lambda _:None,now=lambda:100)
        return result,read

    def test_missing_first_file_then_fresh_exact_process_passes(self):
        props='MainPID=9\nActiveState=active\nInvocationID=abc'
        payload='{"phase":"armed","pid":9,"at_utc":"1970-01-01T00:01:40+00:00"}'
        (guest,_),read=self.run_guard([props,'pending',props,payload])
        self.assertEqual(guest['pid'],9)
        self.assertEqual(read.call_count,4)

    def test_failed_service_cannot_be_hidden_by_pending_file(self):
        with self.assertRaises(RuntimeError):self.run_guard(['MainPID=0\nActiveState=failed'])

    def test_wrong_pid_or_stale_file_never_arms(self):
        props='MainPID=9\nActiveState=active\nInvocationID=abc'
        for payload in ('{"phase":"armed","pid":8,"at_utc":"1970-01-01T00:01:40+00:00"}',
                        '{"phase":"armed","pid":9,"at_utc":"1970-01-01T00:00:00+00:00"}'):
            with self.assertRaises(TimeoutError):self.run_guard([props,payload]*2)

    def test_guest_shutdown_or_malformed_status_fails(self):
        props='MainPID=9\nActiveState=active\nInvocationID=abc'
        for payload in ('{"phase":"guest_shutdown_requested"}','invalid'):
            with self.assertRaises((RuntimeError,ValueError)):self.run_guard([props,payload])


if __name__=='__main__':unittest.main()
