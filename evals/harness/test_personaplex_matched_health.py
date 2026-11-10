import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from training.personaplex import matched_health as h,matched_guard as g


class IndependentHeartbeat(unittest.TestCase):
    def receipt(self,path,**changes):
        value=dict(job_id='8260542',source='actual',passed=True,checked_unix=100.,
            home_bytes=2*1024**3,run_bytes=2*1024**3,separate_filesystem=True)
        value.update(changes);path.write_text(json.dumps(value))

    def test_slow_shared_storage_scan_is_not_in_scheduler_renewal_path(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'storage.json';self.receipt(p)
            with patch.object(h.time,'time',return_value=110.),patch.object(g,'refresh',return_value=110.) as refresh,\
                 patch('training.personaplex.matched_quota_recovery.storage_check',side_effect=AssertionError('NFS scan blocks live renewal')):
                checked,delay=h.heartbeat_once(dict(job_id='8260542',source='actual'),Path(directory)/'guard.json',p,100.)
            self.assertEqual(checked,110.);self.assertEqual(delay,10);refresh.assert_called_once()

    def test_timeout_does_not_add_sleep_or_extend_original_success(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'storage.json';self.receipt(p)
            with patch.object(h.time,'time',return_value=133.),patch.object(g,'refresh',return_value=100.):
                self.assertEqual(h.heartbeat_once(dict(job_id='8260542',source='actual'),Path(directory)/'guard.json',p,100.),(100.,0))

    def test_expired_failed_oversized_or_wrong_identity_storage_never_renews(self):
        for changed in ({'passed':False},{'checked_unix':-20.},{'checked_unix':111.},
                        {'source':'other'},{'job_id':'different'},{'home_bytes':81*1024**3},
                        {'run_bytes':33*1024**3},{'separate_filesystem':False}):
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory)/'storage.json';self.receipt(p,**changed)
                with patch.object(h.time,'time',return_value=110.),patch.object(g,'refresh') as refresh:
                    with self.assertRaises(RuntimeError):h.heartbeat_once(dict(job_id='8260542',source='actual'),Path(directory)/'guard.json',p)
                    refresh.assert_not_called()


if __name__=='__main__':unittest.main()
