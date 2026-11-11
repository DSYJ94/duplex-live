import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from training.personaplex import paced_observations as p


class ObservationAdapters(unittest.TestCase):
    def test_provider_preserves_original_time_and_validates_original_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'provider.json'
            record = dict(checked_unix=1000, source='fixture')
            path.write_text(json.dumps(record))
            with patch.object(p.matched_guard, 'refresh', return_value=1000) as refresh, \
                 patch.object(p.matched_guard, 'ensure') as ensure:
                result = p.provider({'job_id': 'fixture'}, path)
            refresh.assert_called_once_with({'job_id': 'fixture'}, path, last_success=None)
            ensure.assert_called_once_with({'job_id': 'fixture'}, record, now=1000)
            self.assertEqual(result['observed_unix'], 1000)
            self.assertEqual(result['original_provider_receipt'], record)

    def test_provider_timeout_cannot_read_old_receipt(self):
        with patch.object(p.matched_guard, 'refresh', side_effect=subprocess.TimeoutExpired('fixture', 20)):
            with self.assertRaises(subprocess.TimeoutExpired): p.provider({}, '/nonexistent')

    def test_local_process_or_storage_failure_cannot_create_liveness(self):
        with patch.object(p.local_process_identity, 'verify', side_effect=RuntimeError('PID reused')):
            with self.assertRaises(RuntimeError): p.local({}, {}, '/nonexistent')
        with patch.object(p.local_process_identity, 'verify', return_value={'pid': 123}), \
             patch.object(p.matched_health, 'storage_valid', side_effect=RuntimeError('expired')):
            with self.assertRaises(RuntimeError): p.local({}, {}, '/nonexistent')

    def test_local_preserves_storage_timestamp(self):
        with patch.object(p.local_process_identity, 'verify', return_value={'pid': 123}), \
             patch.object(p.matched_health, 'storage_valid', return_value={'checked_unix': 980}):
            result = p.local({'job_id': 'fixture'}, {}, '/fixture', clock=lambda: 1000)
        self.assertEqual(result['observed_unix'], 1000)
        self.assertEqual(result['storage_observed_unix'], 980)
        self.assertNotIn('provider_observed_unix', result)


if __name__ == '__main__': unittest.main()
