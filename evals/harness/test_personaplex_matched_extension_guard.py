import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from training.personaplex import matched_guard as g


class ExtensionGuard(unittest.TestCase):
    def approval(self):
        return dict(status='OWNER_APPROVED_MATCHED_TRAINING_ONLY_CONTINUATION',
            proposal_commit='298521a76959a9d4f60c85180456944d37a27ca6',provider='Isambard',account=g.ACCOUNT,
            maximum_aggregate_GPU_hours=112,maximum_single_job_seconds=86400,simultaneous_GPUs=2,
            original_admission_source='8b822b0d85ccea4f291255204cbc358fa85469c4',
            new_external_model_API_calls=0,new_purchases=False,new_GCP_compute=False,repeat_native_calibration=False,maximum_future_running_intervals=4)

    def test_default_remains96_and_missing_or_pending_amendment_never_extends(self):
        self.assertEqual(g.phase_limit({}),96*3600)
        with tempfile.TemporaryDirectory() as root,patch.object(g,'EXTENSION',Path(root)/'absent.json'):
            with self.assertRaises(FileNotFoundError):g.phase_limit({'continuation_interval':1,'continuation_authorization_sha256':'unapproved'})
            g.EXTENSION.write_text(json.dumps(dict(self.approval(),status='PENDING')))
            with self.assertRaises(ValueError):g.phase_limit({'continuation_interval':1,'continuation_authorization_sha256':g.sha(g.EXTENSION)})

    def test_only_exact_grant_hash_scope_and112_limit_accepted(self):
        with tempfile.TemporaryDirectory() as root,patch.object(g,'EXTENSION',Path(root)/'fixture.json'):
            g.EXTENSION.write_text(json.dumps(self.approval()));lease={'continuation_interval':1,'continuation_authorization_sha256':g.sha(g.EXTENSION)}
            self.assertEqual(g.phase_limit(lease),112*3600)
            for key,value in [('maximum_aggregate_GPU_hours',113),('simultaneous_GPUs',3),('repeat_native_calibration',True),('proposal_commit','different')]:
                g.EXTENSION.write_text(json.dumps(dict(self.approval(),**{key:value})))
                with self.subTest(key=key),self.assertRaises(ValueError):g.phase_limit({'continuation_interval':1,'continuation_authorization_sha256':g.sha(g.EXTENSION)})
            g.EXTENSION.write_text(json.dumps(self.approval()))
            with self.assertRaises(ValueError):g.phase_limit({'continuation_interval':1,'continuation_authorization_sha256':'wrong-hash'})


if __name__=='__main__':unittest.main()
