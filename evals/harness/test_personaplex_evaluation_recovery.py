import json
from pathlib import Path
import unittest
from unittest.mock import patch
from training.personaplex import evaluation_recovery_policy as policy, matched_guard as g
from training.personaplex.evaluation_recovery import accounting


class RecoveryPolicy(unittest.TestCase):
    def lease(self):
        old=json.loads(Path('evals/fixtures/native-moshi-evaluation-8352856-lease.json').read_text())
        old.update(continuation_interval=11,optimizer_updates=0,maximum_seconds=14400,prior_GPU_seconds=269405)
        old[policy.KEY]=policy.authority()
        return old

    def test_zero_update_recovery_fits_original_budget_and_actual_authority(self):
        lease=self.lease();self.assertEqual(g.phase_limit(lease),112*3600)
        self.assertEqual(policy.validate(lease),150)
        self.assertEqual(g.resource(lease),lease)

    def test_cannot_train_extend_interval_or_reset_time_cap(self):
        for changes in [dict(optimizer_updates=1),dict(continuation_interval=12),dict(maximum_seconds=14401),{policy.KEY:'wrong'}]:
            lease=self.lease();lease.update(changes)
            with self.assertRaises(ValueError):g.resource(lease)

    def test_recovery_cannot_exceed_carried_total(self):
        lease=self.lease();lease['prior_GPU_seconds']=112*3600-100
        with self.assertRaises(ValueError):g.resource(lease)

    def test_failed_evaluation_time_is_counted_without_relabelling_it_successful(self):
        text='8352856|duplex-matched-final10-v1|brics.u6ada|FAILED|1:0|2980|gres/gpu=2'
        with patch('training.personaplex.evaluation_recovery.resume.allocation_carry',return_value=263445):
            self.assertEqual(accounting(text,[]),269405)
            with self.assertRaises(ValueError):accounting(text.replace('2980','0'),[])
            with self.assertRaises(ValueError):accounting(text.replace('FAILED','COMPLETED'),[])
