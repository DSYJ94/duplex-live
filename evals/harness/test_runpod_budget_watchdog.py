import unittest
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from portable.context.runpod_budget_watchdog import budget_seconds


class RunpodBudget(unittest.TestCase):
    def receipt(self, seconds=7200):
        return dict(approved_gpu_wallclock_seconds=seconds, gpu_hourly_usd=1.79,
                    maximum_gpu_usd=1.79*seconds/3600, total_compute_budget_usd=50,
                    prior_attributed_spend_usd=20.096006687846966)

    def test_same_pod_can_have_a_bounded_longer_trial_within_existing_budget(self):
        for seconds in (7200,14400,21600,28800,36000,43200,50400):
            self.assertEqual(budget_seconds(self.receipt(seconds)),seconds)

    def test_ten_hours_counts_full_pod_lifetime_and_other_trial_reserve(self):
        r=self.receipt(36000);r['prior_attributed_spend_usd']=23.696006687846966
        self.assertEqual(budget_seconds(r),36000)
        self.assertLess(r['prior_attributed_spend_usd']+r['maximum_gpu_usd']+.25,50)
        r['prior_attributed_spend_usd']=32
        with self.assertRaises(ValueError):budget_seconds(r)

    def test_cost_ceiling_and_disk_reserve_are_enforced(self):
        for change in (dict(total_compute_budget_usd=51), dict(prior_attributed_spend_usd=46.2),
                       dict(maximum_gpu_usd=0), dict(approved_gpu_wallclock_seconds=86400),
                       dict(approved_gpu_wallclock_seconds=True), dict(gpu_hourly_usd=float('nan'))):
            receipt=self.receipt();receipt.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):budget_seconds(receipt)

    def test_replacement_can_use_only_the_remaining_bounded_project_funds(self):
        r=self.receipt(24300);r['prior_attributed_spend_usd']=37.36308350742475
        self.assertEqual(budget_seconds(r),24300)
        self.assertLess(r['prior_attributed_spend_usd']+r['maximum_gpu_usd']+.25,50)
        r['approved_gpu_wallclock_seconds']=25200;r['maximum_gpu_usd']=1.79*7
        with self.assertRaises(ValueError):budget_seconds(r)
        for seconds in (0,599,50401):
            with self.subTest(seconds=seconds),self.assertRaises(ValueError):budget_seconds(self.receipt(seconds))

    def test_twelve_hour_primary_still_reserves_one_hour_paired_behavior_under_fifty(self):
        r=self.receipt(43200);r['prior_attributed_spend_usd']=23.696006687846968
        self.assertEqual(budget_seconds(r),43200)
        primary=r['prior_attributed_spend_usd']+r['maximum_gpu_usd']+.25
        self.assertAlmostEqual(primary,45.42600668784697)
        self.assertLess(primary+3.58+.25,50)
        r['prior_attributed_spend_usd']=28.3
        with self.assertRaises(ValueError):budget_seconds(r)

    def test_fourteen_hour_same_gpu_has_no_second_pod_reserve(self):
        r=self.receipt(50400);r['prior_attributed_spend_usd']=23.696006687846968
        self.assertEqual(budget_seconds(r),50400)
        self.assertAlmostEqual(r['prior_attributed_spend_usd']+r['maximum_gpu_usd']+.25,49.00600668784697)
        r['prior_attributed_spend_usd']+=1.
        with self.assertRaises(ValueError):budget_seconds(r)

    def test_human_expansion_requires_the_exact_recorded_authorization(self):
        from portable.context.runpod_budget_watchdog import AUTHORIZATION
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'authorization.json';path.write_bytes(AUTHORIZATION.read_bytes())
            r=self.receipt(72000);r.update(total_compute_budget_usd=100,
                prior_attributed_spend_usd=36.94071830424946,
                budget_authorization_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            with patch('portable.context.runpod_budget_watchdog.AUTHORIZATION',path):
                self.assertEqual(budget_seconds(r),72000)
                for changed in ('missing','digest','document'):
                    item=r.copy()
                    if changed=='missing':item.pop('budget_authorization_sha256')
                    if changed=='digest':item['budget_authorization_sha256']='0'*64
                    if changed=='document':
                        d=json.loads(path.read_text());d['human_instruction']='unapproved budget change'
                        path.write_text(json.dumps(d));item['budget_authorization_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                    with self.subTest(changed=changed),self.assertRaises(ValueError):budget_seconds(item)

    def test_expanded_cap_still_counts_full_lifetime_and_larger_disk_reserve(self):
        from portable.context.runpod_budget_watchdog import AUTHORIZATION
        r=self.receipt(120000);r.update(total_compute_budget_usd=100,
            prior_attributed_spend_usd=36.94071830424946,
            budget_authorization_sha256=hashlib.sha256(AUTHORIZATION.read_bytes()).hexdigest())
        self.assertEqual(budget_seconds(r),120000)
        for change in (dict(approved_gpu_wallclock_seconds=129601),
                       dict(prior_attributed_spend_usd=41),dict(maximum_gpu_usd=1)):
            item=r.copy();item.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):budget_seconds(item)
        r=self.receipt(7200);r.update(total_compute_budget_usd=100,
            prior_attributed_spend_usd=96.1,
            budget_authorization_sha256=hashlib.sha256(AUTHORIZATION.read_bytes()).hexdigest())
        # GPU3.58 + prior96.1 fits a0.25 reserve, but fails the expanded0.50 reserve.
        with self.assertRaises(ValueError):budget_seconds(r)


if __name__=='__main__':unittest.main()
