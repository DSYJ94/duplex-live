import unittest
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


if __name__=='__main__':unittest.main()
