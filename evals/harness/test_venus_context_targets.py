import json
from pathlib import Path
import unittest
from data.venus.context_targets import facts,build


class ContextTargetTests(unittest.TestCase):
    def test_amortization_zero_rate_and_positive_rate_accounting(self):
        c=dict(operation='amortization_comparison',principal=12000,nominal_annual_rate=0,months=[12],early_repayment_penalty=0)
        assert facts(c)['comparisons']==[dict(months=12,monthly_payment_GBP=1000,total_interest_GBP=0)]
        c.update(principal=1000,nominal_annual_rate=.12,months=[1])
        assert facts(c)['comparisons']==[dict(months=1,monthly_payment_GBP=1010,total_interest_GBP=10)]
    
    
    def test_paint_replaces_old_result_instead_of_adding_results(self):
        c=dict(operation='paint_litres',length=4,revised_length=6,width=5,height=2.4,coats=2,coverage_m2_per_litre=10)
        f=facts(c)
        assert (f['original_litres'],f['revised_litres'])==(9,11)
        assert not f['old_result_delivery_allowed']
    
    
    def test_complete_cohort_stays_unapproved_and_no_receipts_invented(self):
        r=build(json.loads(Path('evals/fixtures/venus-context-repair-authoring-v2-20261101.json').read_text()))
        assert len(r['rows'])==36 and sum(x['fact_targets'] is not None for x in r['rows'])==24
        assert all(not x['teacher_history_available'] and not x['training_approved'] for x in r['rows'])
        assert not r['causal_worker_receipts_available'] and not r['native_dispatch_tested']
        cancelled=[x['fact_targets'] for x in r['rows'] if x['family']=='selective_cancel']
        assert all(x['cancel_only']=='travel' and x['keep_pending']=='meal' and x['meal_total_GBP']=={'pasta':18,'soup':12} for x in cancelled)
    
    
    def test_unknown_contract_not_silently_accepted(self):
        with self.assertRaises(ValueError):facts({'operation':'invent_facts'})
