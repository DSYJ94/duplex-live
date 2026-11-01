"""Deterministic fact targets for authored synthetic scenarios, not native teachers.

These computations supply independent facts for later teacher review. They do
not simulate successful native dispatch, real provider reasoning or audio timing.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def facts(contract):
    operation=contract.get('operation')
    if operation=='amortization_comparison':
        principal=contract['principal'];rate=contract['nominal_annual_rate']/12
        results=[]
        for months in contract['months']:
            payment=principal*rate/(1-(1+rate)**(-months)) if rate else principal/months
            results.append(dict(months=months,monthly_payment_GBP=round(payment,2),
                                total_interest_GBP=round(payment*months-principal,2)))
        return dict(kind=operation,comparisons=results,rounding='unrounded payments used for total interest; displayed amounts rounded to pennies',
                    early_repayment_penalty_GBP=contract['early_repayment_penalty'])
    if operation=='paint_litres':
        def litres(length):
            return math.ceil(2*(length+contract['width'])*contract['height']*contract['coats']/contract['coverage_m2_per_litre'])
        return dict(kind=operation,original_litres=litres(contract['length']),
                    revised_litres=litres(contract['revised_length']),
                    deliverable_revision='revised_only',old_result_delivery_allowed=False)
    if operation=='given_fact_comparison':
        return dict(kind=operation,extra_cost_B_GBP=contract['panel_B_GBP']-contract['panel_A_GBP'],
                    supplied_fact_conclusion='B tolerates the stated damp conditions; A must stay dry.',
                    fictional_materials=True,real_building_advice=False)
    if contract.get('jobs')==['travel','meal']:
        rail,coach=contract['travel']['rail'],contract['travel']['coach'];meal=contract['meal']
        return dict(kind='selective_cancel',travel=dict(rail_minutes_saved=coach['minutes']-rail['minutes'],rail_extra_GBP=rail['GBP']-coach['GBP']),
                    meal_total_GBP={k:v*meal['guests'] for k,v in meal['per_person_GBP'].items()},
                    meal_within_budget={k:v*meal['guests']<=meal['budget_GBP'] for k,v in meal['per_person_GBP'].items()},
                    cancel_only='travel',keep_pending='meal',cancelled_result_delivery_allowed=False)
    raise ValueError('Unknown closed-world synthetic contract')


def build(plan):
    if plan['counts']!={'TRAIN':24,'DEV':12}:raise ValueError('Whole36 draft required')
    rows=[]
    for scene in plan['scenes']:
        contract=scene.get('worker_contract')
        row=dict(scene_id=scene['id'],fold=scene['fold'],family=scene['family'],
                 fact_targets=facts(contract) if contract else None,
                 behavioral_requirements=scene['teacher_requirements'],
                 teacher_history_available=False,audio_available=False,training_approved=False)
        if contract:row['task_contract_sha256']=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
        rows.append(row)
    if len(rows)!=36 or len({x['scene_id'] for x in rows})!=36:raise ValueError('Whole unique cohort required')
    return dict(schema='dl-native-context-fact-targets-v1',rows=rows,complete_fact_targets=True,
                optimizer_updates=0,provider_calls=0,native_dispatch_tested=False,
                causal_worker_receipts_available=False,teacher_speech_targets_available=False,
                qualification='OFFLINE_FACT_TARGETS_ONLY_NOT_NATIVE_TEACHERS')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve original fact targets')
    raw=a.plan.read_bytes();report=build(json.loads(raw));report['plan_sha256']=hashlib.sha256(raw).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
