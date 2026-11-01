"""Authored causal teacher specification, not executed/qualified histories."""
import argparse,hashlib,json
from decimal import Decimal
from pathlib import Path
from .context_targets import facts
from .context_source_quality import number


def money(value):
    cents=int(Decimal(str(value))*100);pence=cents%100
    return number(cents//100)+' pounds'+(' and '+number(pence)+' pence' if pence else '')


def build(plan):
    rows=[]
    for scene in plan['scenes']:
        family=scene['family'];i=int(scene['id'].split('-')[-1]);contract=scene.get('worker_contract');f=facts(contract) if contract else None
        actions=[]
        def add(turn,kind,text,**extra):actions.append(dict(after_user_turn=turn,kind=kind,text=text,**extra))
        if family=='quiet_scope':add(2,'speak','A lid covers an opening and helps keep its contents inside.')
        elif family=='simple_or_delegate':
            add(0,'speak',number(i+5).capitalize()+'.')
            add(1,'delegate',scene['user_turns'][1],task_label='loan')
            text=' '.join(number(x['months']).capitalize()+' months: '+money(x['monthly_payment_GBP'])+' per month, with '+money(x['total_interest_GBP'])+' total interest.' for x in f['comparisons'])
            add(1,'speak',text,requires_result_for='loan')
        elif family=='selective_cancel':
            c=contract
            add(0,'delegate','Compare the synthetic travel options: rail takes ninety minutes at twenty-eight pounds; coach takes one hundred and forty minutes at sixteen pounds.',task_label='travel')
            add(0,'delegate','Compare vegetarian pasta at three pounds per person and soup at two pounds per person for six guests, under a thirty-six pound budget.',task_label='meal')
            add(1,'cancel','Retire only the travel task delivery; preserve the meal task.',task_label='travel')
            add(1,'speak','Travel results are cancelled. The meal comparison is still pending.',requires_cancel_receipt_for='travel',requires_pending='meal')
            add(2,'speak','We can talk about the meal while that comparison finishes.')
            add(2,'speak','Pasta costs eighteen pounds and soup costs twelve pounds for six people. Both are within the thirty-six pound budget.',requires_result_for='meal')
        elif family=='correction':
            add(0,'delegate',scene['user_turns'][0],task_label='paint',revision=1)
            corrected=scene['user_turns'][0].replace(str(contract['length'])+' metres long',str(contract['revised_length'])+' metres long')
            if corrected==scene['user_turns'][0]:raise ValueError('Correction did not replace the original dimension')
            add(1,'replace',corrected,task_label='paint',revision=2,retire_revision=1)
            add(2,'speak','Using the corrected length, you need '+number(f['revised_litres'])+' litres.',requires_result_for='paint',revision=2)
        elif family=='pause_resume':
            add(0,'speak','First, clear loose items from the floor.')
            add(1,'pause','',speech_forbidden=True)
            add(3,'speak','Next, cover the furniture. Finally, mask the edges.',resume_without_restart=True)
        elif family=='result_delivery':
            add(0,'delegate',scene['user_turns'][0],task_label='panels')
            add(1,'speak','Insulation slows heat moving between places.')
            add(2,'speak','Panel B costs '+money(f['extra_cost_B_GBP'])+' more and tolerates the stated damp conditions. Panel A must stay dry.',requires_result_for='panels')
        else:raise ValueError('All authored families need explicit targets')
        rows.append(dict(id=scene['id'],fold=scene['fold'],family=family,actions=actions,fact_targets=f,training_approved=False))
    if len(rows)!=36:raise ValueError('Whole cohort required')
    return dict(schema='dl-native-context-teacher-plan-v1',status='DRAFT_REQUIRES_ACTUAL_CAUSAL_RECEIPTS_SPEECH_TIMING_AND_INDEPENDENT_APPROVAL',scenes=rows,counts={'TRAIN':24,'DEV':12},actual_worker_receipts=False,native_student_dispatch_claimed=False,speech_audio_generated=False,training_approved=False,external_calls=0,optimizer_updates=0,admission_requirements=['Bind all72 unchanged source recordings and their complete derived word review.','Execute task/cancel/replace work with actual task identities; enforce result-before-speech and selective retirement.','Generate and independently audit the complete assistant speech target collection; silence gets no speech request.','Use measured word timings for native scheduling; real pause must resume the unsaid tail.','Keep full original216TRAIN/52DEV and retention objectives; only admit new native caches after complete semantic/causal checks.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior draft')
    raw=a.source.read_bytes();r=build(json.loads(raw));r['source_plan_sha256']=hashlib.sha256(raw).hexdigest();a.output.write_text(json.dumps(r,indent=2)+'\n')
