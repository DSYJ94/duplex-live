"""Focused control repair sources; retain all old TRAIN and DEV unchanged."""
import argparse,copy,hashlib,json,random
from pathlib import Path

PARENT=Path('evals/fixtures/venus-control-corpus-v4-20261026.json')

def build():
    original=json.loads(PARENT.read_text());retained=[copy.deepcopy(s) for s in original['scenes'] if s['fold'] in ('TRAIN','DEV')]
    additions=[]
    def add(fold,family,group,turns,program,world=None):
        additions.append(dict(id=f'venus-repair-v1-{fold.lower()}-{family}-{len(additions):04d}',fold=fold,family=family,
            split_group=fold+':repair:'+group,user_turns=turns,teacher_program=program,synthetic_private_world=world,source='authored_synthetic',owner_conversations=False))
    specs=[('TRAIN',['apricot','daffodil','bluebell','starlight','daybreak','windmill','sunbeam','rainbow'],
            ['suitcase','toolbox','mailbox','backpack','notepad','wallet','shelf','desk'],
            [('What does a key usually open?','A lock.')],['jade compass','ruby anchor','silver lantern','golden feather']),
           ('DEV',['sunflower','seashell'],['saucepan','drawer'],
            [('What does a thermometer measure?','Temperature.'),('How many days are in a week?','Seven.'),('What is the opposite of smooth?','Rough.'),('What do you use a sieve for?','Separating larger pieces from smaller ones.')],['orange ribbon','purple diamond']),
           ('HOLDOUT',['lavender','snowdrop'],['cupboard','bookcase'],
            [("What do a bird's wings help it do?",'They help it fly.'),('What is the opposite of loose?','Tight.'),('How many minutes are in an hour?','Sixty.'),('What is a ruler used to measure?','Length.')],['green diamond','red crescent'])]
    quiet_patterns=[
        'For now, make no sound at all. Wait until a later message says {cue}. Do not acknowledge this instruction or ask me for the cue.',
        'I need silent listening, including no brief acknowledgements. Only speak after I later say {cue}. The cue here is part of the instruction.',
        'Remain completely quiet while I think. A future utterance containing {cue} releases you. Until then, no speech or filler sounds.'
    ]
    for fold,cues,keys,facts,values in specs:
        for cue in cues:
            for variant in range(3 if fold=='TRAIN' else 2):
                question,answer=facts[variant%len(facts)]
                add(fold,'quiet','cue-'+cue,[quiet_patterns[variant].format(cue=cue),'I have not released you yet. Please keep listening silently.',cue+'. '+question],dict(kind='quiet_until_later_cue',cue=cue,quiet_gap_seconds=4,release_answer=answer,no_acknowledgment=True))
        for key in keys:
            for variant in range(3 if fold=='TRAIN' else 2):
                value=values[variant%len(values)];question,answer=facts[variant%len(facts)]
                add(fold,'cancel','cancel-'+key,[f'Please start a background lookup for the marker in {key}.','Cancel that pending lookup now. Do not bring back its result.',question],
                    dict(kind='cancel_lookup',query='lookup '+key,cancel_query='cancel pending',worker_delay_seconds=18,cancel_acknowledgment='Cancelled.',ordinary_answer=answer,no_cancelled_value_delivery=True),{key:value})
        for key in keys[:4 if fold=='TRAIN' else 2]:
            for value in values:
                add(fold,'delegate','lookup-'+key,[f'Use the background worker to look up the marker in {key}.'],dict(kind='lookup',query='lookup '+key,worker_delay_seconds=12,result_answer='The marker is '+value+'.',result_must_follow_receipt=True),{key:value})
        if fold!='TRAIN':
            for i,(question,answer) in enumerate(facts):add(fold,'ordinary','ordinary-'+str(i),[question],dict(kind='ordinary_answer',answer=answer,delegate_forbidden=True))
        if fold=='HOLDOUT':
            for n in range(4):
                first,second=[('pick up the empty glass','set it on the coaster'),('take the scarf','hang it on the hook')][n%2]
                request=['Explain these actions slowly: {a}, then {b}.','Talk me through this gently: {a}, followed by {b}.','Walk through this sequence: {a}, and after that {b}.','Describe one step at a time: {a}, then {b}.'][n].format(a=first,b=second)
                add(fold,'pause','steps-'+str(n),[request,'Stop speaking for a moment. Wait until I say continue.','Continue.'],dict(kind='pause_resume',first_text='First, '+first+'.',remaining_text='Then, '+second+'.',quiet_gap_seconds=4,no_restart=True))
            for key in keys:
                for variant,value in enumerate(values):
                    question,answer=facts[variant]
                    add(fold,'pending','pending-'+key,[f'Use the background worker to look up the marker in {key}.','While you wait, '+question],dict(kind='lookup_with_followup',query='lookup '+key,worker_delay_seconds=22,followup_answer=answer,result_answer='The marker is '+value+'.',answer_followup_while_pending=True,result_must_follow_receipt=True),{key:value})
    scenes=retained+additions;counts={f:sum(s['fold']==f for s in scenes) for f in ['TRAIN','DEV','HOLDOUT']}
    assert counts==dict(TRAIN=216,DEV=52,HOLDOUT=24) and len(additions)==104
    assert {s['id']:s for s in retained}=={s['id']:s for s in original['scenes'] if s['fold'] in ('TRAIN','DEV')}
    assert not ({s['id'] for s in scenes}&{s['id'] for s in original['scenes'] if s['fold']=='HOLDOUT'})
    random.Random(20261027).shuffle(scenes)
    return dict(schema='dl-venus-focused-control-repair-v1',status='UNAPPROVED_PENDING_COMPLETE_SOURCE_TEACHER_SEMANTIC_NATIVE_GATES',
        parent_plan=str(PARENT),parent_plan_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),counts=counts,
        added_counts={f:sum(s['fold']==f for s in additions) for f in counts},scenes=scenes,
        old_train_and_dev_preserved_exactly=True,old_holdout_retained_in_parent_as_inspected_diagnostic=True,
        fresh_holdout_never_used_for_optimizer=True,optimizer_updates=0,external_api_calls=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);assert not out.exists();out.write_text(json.dumps(build(),indent=2)+'\n')
