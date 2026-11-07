"""Author a new96 TRAIN proposal; no audio, API, cache or optimizer execution."""
import argparse,hashlib,json
from pathlib import Path

ORDINARY=[
('I have two identical watering cans. One is full and one is empty. Which should feel heavier?','The full can should feel heavier because it also contains water.'),
('We left an ice cube on a warm plate. What change would you expect?','The ice should melt as heat passes into it from the warmer surroundings.'),
('The map says this path goes uphill. What should I expect as I walk along it?','You should expect to gain height as you walk along the path.'),
('Why does a backpack often have padded shoulder straps?','The padding spreads the pressure and can make the backpack more comfortable to carry.'),
('A cardboard box is marked fragile. What is that telling me?','Its contents may break easily, so the box needs careful handling.'),
('Why might I put a lid on a jar before carrying it across the room?','A lid can help keep the contents inside and reduce the chance of a spill.'),
('I need to mark a page without folding it. What simple object could I use?','You could use a bookmark or a clean strip of paper.'),
('Why does a sponge become heavier after soaking up water?','It holds water in its small spaces, adding to its weight.'),
('When people say a task is optional, do they mean it is required?','No. Optional means you can choose whether to do it.'),
('My book is upside down. What simple movement would make it easier to read?','Turn the book around so the writing is the right way up.'),
('What does a kitchen timer help me keep track of?','It helps you keep track of how much time has passed or remains.'),
('Why can a reusable bottle be convenient on a long walk?','It lets you carry a drink and refill the bottle when suitable water is available.'),
('What is the difference between borrowing a book and keeping it as a gift?','A borrowed book is meant to be returned. A gift is yours to keep.'),
('If I want to sort shirts by colour, what should I compare?','Compare their colours and put shirts with matching colours together.'),
('A parcel is too wide for a narrow slot. What is the straightforward problem?','The opening is narrower than the parcel, so the parcel will not fit through it.'),
('Why would someone label two identical storage boxes?','Labels help show what is inside without opening each box.'),
('If one train arrives before another, which arrival happens earlier?','The first train to arrive has the earlier arrival.'),
('What does it mean to fold a sheet of paper in half?','Bring two opposite sides together so the paper forms two matching parts.'),
('Why do we use oven gloves when handling a hot tray?','They provide insulation that helps protect our hands from the heat.'),
('I want to keep a stack of papers together temporarily. What could help?','A paper clip can hold the papers together and is easy to remove.'),
('What is the purpose of an address on a parcel?','It tells the delivery service where the parcel is meant to go.'),
('What changes when a room becomes brighter?','More light reaches the room, making things easier to see.'),
('If a shop is closed for the day, can I assume its normal counter is serving customers?','No. Closed for the day means normal service has ended until it opens again.'),
('Why would I check the size of a picture before buying a frame?','The frame needs to fit the picture, so their dimensions must be compatible.'),
('Can you explain what a cache does in a computer in one short sentence?','A cache keeps reusable data nearby so it can often be accessed more quickly.'),
('What is a queue in programming, explained without code?','A queue stores items in order and usually handles the earliest added item first.'),
('How is a stack different from a queue in a simple programming example?','A stack handles the most recently added item first; a queue usually handles the earliest added item first.'),
('Why might a program put a repeated operation inside a function?','A function lets the program reuse that operation without copying the same instructions everywhere.'),
('What is a variable in a computer program?','It is a name associated with a value that the program can use.'),
('What does a unit test usually check?','It checks a small part of a program against an expected result or behaviour.'),
('How is finding a bug different from fixing it?','Finding a bug identifies a problem. Fixing it changes the program to address that problem.'),
('What is a software dependency?','It is another component or package that the software needs in order to work.'),
('What does a database index help with?','It can help the database find relevant records without checking every record in turn.'),
('Why might a database use a transaction for several related changes?','A transaction groups the changes so they can succeed together or be rolled back together.'),
('What does a backup provide that the only working copy does not?','It provides another copy that may help restore the data if the working copy is lost or damaged.'),
('What is the difference between a file name and the contents of a file?','The name helps identify the file. The contents are the data stored inside it.'),
('How is network delay different from network bandwidth?','Delay describes how long communication takes; bandwidth describes how much data can be carried over time.'),
('What is the purpose of resolving a website name to a network address?','It helps the computer locate the server associated with that name.'),
('Why can retrying a failed network request sometimes help?','A temporary connection problem may have cleared, allowing a later attempt to succeed.'),
('Why should a program avoid retrying a permanent error forever?','Repeating the same request will not fix a permanent problem and can waste resources.'),
('How would you explain an overfit model to a beginner?','It has learned its training examples too closely and may perform poorly on new examples.'),
('Does a low training loss prove that a model behaves well on new examples?','No. New examples need separate evaluation because training performance does not guarantee generalisation.'),
('What does a loss function do during machine learning?','It measures an aspect of prediction error so training can work towards reducing it.'),
('What does gradient accumulation combine before an optimizer update?','It combines gradients from several smaller batches before making one update.'),
('What does setting a random seed help with when running an experiment?','It helps make random choices repeatable, although other computations may still be nondeterministic.'),
('Why keep evaluation examples separate from examples used for training?','They help measure how the model handles examples it has not been trained on.'),
('Why is it useful to save optimizer state as well as model weights?','The optimizer has its own history, which is needed to continue the same training process accurately.'),
('What is the difference between a model answering correctly and its speech being clear?','One concerns the meaning of the answer; the other concerns how well the spoken audio conveys it.'),
]
KEYS='lantern cushion coaster hamper dresser rack crate pouch packet tray bench curtain mantel pantry counter locker caddy satchel case bin jug bowl apron tent'.split()
COLOURS='amber purple blue green ivory orange silver yellow'.split()
SHAPES='triangle stripe diamond square oval circle star crescent'.split()

def build():
    scenes=[]
    def add(family,turns,program,world=None):
        index=len(scenes);scenes.append(dict(id=f'venus-retention-request-v1-train-{family}-{index:04d}',fold='TRAIN',family=family,split_group=f'TRAIN:retention-request-v1:{family}:{index}',user_turns=turns,teacher_program=program,synthetic_private_world=world,source='authored_synthetic',owner_conversations=False,proposal_only=True))
    for question,answer in ORDINARY:add('ordinary',[question],dict(kind='ordinary_answer',answer=answer,delegate_forbidden=True))
    for i,key in enumerate(KEYS):
        value=COLOURS[i%8]+' '+SHAPES[(i//8+i)%8];query='lookup '+key;world={key:value}
        question=f'Please ask the background worker to retrieve the marker in the {key}. Tell me the confirmed result rather than guessing.'
        if i<12:add('delegate',[question],dict(kind='lookup',query=query,worker_delay_seconds=12,result_answer='The marker is '+value+'.',result_must_follow_receipt=True),world)
        else:
            follow,answer=ORDINARY[i]
            add('pending',[question,'While that task is running, '+follow],dict(kind='lookup_with_followup',query=query,worker_delay_seconds=35,followup_answer=answer,result_answer='The marker is '+value+'.',answer_followup_while_pending=True,result_must_follow_receipt=True),world)
    for i,key in enumerate(KEYS):
        follow,answer=ORDINARY[24+i];value=COLOURS[(i+3)%8]+' '+SHAPES[(i+2)%8]
        add('cancel',[f'Please start a background lookup for the marker in the {key}.','Cancel the pending task now.',follow],dict(kind='cancel_lookup',query='lookup '+key,cancel_query='cancel pending',worker_delay_seconds=35,cancel_acknowledgment='I have cancelled the pending task.',ordinary_answer=answer,no_cancelled_value_delivery=True),{key:value})
    assert len(ORDINARY)==48 and len(scenes)==96
    assert {f:sum(s['family']==f for s in scenes) for f in ('ordinary','delegate','pending','cancel')}==dict(ordinary=48,delegate=12,pending=12,cancel=24)
    return dict(schema='venus-control-source-proposal-v1',status='AUTHORED_PROPOSAL_NOT_AUDIO_APPROVED_NOT_TRAINED',scenes=scenes,counts=dict(TRAIN=96,DEV=0,HOLDOUT=0),original240TRAIN64DEV_unchanged=True,original_sources_and_receipts_unchanged=True,external_API_calls=0,optimizer_updates=0,heldout_answers_used=False,source_audio_generated=False,teacher_receipts_generated=False,native_cache_generated=False,approval_required_before_training=['Source semantic review','Applicable synthetic audio payload authorization','Complete source word/peak audit','Genuine worker receipts including pending cancellation','Native teacher/history alignment and original retention preservation','Versioned composite corpus approval','Independent complete-state restore and native evaluation'],purpose='MiRL-inspired supervised coverage/retention and valid-request repair, not native RL or evidence of success',fresh_evaluation='Separate fresh cohort still needs independent authorship/sealing; this file is TRAIN proposals only')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);assert not out.exists()
    data=build();out.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(dict(path=str(out),scenes=len(data['scenes']),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),status=data['status'])))
