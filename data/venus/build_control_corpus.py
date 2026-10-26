"""Authored synthetic native-control corpus; no owner data or external API."""
import argparse,hashlib,json,random
from pathlib import Path

ORDINARY=[
 ('What makes a shadow?','An object blocks light and creates a shadow.'),
 ('Why does a wet towel dry?','Liquid water evaporates into the surrounding air.'),
 ('What does a compass needle point towards?','It aligns with the Earth\'s magnetic field.'),
 ('Why does a ball fall when I let go?','Gravity pulls it towards the ground.'),
 ('What makes an echo?','Sound reflects from a surface and returns to your ears.'),
 ('Why does a mirror show my face?','Its smooth surface reflects light from your face.'),
 ('Why do we put lids on jars?','A lid closes the opening and helps keep the contents inside.'),
 ('What does a ruler measure?','It measures length or distance.'),
 ('What does a thermometer measure?','It measures temperature.'),
 ('What does a weighing scale measure?','It measures how heavy something is.'),
 ('Why does sugar disappear when stirred into water?','It dissolves and spreads through the water.'),
 ('Why does a magnet attract an iron nail?','Iron responds strongly to a magnetic field.'),
 ('What happens to water when it boils?','Liquid water turns into water vapour.'),
 ('Why does the Moon appear bright?','It reflects sunlight.'),
 ('What causes day and night?','The Earth rotates, bringing places into and out of sunlight.'),
 ('What does a seed need to start growing?','It needs suitable moisture, warmth and oxygen.'),
 ('Why does rubbing your hands warm them?','Friction converts movement into heat.'),
 ('What do wheels help a cart do?','They let it roll instead of sliding.'),
 ('What does a battery provide in a torch?','It provides electrical energy for the light.'),
 ('Why is a raincoat useful in rain?','Its material helps prevent water reaching your clothes.'),
 ('What is the opposite of narrow?','Wide.'),
 ('What is the opposite of noisy?','Quiet.'),
 ('What is two plus four?','Six.'),
 ('What is seven minus three?','Four.'),
 ('How many sides does a triangle have?','Three.'),
 ('How many sides does a square have?','Four.'),
 ('What shape has no corners and is round?','A circle.'),
 ('What do scissors do?','They cut materials between two blades.'),
 ('Why do we use an umbrella?','It provides shelter from rain or sunlight.'),
 ('What is a calendar used for?','It shows and helps organise dates.'),
 ('What is a map used for?','It shows locations and how places relate to each other.'),
 ('Why can a sponge absorb water?','Water enters the many small spaces inside it.'),
]
DEV_FACTS=[('What is a clock used for?','It tells the time.'),('How many legs does an insect have?','Six.'),('What do roots take up from soil?','Water and dissolved minerals.'),('Why does a door have hinges?','They allow the door to rotate open and closed.'),('What is the opposite of empty?','Full.'),('What is three plus four?','Seven.'),('Why does a boat need a rudder?','It helps steer the boat.'),('What is a bookmark for?','It marks a place in a book.')]
HOLD_FACTS=[('What does a telescope help us see?','Distant objects.'),('How many months are in a year?','Twelve.'),('What does a lever help us do?','It helps move a load using a pivot.'),('Why does a window have glass?','It lets light through while closing the opening.'),('What is the opposite of shallow?','Deep.'),('What is five minus two?','Three.'),('What is a funnel used for?','It guides material into a smaller opening.'),('What does a knot do to a rope?','It can fasten the rope or hold it in place.')]
STEPS=[('rinse the cup','dry it with a towel'),('fold the shirt','place it in the drawer'),('pick up the pencil','put it in the box'),('open the notebook','write the date'),('wash the spoon','leave it on the rack'),('lift the lid','place it on the mat')]
QUIET_PATTERNS=[
 'Stay silent until I say {cue}. Do not acknowledge this instruction. The word in this instruction is not a release.',
 'I need quiet while I think. Say nothing until a later message from me says {cue}. No acknowledgment now.',
 'Please listen without speaking. Wait for my later cue, {cue}. Mentioning the cue now does not release you.',
 'Do not make any spoken response until I later say {cue}. Keep listening, including through pauses.',
]
VALUES=['cobalt triangle','silver crescent','amber square','violet circle']

def build():
 scenes=[]
 for fold,cues,facts,steps,keys in [
  ('TRAIN',['marigold','cinnamon','saffron','compass','harbor','willow','tulip','velvet'],ORDINARY,STEPS,[x for x in ['pencil','teapot','blanket','basket','button','bottle','window','folder']]),
  ('DEV',['meadow','granite'],DEV_FACTS,[('roll the cloth','put it in the basket'),('take the lid off','set it beside the bowl')],['pillow','bucket']),
  ('HOLDOUT',['pebble','orchard'],HOLD_FACTS,[('stack the cards','put the stack in the tray'),('remove the cap from the pen','place the cap on the desk')],['jacket','cabinet']),
 ]:
  values=VALUES if fold=='TRAIN' else (['golden arrow','maroon star','ivory oval','teal diamond'] if fold=='DEV' else ['bronze star','mint hexagon','navy rectangle','coral ellipse'])
  patterns=QUIET_PATTERNS if fold=='TRAIN' else ([
   'Please make no spoken response until a later message from me contains {cue}. Wait silently now.',
   'I am going to think aloud. Keep completely quiet until I later use {cue} as your release signal.',
   'Hold your voice until I say {cue} in a future turn. This instruction does not release you.',
   'Listen in silence for now. The later release word is {cue}. Please do not reply to this instruction.'
  ] if fold=='DEV' else [
   'Wait without speaking until my later cue {cue}. Hearing it in this instruction is not the cue.',
   'For the next part, only listen. Resume speech after I later say {cue}. Stay silent now.',
   'Please suppress all acknowledgments and replies until a future utterance from me says {cue}.',
   'I need silent listening. Wait for {cue} in a later message before answering me.'
  ])
  def add(family,group,turns,program,world=None):
   ident=f'venus-control-v1-{fold.lower()}-{family}-{len(scenes):04d}'
   scenes.append(dict(id=ident,fold=fold,family=family,split_group=fold+':'+group,user_turns=turns,teacher_program=program,synthetic_private_world=world,source='authored_synthetic',owner_conversations=False))
  for cue in cues:
   for n,pattern in enumerate(patterns):
    add('quiet','cue-'+cue,[pattern.format(cue=cue),'I am still thinking. Keep waiting silently.',f'{cue}. '+facts[0][0]],dict(kind='quiet_until_later_cue',cue=cue,quiet_gap_seconds=4,release_answer=facts[0][1],no_acknowledgment=True))
  for pair in steps:
   for variant in range(4 if fold=='TRAIN' else 2):
    request=['Describe these steps slowly: {a}, then {b}.','Walk me through this slowly: {a}, and then {b}.','Give me the steps one at a time: {a}, then {b}.','Explain this sequence slowly: {a}, followed by {b}.'][variant].format(a=pair[0],b=pair[1])
    add('pause','steps-'+pair[0],[request,'Pause now. Stay silent until I say continue.','Continue.'],dict(kind='pause_resume',first_text='First, '+pair[0]+'.',remaining_text='Then, '+pair[1]+'.',quiet_gap_seconds=4,no_restart=True))
  for key in keys:
   for world in range(4):
    value=values[world];fixture={key:value}
    add('delegate','lookup-'+key,[f'Ask the background worker to look up the marker in {key}.'],dict(kind='lookup',query='lookup '+key,worker_delay_seconds=([8,10,12,14] if fold=='TRAIN' else [9,11,13,15] if fold=='DEV' else [7,12,16,18])[keys.index(key)%4],result_answer='The marker is '+value+'.',result_must_follow_receipt=True),fixture)
  for key in keys[:4 if fold=='TRAIN' else 1]:
   for world in range(4):
    value=values[world];fixture={key:value}
    add('pending','pending-'+key,[f'Ask the background worker to look up the marker in {key}.','While that runs, '+facts[2][0]],dict(kind='lookup_with_followup',query='lookup '+key,worker_delay_seconds=([12,14,16,18] if fold=='TRAIN' else [13,15,17,19] if fold=='DEV' else [11,16,20,22])[keys.index(key)%4],followup_answer=facts[2][1],result_answer='The marker is '+value+'.',answer_followup_while_pending=True,result_must_follow_receipt=True),fixture)
  for key in keys[:4 if fold=='TRAIN' else 1]:
   for world in range(4):
    fixture={key:values[world]}
    add('cancel','cancel-'+key,[f'Ask the background worker to look up the marker in {key}.','Cancel that task.',facts[1][0]],dict(kind='cancel_lookup',query='lookup '+key,cancel_query='cancel pending',worker_delay_seconds=([8,10,12,14] if fold=='TRAIN' else [9,11,13,15] if fold=='DEV' else [7,12,16,18])[keys.index(key)%4],cancel_acknowledgment='Cancelled.',ordinary_answer=facts[1][1],no_cancelled_value_delivery=True),fixture)
  for n,(question,answer) in enumerate(facts):
   add('ordinary','ordinary-'+str(n),[question],dict(kind='ordinary_answer',answer=answer,delegate_forbidden=True))
 random.Random(20261026).shuffle(scenes)
 counts={fold:{family:sum(s['fold']==fold and s['family']==family for s in scenes) for family in ['quiet','pause','delegate','pending','cancel','ordinary']} for fold in ['TRAIN','DEV','HOLDOUT']}
 assert sum(counts['TRAIN'].values())==152 and sum(counts['DEV'].values())==36 and sum(counts['HOLDOUT'].values())==36
 return dict(schema='dl-venus-native-control-corpus-v4',status='UNAPPROVED_PENDING_AUDIO_WORD_SEMANTIC_AND_NATIVE_CONVERSION',model_revision='4681004131ea1f1fb9aca9f0db76be322657cc79',seed=20261026,counts=counts,scenes=scenes,sealed_existing_project_evaluations_accessed=False,external_api_calls=0)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);assert not out.exists();plan=build();out.write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(dict(counts=plan['counts'],sha256=hashlib.sha256(out.read_bytes()).hexdigest(),scenes=len(plan['scenes']))))
if __name__=='__main__':main()
