"""Authored assistant-voice source material, separate from control teachers.

These are synthesis candidates, not approved acoustic training examples. User
context and delivery notes are never sent as spoken text. Empty targets are
deliberate silence; no acknowledgement is synthesized for them.
"""
import argparse,hashlib,json
from pathlib import Path

# Six unique scripts per family: first four TRAIN, fifth DEV, sixth HOLDOUT.
# Shared prosodic families are intentional; script text stays split-disjoint.
FAMILIES={
 'warm_conversation':[
 ('I finally got the little garden planted.','That sounds lovely. What did you decide to grow?'),
 ('My first loaf of bread actually worked.','Oh, that is a good feeling. How did the crust turn out?'),
 ('We found a quiet place by the river.','That sounds like a nice spot to stop for a while.'),
 ('I have a free afternoon for once.','Nice. Would you rather get outside or have a quiet afternoon in?'),
 ('The new puppy has finally settled down.','That must be a relief. Has it found a favourite sleeping spot?'),
 ('I saw the moon through a telescope tonight.','That sounds wonderful. Could you make out the craters?')],
 'explanation_and_emphasis':[
 ('Why does a metal spoon feel colder than wood?','The metal carries heat away from your hand faster. Both can be at the same temperature.'),
 ('Why is the sky blue?','Air scatters blue light more strongly than red light, so blue light reaches us from across the sky.'),
 ('Why does a bicycle stay balanced more easily when moving?','Moving lets you steer the wheels back underneath you. Those small corrections help keep you balanced.'),
 ('Why does salt help melt ice?','Salt lowers the freezing point of water. That lets some ice melt even below the usual freezing temperature.'),
 ('Why do leaves change colour?','As the green pigment breaks down, other colours become easier to see.'),
 ('Why do I hear an echo in a large empty room?','Sound reflects off the walls. With fewer soft surfaces to absorb it, those reflections are easier to hear.')],
 'contrastive_focus':[
 ('Did you mean the blue folder or the green folder?','The green folder. The blue one is the older copy.'),
 ('Should I stir before or after adding the flour?','Add the flour first, then stir gently.'),
 ('Is that a deadline for starting or finishing?','Finishing. The work needs to be complete by then.'),
 ('Do I turn left at this corner?','At the next corner. Keep going past this one.'),
 ('Was the appointment moved to Tuesday?','Wednesday, actually. Tuesday was the original date.'),
 ('Do I need a deep bowl?','A wide bowl matters more than a deep one for this step.')],
 'curious_questions':[
 ('I want to try something new this weekend.','What sort of thing sounds appealing: something active, creative, or relaxing?'),
 ('This recipe did not turn out right.','What happened? Was it too dry, too dense, or something else?'),
 ('I am thinking about learning an instrument.','Do you already have an instrument in mind?'),
 ('I need a better way to organise my notes.','How do you usually use them: to study, plan, or find information later?'),
 ('I would like to go walking somewhere different.','Would you prefer a short easy walk or a longer route?'),
 ('I want to make the room feel more comfortable.','What would you like to change first: the lighting, the layout, or the colours?')],
 'honest_uncertainty':[
 ('Will it rain here this evening?','I would need a current local forecast to answer that reliably.'),
 ('Is the small cafe still open on Sundays?','I am not sure of its current hours. I can check if you tell me which cafe.'),
 ('Can you tell what made that sound?','I cannot identify it confidently from that description. What did it sound like?'),
 ('Will this plant grow well on my balcony?','It depends on how much direct sunlight the balcony gets. Roughly how many hours does it have?'),
 ('Is this train running late?','I would need the live departure information to confirm that.'),
 ('Is this the best route home right now?','That depends on the current traffic. I can check the route if you give me the destination.')],
 'self_correction':[
 ('You said five, but the note says six.','You are right. It should be six. Thanks for catching that.'),
 ('I meant the smaller box.','Ah, the smaller one. I misunderstood which box you meant.'),
 ('That was my sister, not my brother.','Your sister. I have corrected that.'),
 ('I asked for a short explanation.','Fair point. Here is the short version: the two surfaces exchange heat.'),
 ('I said fifteen, not fifty.','Fifteen. Thanks for clarifying.'),
 ('We are meeting outside, not inside.','Outside. I understand now.')],
 'allowed_backchannel':[
 ('I am going to tell the whole story. A short acknowledgement is fine.','Mm-hm.'),
 ('You can give a brief acknowledgement while I explain what happened.','Right.'),
 ('Let me talk this through. You can acknowledge briefly.','I see.'),
 ('Please listen while I explain. A quick acknowledgement is welcome.','Uh-huh.'),
 ('I will describe the situation. A little acknowledgement is okay.','Mm-hm, go on.'),
 ('Let me finish the story. A brief acknowledgement is fine.','Okay, I am listening.')],
 'deliberate_silence':[
 ('Keep quiet until I say marigold. Do not acknowledge this instruction.',''),
 ('Listen without speaking until I say copper. No verbal reply yet.',''),
 ('I am thinking aloud. Please stay completely silent for now.',''),
 ('Do not make any sound until I ask you to continue.',''),
 ('Keep listening silently. I have not given the release word yet.',''),
 ('Remain silent until I say juniper, including no acknowledgements.','')],
 'calm_pacing':[
 ('Give me a simple way to start tidying my desk.','Start with one small area. Clear the loose papers, put away what belongs elsewhere, and then stop for a moment.'),
 ('How do I make tea without rushing?','Warm the cup, add the tea, and pour in the hot water. Give it a little time to steep before you taste it.'),
 ('What should I pack first for a short walk?','Take some water and a light layer. Then check your shoes and choose a route that suits the time you have.'),
 ('How can I start writing this note?','Begin with the one thing you want the reader to know. You can tidy the wording after that.'),
 ('How should I approach this small repair?','First, look closely at what is loose. Gather the right tools before you start taking anything apart.'),
 ('How do I begin sorting a shelf of books?','Take a small section first. Decide what stays, what moves, and what you no longer need.')],
 'numbers_and_pronunciation':[
 ('Read this time naturally: half past eight.','Half past eight.'),
 ('Say the amount three pounds and fifty pence.','Three pounds and fifty pence.'),
 ('Read the temperature: eighteen degrees Celsius.','Eighteen degrees Celsius.'),
 ('Say the distance two and a half kilometres.','Two and a half kilometres.'),
 ('Read this date: the twenty third of April.','The twenty third of April.'),
 ('Say the quantity one hundred and twenty four.','One hundred and twenty four.')],
 'pending_work':[
 ('You are already checking the record. Let me know briefly that it is still running.','The check is still running. You can keep talking while I wait for the result.'),
 ('The longer calculation is still running. Can we keep chatting?','Yes. The calculation can keep running in the background.'),
 ('You have already started the search. What happens now?','I am waiting for the search to finish. I will bring the result back when it arrives.'),
 ('The comparison has started. Please give me a short update.','The comparison is in progress. I do not have the result yet.'),
 ('You have already asked the worker. Can I ask something else?','Of course. The background task can continue while we talk.'),
 ('The lookup is running. Please do not guess the answer.','I will wait for the actual result.')],
 'resume_after_interruption':[
 ('Continue the explanation about packing, but keep it short.','As I was saying, keep the heavier items close to your back. That usually makes the bag more comfortable.'),
 ('Carry on from the point about the soil.','The next step is to check whether the soil feels dry below the surface.'),
 ('Resume the recipe after the chopping step.','Once the vegetables are chopped, warm the pan and add a little oil.'),
 ('Continue with the next instruction.','Next, fold the paper along the centre line and press the crease gently.'),
 ('Pick up the explanation about shadows.','The shadow becomes longer when the light reaches the object from a lower angle.'),
 ('Continue your explanation about the route.','After the bridge, follow the path beside the river until you reach the gate.')]
}

def build():
    rows=[]
    for family,scripts in FAMILIES.items():
        assert len(scripts)==6
        for i,(context,text) in enumerate(scripts):
            rows.append(dict(id=f'voice-v1-{family}-{i+1:02d}',split='TRAIN' if i<4 else 'DEV' if i==4 else 'HOLDOUT',family=family,
                user_context=context,assistant_text=text,deliberate_silence=not text,
                permitted_acknowledgement=family=='allowed_backchannel',
                delivery='Conversational and clear; natural phrase breaks and appropriate emphasis. No theatrical delivery or added words.',
                source='authored-synthetic-20261027',training_approved=False))
    assert len(rows)==72 and len({r['assistant_text'] for r in rows if r['assistant_text']})==66
    return dict(schema='dl-native-voice-quality-source-v1',status='PREPARED_UNQUALIFIED',cases=rows,
        split_counts={s:sum(r['split']==s for r in rows) for s in ['TRAIN','DEV','HOLDOUT']},
        synthesis_policy=dict(provider='ElevenLabs',provenance_class='closed-content',authority='decisions/0003-provenance-classes.md#amendment-a-2026-09-30-closed-content-cleared-generally',
            payload='Only exact assistant_text; context, split, expected silence and scoring annotations excluded',empty_targets_never_synthesized=True,
            no_real_person_cloning=True,no_effects_or_gain_repair=True,maximum_characters=12000),
        required_approvals=['complete word/peak/source audit','assistant prosody listening review','native alignment and causal teacher checks before acoustic optimizer use'],
        reference_text='There is no need to rush. We can take this one step at a time. Tell me what you have in mind, and I will help you work through it. If something needs a closer look, I can check it in the background while we keep talking.',
        role='Assistant acoustic targets and voice evaluation, not a substitute for native control-policy trajectories',
        optimizer_updates=0,qualified=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    assert not out.exists(),'Version new artifacts instead of overwriting';out.write_text(json.dumps(build(),indent=2)+'\n')
