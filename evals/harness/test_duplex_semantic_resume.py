"""CPU contract/media fixtures, never neural routing or device acceptance."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import evals.duplex_hybrid_probe as probe
from evals.duplex_reasoning_worker import ReasoningWorld,run_reasoning_worker
from evals.duplex_resume import validate_target,require_retained
from evals.recovery_worker import WorkerContractError
from evals.harness.test_duplex_pause import Clock,Listener,Speaker,Controller


class SemanticResume(unittest.TestCase):
 def test_worker_offers_only_actual_retained_identity_and_no_fake_receipt(self):
  pending=[dict(id='old',text='Generated answer',request_text='Explain')];target=dict(speech_id='old',event_id='speech-3')
  class Action:
   provider_name='fixture-only'
   def __init__(self,arguments):self.arguments=arguments
   def __call__(self,payload):
    self.payload=payload
    return dict(id='fixture',status='completed',model='fixture-only',output=[dict(type='function_call',name='resume_speech',arguments=json.dumps(self.arguments))])
  for offered,args,passed in [(target,target,True),(None,target,False),(target,dict(speech_id='old',event_id='speech-2'),False),(target,dict(target,content='New words'),False)]:
   world=ReasoningWorld({});revision=world.reserve('current');executor=Action(args)
   def run():return run_reasoning_worker([dict(role='user',text='Please continue.')],world,'current','fixture-only',revision,'state-aware',pending,executor,resumable_speech=offered)
   if passed:
    result=run();self.assertEqual(result['route'],'resume');self.assertEqual(result['resume_target'],target)
    self.assertEqual(result['content'],'');self.assertEqual(result['consume_pending_speech_ids'],[])
    self.assertIn('resume_speech',[x['name'] for x in executor.payload['tools']])
   else:
    with self.assertRaises(WorkerContractError):run()
   self.assertEqual(world.receipts,[])
  with self.assertRaises(WorkerContractError):validate_target(dict(speech_id='missing',event_id='speech-3'),pending)
  active=dict(task_id='old',event_id='speech-4',result=dict(route='answer'))
  with self.assertRaises(WorkerContractError):require_retained(target,active,dict(at_s=1))

 def media(self,resume):
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d);user=np.zeros(100*1920,dtype=np.float32);user[:3*1920]=.1;user[20*1920:23*1920]=.1
   wav=folder/'input.wav';sf.write(wav,user,24000,subtype='FLOAT');ref=folder/'reference.json';ref.write_text(json.dumps(dict(user_words=[dict(text='Explain phases. Continue.')])) )
   speaker=Speaker();clock=Clock();seen=[]
   class LexicalListener(Listener):
    class SPM:
     def decode(self,tokens):return ''.join({100:'Explain',101:' phases.',102:'Continue.'}[t] for t in tokens)
    spm=SPM()
   def execute(context,world,task,**kwargs):
    target=kwargs.get('resumable_speech');seen.append(target)
    if target is not None and resume:return dict(route='resume',content='',consume_pending_speech_ids=[],receipts=[],resume_target=target)
    return dict(route='answer',content='A new short answer.' if target else 'The lit portion of the Moon changes as it orbits Earth.',consume_pending_speech_ids=[target['speech_id']] if target else [],receipts=[])
   row=dict(scenario_id='fixture',split='fresh-evaluation',path=str(wav),reference_path=str(ref),wav_sha256=probe.sha256(wav),world={})
   with patch.object(probe,'time',clock),patch('evals.duplex_reasoning_worker.run_reasoning_worker',execute):
    result=probe.run_case(LexicalListener('stop-only'),speaker,row,folder/'case',0,'state-aware','acoustic-pause',Controller(),worker_kind='reasoning-development',include_speech_context=True,semantic_pause=True)
   events=json.loads((folder/'case/events.json').read_text());audio,_=sf.read(folder/'case/model.wav',dtype='float32');heard=[]
   for group in result['playout']['events'].values():
    for packet in group['packets']:
     first=packet['playback_start_sample'];last=first+packet['samples']
     if group['actual_stop_s'] is not None:last=min(last,int(np.ceil(group['actual_stop_s']*24000)))
     if last>first:heard.append((first,audio[first:last]))
   return speaker,result,events,np.concatenate([a for _,a in sorted(heard)]),seen

 def test_lexical_resume_keeps_same_generator_and_every_source_sample_once(self):
  speaker,result,events,heard,seen=self.media(True)
  self.assertEqual(speaker.starts,1);self.assertEqual(seen,[None,dict(speech_id='turn-1',event_id='speech-1')])
  self.assertEqual(sum(e['kind']=='semantic_resume_selected' for e in events),1)
  self.assertEqual(sum(e['kind']=='speech_resumed' for e in events),1)
  self.assertFalse(any(e['kind']=='speech_pause_committed' for e in events))
  self.assertTrue(np.array_equal(heard,np.concatenate(speaker.generated)))
  self.assertEqual(len(result['fully_played_responses']),1)

 def test_non_resume_decision_commits_pause_and_consumes_old_answer(self):
  speaker,result,events,heard,seen=self.media(False)
  self.assertEqual(speaker.starts,2)
  self.assertFalse(any(e['kind']=='speech_resumed' for e in events))
  self.assertEqual(sum(e['kind']=='speech_pause_committed' for e in events),1)
  self.assertEqual([x['task_id'] for x in result['fully_played_responses']],['turn-2'])
