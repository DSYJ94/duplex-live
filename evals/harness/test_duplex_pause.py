import json,tempfile,time,unittest
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import evals.duplex_hybrid_probe as probe
from evals.duplex_pause import unheard_packets,grouped_completions
from evals.audio_playout import render_packets
class Clock:
 def __init__(self):self.t=0.
 def perf_counter(self):return self.t
 def monotonic(self):return self.t
 def sleep(self,s):self.t+=max(s,.0001);time.sleep(0)
class Listener:
 class SPM:
  def decode(self,tokens):return ''.join({100:'Explain',101:' phases.',102:'Stop.',103:' Give one reason.'}[t] for t in tokens)
 spm=SPM()
 def __init__(self,lexical=False):self.lexical=lexical
 def reset(self):self.i=0
 def step(self,pcm):
  tokens={4:100,5:101,24:102} if self.lexical=='stop-only' else {4:100,5:101,24:102,25:103} if self.lexical else {4:100,5:101}
  token=tokens.get(self.i,3);self.i+=1
  return dict(token=token,heads=[[0.9],[0.9],[0.9]])
class Speaker:
 def __init__(self):self.active=False;self.starts=0;self.generated=[]
 def stop(self):self.active=False
 def start(self,text):self.starts+=1;self.offset=0;self.active=True
 def step(self):
  p=(np.arange(1920,dtype=np.float32)+self.offset*1920)/1000000;self.offset+=1
  self.active=self.offset<75
  self.generated.append(p.copy());return dict(pcm=[p],compute_s=0.,done=not self.active)
class Controller:
 model_id='fixture-only';provider_name='fixture-only'
def worker(context,world,task,**kwargs):return dict(route='answer',content='The lit portion of the Moon changes as it orbits Earth.',consume_pending_speech_ids=[p['id'] for p in kwargs['pending_speech']] if len(context)>1 else [],receipts=[])

class ExactSpeechPause(unittest.TestCase):
 def run_probe(self,policy,lexical=False,twice=False,audit=False,silent=False):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);user=np.zeros(100*1920,dtype=np.float32)
   user[:3*1920]=.1;user[20*1920:23*1920]=.1
   if twice:user[50*1920:53*1920]=.1
   wav=p/'input.wav';sf.write(wav,user,24000,subtype='FLOAT');ref=p/'reference.json';ref.write_text(json.dumps(dict(user_words=[dict(text='Explain phases. Uh huh.')])) )
   speaker=Speaker();clock=Clock();row=dict(scenario_id='case',split='fresh-evaluation',path=str(wav),reference_path=str(ref),wav_sha256=probe.sha256(wav),world={})
   def execute(*args,**kwargs):
    result=worker(*args,**kwargs)
    if silent and len(args[0])>1:result.update(route='silence',content='')
    return result
   with patch.object(probe,'time',clock),patch.object(probe,'run_worker',execute):
    result=probe.run_case(Listener(lexical),speaker,row,p/'case',0,'state-aware',policy,Controller())
   if audit:
    from evals import duplex_hybrid_audit
    (p/'report.json').write_text(json.dumps(dict(schema='dl-concurrent-hybrid-development-v1',same_process_same_gpu=True,
      component_execution_completed=True,hybrid_system_qualified=False,results=[result])))
    # Exercise main's actual decode-result loop, including word timestamps, with
    # a CPU-only recognizer fixture. This is parser coverage, not speech scoring.
    recognizer=SimpleNamespace(load_model=lambda *a,**k:object(),transcribe=lambda *a,**k:dict(text='Fixture transcript',segments=[dict(start=0.,end=.001,words=[dict(start=0.,end=.001,word='Fixture')])]))
    with patch.dict(sys.modules,{'whisper_timestamped':recognizer}),patch.object(sys,'argv',['audit','--run',str(p),'--out',str(p/'audit')]),patch('builtins.print'):
     duplex_hybrid_audit.main()
    self.last_audit=json.loads((p/'audit/report.json').read_text())
   events=json.loads((p/'case/events.json').read_text());audio,_=sf.read(p/'case/model.wav',dtype='float32');heard=[]
   for group in result['playout']['events'].values():
    for packet in group['packets']:
     first=packet['playback_start_sample'];last=first+packet['samples']
     if group['actual_stop_s'] is not None:last=min(last,int(np.ceil(group['actual_stop_s']*24000)))
     if last>first:heard.append((first,audio[first:last]))
   heard=np.concatenate([part for _,part in sorted(heard)])
   return speaker,result,events,heard
 def test_preserved_default_reproduces_same_answer_restart_without_new_words(self):
  speaker,result,events,heard=self.run_probe('acoustic')
  self.assertEqual(speaker.starts,2);self.assertEqual(sum(e['kind']=='controller_dispatched' for e in events),1)
 def test_pause_stops_promptly_and_resumes_every_sample_once_without_new_generation(self):
  speaker,result,events,heard=self.run_probe('acoustic-pause')
  self.assertEqual(speaker.starts,1)
  self.assertEqual(sum(e['kind']=='speech_paused' for e in events),1)
  self.assertEqual(sum(e['kind']=='speech_resumed' for e in events),1)
  pause=next(e for e in events if e['kind']=='speech_paused');start=max(e['at_s'] for e in events if e['kind']=='user_speech_started' and e['at_s']<=pause['at_s'])
  self.assertLessEqual(pause['at_s']-start,.001)
  self.assertTrue(np.array_equal(heard,np.concatenate(speaker.generated)))
  self.assertEqual(len(result['fully_played_responses']),1)
  self.assertEqual(len(result['fully_played_responses'][0]['playback_event_ids']),2)
 def test_repeated_nonlexical_sounds_preserve_one_exact_generation(self):
  speaker,result,events,heard=self.run_probe('acoustic-pause',twice=True)
  self.assertEqual(speaker.starts,1);self.assertEqual(sum(e['kind']=='speech_resumed' for e in events),2)
  self.assertTrue(np.array_equal(heard,np.concatenate(speaker.generated)))
 def test_actual_new_words_commit_interruption_and_resolve_old_obligation(self):
  speaker,result,events,heard=self.run_probe('acoustic-pause',lexical=True)
  self.assertEqual(speaker.starts,2)
  self.assertEqual(sum(e['kind']=='speech_pause_committed' for e in events),1)
  self.assertFalse(any(e['kind']=='speech_resumed' for e in events))
  self.assertEqual([r['task_id'] for r in result['fully_played_responses']],['turn-2'])
  self.assertTrue(any(e['kind']=='speech_obligation_consumed' and e['task_id']=='turn-1' for e in events))
 def test_unheard_tail_matches_independent_playout_with_gaps_and_partial_packets(self):
  values=[np.arange(1,18,dtype=np.float32),np.arange(18,39,dtype=np.float32),np.arange(39,57,dtype=np.float32)]
  packets=[dict(event_id='a',available_s=t,pcm=p) for t,p in zip((.01,.03,.2),values)]
  audio,ledger=render_packets([(p['available_s'],p['pcm']) for p in packets],1000,.02)
  for cut in (.0,.03,.037,.049,.07,.207,.3):
   heard=[]
   for row in ledger['packets']:
    first=row['playback_start_sample'];last=min(first+row['samples'],int(np.ceil(cut*1000)))
    if last>first:heard.append(audio[first:last])
   tail=unheard_packets(packets,'a',cut,1000,.02)
   self.assertTrue(np.array_equal(np.concatenate(heard+tail),np.concatenate(values)))
 def test_full_pause_audit_requires_same_generation_and_excludes_other_answers(self):
  _,case,events,_=self.run_probe('acoustic-pause')
  starts={e['event_id']:e for e in events if e['kind']=='speech_started'}
  groups=grouped_completions(case,case['playout'],starts)
  self.assertEqual(len(groups),1);self.assertTrue(groups[0]['intentional_pause_included'])
  second=case['fully_played_responses'][0]['playback_event_ids'][-1]
  starts[second]['same_generator']=False
  with self.assertRaises(ValueError):grouped_completions(case,case['playout'],starts)
  starts[second]['same_generator']=True
  case['playout']['events']['other']=dict(actual_stop_s=None,packets=[dict(playback_start_sample=int(2.*24000),samples=100)])
  with self.assertRaises(ValueError):grouped_completions(case,case['playout'],starts)
 def test_actual_audit_parser_keeps_work_item_after_word_timestamp_loop(self):
  self.run_probe('acoustic-pause',audit=True)
  rows=self.last_audit['results'];full=[r for r in rows if r['fully_played']]
  self.assertEqual(len(full),1);self.assertTrue(full[0]['pause_aggregate'])
  self.assertTrue(all(r['word_error_rate'] is None for r in rows if not r['fully_played']))
 def test_silent_withdrawal_does_not_start_another_speech_generation(self):
  speaker,result,events,heard=self.run_probe('acoustic-pause',lexical='stop-only',silent=True)
  self.assertEqual(speaker.starts,1)
  self.assertEqual(result['fully_played_responses'],[])
  self.assertTrue(any(e['kind']=='silent_control_completed' and e['speech_ids']==['turn-1'] for e in events))

if __name__=='__main__':unittest.main()
