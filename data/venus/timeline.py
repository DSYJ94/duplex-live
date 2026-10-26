"""Compile causal TRAIN teachers from authored programs, exact WAVs and real task receipts."""
import argparse,array,hashlib,json,math,wave
from pathlib import Path

def compile_scene(scene,audio,receipt):
 p=scene['teacher_program'];texts=scene['user_turns'];dur=[audio[t]['seconds'] for t in texts];starts=[0.0];actions=[];feedback=[]
 f0=math.ceil(dur[0])
 def say(frame,text,final=True):actions.append(dict(frame=frame,kind='speak',text=text,final=final))
 if p['kind']=='ordinary_answer':say(f0,p['answer']);end=f0+8
 elif p['kind']=='quiet_until_later_cue':
  starts.extend([f0+4,math.ceil(f0+4+dur[1])+4]);release=math.ceil(starts[2]+dur[2]);say(release,p['release_answer']);end=release+8
 elif p['kind']=='pause_resume':
  say(f0,p['first_text'],False);starts.append(f0+1);pause_frame=f0+1
  actions.append(dict(frame=pause_frame,kind='end_turn',text='',final=True))
  starts.append(math.ceil(starts[1]+dur[1])+4);resume=math.ceil(starts[2]+dur[2]);say(resume,p['remaining_text']);end=resume+8
 else:
  assert receipt is not None
  actions.append(dict(frame=f0,kind='delegate',text=p['query'],final=True))
  if p['kind']=='lookup':
   available=f0+math.ceil(receipt['feedback_available_after_seconds']);feedback.append(dict(frame=available,text=receipt['feedback']));say(available,p['result_answer']);end=available+8
  elif p['kind']=='lookup_with_followup':
   starts.append(f0+1);answer=math.ceil(starts[1]+dur[1]);say(answer,p['followup_answer'])
   available=f0+math.ceil(receipt['feedback_available_after_seconds']);assert answer+2<available,'Follow-up must fit while work is genuinely pending'
   feedback.append(dict(frame=available,text=receipt['feedback']));say(available,p['result_answer']);end=available+8
  elif p['kind']=='cancel_lookup':
   starts.append(f0+1);cancel=math.ceil(starts[1]+dur[1]);actions.append(dict(frame=cancel,kind='delegate',text=p['cancel_query'],final=True))
   available=f0+math.ceil(receipt['feedback_available_after_seconds']);assert available>=cancel
   feedback.append(dict(frame=available,text=receipt['feedback']));say(available,p['cancel_acknowledgment'])
   starts.append(max(available+2,math.ceil(starts[1]+dur[1])+2));answer=math.ceil(starts[2]+dur[2]);say(answer,p['ordinary_answer']);end=answer+8
  else:raise ValueError('Unknown native teacher program')
 assert len(starts)==len(texts) and all(starts[i+1]>=starts[i]+dur[i] for i in range(len(starts)-1))
 return dict(scene_id=scene['id'],fold=scene['fold'],family=scene['family'],user_turns=[dict(text=t,start_seconds=s,audio_id=audio[t]['id'],sha256=audio[t]['sha256'],seconds=d) for t,s,d in zip(texts,starts,dur)],actions=sorted(actions,key=lambda a:a['frame']),backend_feedback=feedback,frames=end,teacher_forced=True,serving_oracle_dispatch=False,training_approved=False)

def main(a):
 source=Path(a.plan);plan=json.loads(source.read_text());audio_root=Path(a.audio_root);ar=json.loads((audio_root/'render-receipts.json').read_text());assert ar['complete'];audio={r['text']:r for r in ar['utterances']}
 assert set(audio)=={t for s in plan['scenes'] for t in s['user_turns']},'Only exact unchanged USER texts may reuse audio'
 for r in ar['utterances']:
  assert r['seconds']>0 and r['peak']<=.995,'Empty or clipped source is never usable silence'
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);manifest=[]
 for scene in plan['scenes']:
  receipt=json.loads((Path(a.receipts)/(scene['id']+'.json')).read_text()) if scene['synthetic_private_world'] else None
  timeline=compile_scene(scene,audio,receipt);pcm=array.array('h',[0])*(timeline['frames']*16000)
  for turn in timeline['user_turns']:
   raw=audio_root/(turn['audio_id']+'.wav');assert hashlib.sha256(raw.read_bytes()).hexdigest()==turn['sha256']
   with wave.open(str(raw)) as w:assert w.getnframes()>0;values=array.array('h',w.readframes(w.getnframes()))
   start=round(turn['start_seconds']*16000);assert not any(pcm[start:start+len(values)]);pcm[start:start+len(values)]=values
  folder=out/scene['id'];folder.mkdir()
  with wave.open(str(folder/'user.wav'),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(pcm.tobytes())
  timeline['user_wav_sha256']=hashlib.sha256((folder/'user.wav').read_bytes()).hexdigest();(folder/'timeline.json').write_text(json.dumps(timeline,indent=2)+'\n')
  if receipt:(folder/'real-worker.json').write_text(json.dumps(receipt,indent=2)+'\n')
  manifest.append(dict(id=scene['id'],fold=scene['fold'],family=scene['family'],frames=timeline['frames'],timeline_sha256=hashlib.sha256((folder/'timeline.json').read_bytes()).hexdigest(),user_wav_sha256=timeline['user_wav_sha256']))
 (out/'manifest.json').write_text(json.dumps(dict(schema='dl-venus-native-control-timelines-v1',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),audio_original_plan_sha256=ar['plan_sha256'],audio_receipts_sha256=hashlib.sha256((audio_root/'render-receipts.json').read_bytes()).hexdigest(),exact_audio_text_set_matches=True,scenes=manifest,training_approved=False,independent_word_review='PENDING',independent_semantic_review='PENDING'),indent=2)+'\n');print(json.dumps(dict(scenes=len(manifest),frames=sum(x['frames'] for x in manifest))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--audio-root',required=True);p.add_argument('--receipts',required=True);p.add_argument('--output',required=True);main(p.parse_args())
