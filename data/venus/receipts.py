"""Real read-only synthetic lookup work; shared by teacher collection and native serving."""
import argparse,asyncio,hashlib,json,re,time
from pathlib import Path
from portable.venus.native_app import ArithmeticWorker

def lookup_key(query):
 """Bounded emitted-tool grammar; never infer a missing key from the world."""
 if not re.match(r'^\s*look\s*up\b',query,re.I):return None
 match=re.fullmatch(r'\s*look\s*up\s+(?:(?:the\s+)?marker(?:\s+in)?\s+)?([a-z]+)[. ]*',query,re.I)
 if not match:raise ValueError('Lookup requires one explicit alphabetic key.')
 return match.group(1).lower()

class LookupWorker(ArithmeticWorker):
 def __init__(self,world,delay=8):super().__init__(delay);self.world=dict(world)
 async def execute(self,ident,query):
  try:key=lookup_key(query)
  except ValueError as error:
   self.event('worker_started',work_id=ident)
   receipt=dict(work_id=ident,query=query,result=str(error),error='lookup_requires_explicit_key')
   receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()
   self.completed.append(receipt);self.event('worker_completed',**receipt);return
  if key is None:return await super().execute(ident,query)
  self.event('worker_started',work_id=ident)
  try:
   await asyncio.sleep(self.delay)
   value=self.world.get(key)
   receipt=dict(work_id=ident,query=query,result=('Lookup '+key+': '+value) if value is not None else ('Entry '+key+' was not found.'),world_sha256=hashlib.sha256(json.dumps(self.world,sort_keys=True).encode()).hexdigest())
   receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()
   self.completed.append(receipt);self.event('worker_completed',**receipt)
  except asyncio.CancelledError:self.event('worker_cancelled',work_id=ident);raise

async def collect(scene,audio):
 program=scene['teacher_program'];w=LookupWorker(scene['synthetic_private_world'],program['worker_delay_seconds']);epoch=time.monotonic();w.submit(program['query'])
 if program['kind']=='cancel_lookup':
  delay=1+audio[scene['user_turns'][1]]['seconds']+1
  assert delay<program['worker_delay_seconds']-.5,'Cancel source utterance leaves no genuine pending interval'
  await asyncio.sleep(delay);w.submit(program['cancel_query'])
 await asyncio.gather(*w.tasks.values(),return_exceptions=True)
 payload=w.take_results();events=w.events
 value=dict(scene_id=scene['id'],teacher_forced_request=True,native_student_dispatch_claimed=False,epoch_monotonic=epoch,events=events,feedback=payload,feedback_available_after_seconds=max(e['monotonic'] for e in events)-epoch,world=scene['synthetic_private_world'])
 if program['kind']=='cancel_lookup':
  assert any(e['type']=='worker_cancelled' for e in events) and not any(e['type']=='worker_completed' for e in events)
 else:assert any(e['type']=='worker_completed' for e in events)
 await w.close();return value

async def main(a):
 plan=json.loads(Path(a.plan).read_text());ar=json.loads(Path(a.audio).read_text());assert ar['complete'];audio={r['text']:r for r in ar['utterances']}
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);semaphore=asyncio.Semaphore(8)
 async def one(s):
  async with semaphore:
   r=await collect(s,audio);(out/(s['id']+'.json')).write_text(json.dumps(r,indent=2)+'\n');return r
 selected=[s for s in plan['scenes'] if s['synthetic_private_world'] is not None]
 rows=await asyncio.gather(*(one(s) for s in selected))
 (out/'collection.json').write_text(json.dumps(dict(complete=len(rows)==len(selected),scenes=len(rows),actual_read_only_workers=True,model_inference=False),indent=2)+'\n')
 print(json.dumps(dict(receipts=len(rows),complete=True)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--audio',required=True);p.add_argument('--output',required=True);asyncio.run(main(p.parse_args()))
