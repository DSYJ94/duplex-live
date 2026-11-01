import json,unittest
from portable.venus.context_reasoning import EvidenceTransport,contextual_payload,validate_contextual_payload
from portable.venus.reasoning import payload,validate_payload
from portable.venus.task_context import AudioEvidence

class ContextTests(unittest.IsolatedAsyncioTestCase):
 def recognition(self,e):return dict(text='Compare loan terms at eight percent.',model='fixture-asr',revision='fixture-revision',audio_receipt=e.receipt())
 def test_explicit_general_profile_does_not_weaken_legacy(self):
  e=AudioEvidence(0,0,bytes(32000));v=contextual_payload(payload('Compare loan terms',{}),self.recognition(e));validate_contextual_payload(v)
  with self.assertRaises(ValueError):validate_payload(v)
  for bad in (dict(v,model='other'),dict(v,tools=[]),dict(v,store=True)):
   with self.assertRaises(ValueError):validate_contextual_payload(bad)
 async def test_wrong_audio_and_added_answer_fields_never_reach_provider(self):
  e=AudioEvidence(0,0,bytes(32000));calls=[]
  async def send(v):calls.append(v)
  async def wrong(e):return self.recognition(AudioEvidence(1,1,bytes(32000)))
  with self.assertRaises(ValueError):await EvidenceTransport(wrong,send)(payload('task',{}),e)
  async def answer(e):return dict(self.recognition(e),expected_answer='forbidden scoring field')
  with self.assertRaises(ValueError):await EvidenceTransport(answer,send)(payload('task',{}),e)
  self.assertEqual(calls,[])
 async def test_active_identities_and_unverified_source_are_preserved(self):
  e=AudioEvidence(2,2,bytes(32000));active={'travel':dict(query='Compare rail and coach',revision=1),'meal':dict(query='Compare meal costs',revision=1)};sent=[]
  async def transcribe(e):return self.recognition(e)
  async def send(v):sent.append(v);return dict(actual_transport=False,response={'fixture':True})
  reply=await EvidenceTransport(transcribe,send)(payload('Cancel only travel',active),e)
  self.assertFalse(reply['actual_transport']);v=sent[0];validate_contextual_payload(v)
  source=json.loads(v['input']);self.assertEqual(source['active_requests'],active)
  cancel=next(t for t in v['tools'] if t['name']=='cancel_reasoning')
  self.assertEqual(cancel['parameters']['properties']['task_ids']['items']['enum'],['meal','travel'])
  self.assertIn('not verified human testimony',source['recognized_audio']['provenance'])

 def test_legacy_active_request_order_is_stable_after_json_roundtrip(self):
  active={'travel':dict(query='travel',revision=1),'meal':dict(query='meal',revision=1)}
  v=payload('cancel travel',active)
  validate_payload(v);validate_payload(json.loads(json.dumps(v)))

 async def test_real_worker_preserves_unrelated_task_and_suppresses_cancelled_late_reply(self):
  import asyncio,itertools
  from portable.venus.reasoning import ReasoningWorker,MODEL
  e=AudioEvidence(0,0,bytes(32000));release={q:asyncio.Event() for q in ('loan','meal')};started={q:asyncio.Event() for q in release};ids=itertools.count()
  async def transcribe(e):return self.recognition(e)
  async def send(value):
   validate_contextual_payload(value);source=json.loads(value['input']);q=source['native_selected_task']
   operation='respond_to_user';args=dict(content='Fixture result for '+q)
   if q in release:started[q].set();await release[q].wait()
   else:
    operation='cancel_reasoning';args=dict(content='Retired loan delivery.',task_ids=[k for k,v in source['active_requests'].items() if v['query']=='loan'])
   return dict(actual_transport=False,response=dict(id='fixture-'+str(next(ids)),model=MODEL,status='completed',output=[dict(type='function_call',name=operation,arguments=json.dumps(args))]))
  w=ReasoningWorker(None,evidence_request=EvidenceTransport(transcribe,send))
  a=w.submit('loan',audio_evidence=e);b=w.submit('meal',audio_evidence=e)
  await started['loan'].wait();await started['meal'].wait()
  cancel=w.submit('cancel loan',audio_evidence=e);await w.tasks[cancel]
  self.assertTrue(w.states[a]['cancelled']);self.assertFalse(w.states[b]['cancelled'])
  ack=w.take_results();self.assertIn('Retired loan delivery',ack);self.assertIn('"remote_execution_aborted": false',ack)
  release['loan'].set();release['meal'].set();await asyncio.gather(w.tasks[a],w.tasks[b])
  result=w.take_results();self.assertIn('Fixture result for meal',result);self.assertNotIn('Fixture result for loan',result);self.assertIsNone(w.take_results())
  self.assertTrue(all('audio_evidence' not in s for s in w.states.values()));await w.close()

 async def test_actual_context_http_proxy_and_receipt_bind_posted_payload(self):
  import hashlib,tempfile,threading,time
  from portable.context.local_responses_proxy import ProxyServer
  from portable.venus.context_reasoning import ContextProxy
  from portable.venus.reasoning import ReasoningWorker,MODEL
  seen=[]
  def post(url,headers,json,timeout,allow_redirects):
   seen.append(json)
   value=dict(id='fixture-context-http',model=MODEL,status='completed',output=[dict(type='function_call',name='respond_to_user',arguments=__import__('json').dumps(dict(content='Fixture solved answer.')))])
   class Reply:
    status_code=200;content=__import__('json').dumps(value).encode();headers={'x-request-id':'fixture-context'}
    def json(self):return value
   return Reply()
  with tempfile.TemporaryDirectory() as directory:
   from pathlib import Path
   server=ProxyServer(0,'fixture-only-key','context-http',Path(directory),1,time.time()+20,post=post,profile='venus-context-reasoning')
   thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
   try:
    e=AudioEvidence(0,0,bytes(32000))
    async def transcribe(e):return self.recognition(e)
    proxy=ContextProxy('http://127.0.0.1:'+str(server.server_address[1])+'/v1/responses','context-http')
    worker=ReasoningWorker(None,evidence_request=EvidenceTransport(transcribe,proxy));ident=worker.submit('Compare loan terms',audio_evidence=e);await worker.tasks[ident]
    receipt=worker.states[ident]['receipt'];self.assertFalse(receipt['actual_transport'])
    self.assertEqual(receipt['request_payload_sha256'],hashlib.sha256(json.dumps(seen[0],sort_keys=True).encode()).hexdigest())
    self.assertNotEqual(receipt['request_payload_sha256'],receipt['native_task_payload_sha256']);await worker.close()
   finally:server.shutdown();server.server_close();thread.join(3)

 async def test_disconnect_during_transcription_prevents_new_provider_dispatch(self):
  import asyncio
  from portable.venus.reasoning import ReasoningWorker
  started=asyncio.Event();release=asyncio.Event();sent=[]
  async def transcribe(e):started.set();await release.wait();return self.recognition(e)
  async def send(value):sent.append(value);raise AssertionError('No dispatch after disconnect')
  worker=ReasoningWorker(None,evidence_request=EvidenceTransport(transcribe,send));ident=worker.submit('task',audio_evidence=AudioEvidence(0,0,bytes(32000)));await started.wait();await worker.close();release.set();await worker.tasks[ident]
  self.assertEqual(sent,[]);self.assertIsNone(worker.take_results());self.assertNotIn('audio_evidence',worker.states[ident])

 def test_general_profile_supports_complete_comparisons_with_bounded_response(self):
  from portable.venus.reasoning import ReasoningWorker
  e=AudioEvidence(0,0,bytes(32000));v=contextual_payload(payload('Compare loan terms',{}),self.recognition(e))
  self.assertIn('one hundred and twenty spoken English words',v['instructions'])
  self.assertEqual(ReasoningWorker(None,evidence_request=EvidenceTransport(lambda e:None,lambda v:None)).maximum_result_words,120)
  self.assertEqual(ReasoningWorker(lambda v:None).maximum_result_words,60)
