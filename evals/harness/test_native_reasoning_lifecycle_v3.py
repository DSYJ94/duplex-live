import json
import unittest

from evals.native_reasoning_lifecycle_v3 import NativeReasoningLifecycle, ACTION_CONTRACT


def response(identity,operation,content,targets=None):
    args=dict(content=content)
    if targets is not None:args['task_ids']=targets
    return dict(id='response-'+identity,model='fixture-model',status='completed',
                output=[dict(type='function_call',name=operation,arguments=json.dumps(args))])


class ExplicitActionContract(unittest.IsolatedAsyncioTestCase):
    async def prepare(self):
        world=NativeReasoningLifecycle(lambda:10.,0.)
        context=[dict(speaker='user',text='Solve the original task.',received_at_s=1.)]
        await world.reserve_native_request('original',context)
        await world.start_backend('original','fixture-model')
        await world.complete_backend('original',response('original','respond_to_user','Original answer.'),actual_transport=False)
        updated=context+[dict(speaker='user',text='Change one constraint.',received_at_s=2.)]
        await world.reserve_native_request('new',updated)
        payload=await world.start_backend('new','fixture-model')
        return world,payload,updated

    async def test_ready_but_undelivered_original_is_explicitly_an_active_target(self):
        world,payload,context=await self.prepare();data=json.loads(payload['input'])
        self.assertEqual(data['arrived_user_transcript'],context)
        self.assertEqual(data['active_requests']['original']['phase'],'result_ready')
        self.assertEqual([t['name']for t in payload['tools']],['respond_to_user','cancel_reasoning','replace_reasoning'])
        self.assertEqual(payload['instructions'],ACTION_CONTRACT)
        self.assertIn('preserves every existing active request',payload['instructions'])
        self.assertEqual(payload['tools'][2]['parameters']['properties']['task_ids']['items']['enum'],['original'])
        self.assertEqual(await world.delivery_state('original'),'pending_decision')

    async def test_actual_replacement_retires_original_without_changing_provider_response(self):
        world,_,_=await self.prepare();raw=response('new','replace_reasoning','Corrected answer.',['original'])
        receipt=await world.complete_backend('new',raw,actual_transport=False)
        self.assertEqual(world.tasks['new']['response'],raw)
        self.assertEqual(receipt['operation'],'replace_reasoning');self.assertEqual(receipt['revision'],2)
        self.assertEqual(await world.delivery_state('original'),'retired')

    async def test_model_can_preserve_unrelated_work_and_host_never_rewrites_wrong_action(self):
        world,_,_=await self.prepare();raw=response('new','respond_to_user','Independent answer.')
        receipt=await world.complete_backend('new',raw,actual_transport=False)
        self.assertEqual(receipt['operation'],'respond_to_user')
        self.assertFalse(world.tasks['original']['cancelled']);self.assertEqual(world.tasks['new']['response'],raw)
        self.assertEqual(await world.delivery_state('original'),'ready')
        self.assertFalse(world.evidence()['intent_or_case_classifier'])


if __name__=='__main__':unittest.main()
