import unittest
from evals.duplex_speech_obligations import snapshot,consume


def response(identity,route,text):
    return dict(task_id=identity,request_text='Actual user question '+identity,result=dict(route=route,content=text))


class SpeechObligations(unittest.TestCase):
    def test_fulfilled_ordinary_answer_does_not_discard_unrelated_or_private_work(self):
        answer=response('ordinary','answer','Counterclockwise.')
        other=response('other','answer','Four.')
        private=response('lookup','delegate','A private result.')
        queue=[answer,other,private];pending=snapshot(queue,answer)
        self.assertEqual([r['id'] for r in pending],['ordinary','other'])
        self.assertEqual(consume(queue,['ordinary'],pending),[other,private])
        self.assertEqual(consume(queue,[],pending),queue)
        with self.assertRaises(ValueError):consume(queue,['lookup'],pending)
        with self.assertRaises(ValueError):consume(queue,['ordinary','ordinary'],pending)

    def test_conflicting_queued_text_is_rejected_instead_of_forging_heard_state(self):
        first=response('same','answer','First actual answer.')
        second=response('same','answer','Another answer.')
        with self.assertRaises(RuntimeError):snapshot([first],second)


if __name__=='__main__':unittest.main()
