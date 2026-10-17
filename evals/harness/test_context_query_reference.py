import unittest
from evals.context_query_reference import compose,ASR_SOURCE


class QueryReferenceTest(unittest.TestCase):
    def user(self,text='What is the room marker?',start=1.,end=2.):
        return dict(role='user',text=text,start_s=start,end_s=end,source=ASR_SOURCE)

    def test_actual_user_context_is_bound_to_reply_without_assistant_claims(self):
        context=[self.user(),dict(role='assistant',text='Unverified orange circle'),
            self.user('Project Willow West.',3.,4.)]
        text,meta=compose('Verified result: blue oval.',context,4.2,'delegate')
        self.assertIn('Earlier user speech: What is the room marker?',text)
        self.assertIn('Latest user speech: Project Willow West.',text)
        self.assertNotIn('Unverified orange circle',text)
        self.assertTrue(text.endswith('Verified result: blue oval.'))
        self.assertTrue(meta['supplied']);self.assertFalse(meta['reference_transcript_supplied'])

    def test_reference_or_future_or_invalid_words_are_rejected(self):
        for key,value in [('source','measured completed words: diagnostic oracle'),('end_s',3.1),
                          ('start_s',-1.),('end_s',float('nan')),('text','')]:
            with self.subTest(key=key,value=value):
                user=self.user();user[key]=value
                with self.assertRaises(RuntimeError):compose('reply',[user],2.1,'answer')

    def test_failed_recognition_does_not_become_a_query_anchor(self):
        text,meta=compose('Please repeat.',[],1.,'error')
        self.assertEqual(text,'Please repeat.');self.assertFalse(meta['supplied'])

    def test_long_text_is_rejected_instead_of_cutting_the_latest_question(self):
        with self.assertRaises(RuntimeError):compose('reply',[self.user('x'*4001)],3.,'answer')
        with self.assertRaises(RuntimeError):compose('reply',[],3.,'answer')


if __name__=='__main__':unittest.main()
