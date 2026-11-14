import unittest
from data.synthetic.reasoning_audio1072_scribe import reusable

class ReuseTest(unittest.TestCase):
    def test_only_exact_identity_can_reuse(self):
        row=dict(id='one',sha256='abc',source_text='five pounds',fold='TRAIN')
        previous=dict(id='one',source_sha256='abc',source_words='five pounds',fold='TRAIN')
        self.assertTrue(reusable(row,previous))
        for key,value in [('id','other'),('source_sha256','changed'),('source_words','five'),('fold','DEV')]:
            self.assertFalse(reusable(row,dict(previous,**{key:value})),key)
        self.assertFalse(reusable(row,None))
