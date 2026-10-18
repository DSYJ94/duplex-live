import unittest
from evals.function_name_grammar import TokenPieceIndex


class OpenerGrammar(unittest.TestCase):
    def setUp(self):
        pieces=['<','function','=','answer','resume_speech','cancel_reasoning','>','>\n','>\n<parameter=content>','bad',' ',' <function=answer>\n','<function=resume_speech>','<function=cancel_reasoning>','<function=answer>','<fun','ction=','answer>\n','resume_speech>\n']
        self.ids={x:i for i,x in enumerate(pieces)};self.index=TokenPieceIndex(enumerate(pieces));self.openers=['<function=answer>','<function=resume_speech>']
    def test_unavailable_name_is_excluded_without_selecting_between_valid_actions(self):
        allowed=self.index.allowed(self.openers,'<function=')
        self.assertIn(self.ids['answer'],allowed);self.assertIn(self.ids['resume_speech'],allowed);self.assertNotIn(self.ids['cancel_reasoning'],allowed)
    def test_merged_opener_body_tokens_and_leading_whitespace_are_preserved(self):
        self.assertIn(self.ids['>\n<parameter=content>'],self.index.allowed(self.openers,'<function=answer'))
        self.assertIn(self.ids[' <function=answer>\n'],self.index.allowed(self.openers,''))
        self.assertIn(self.ids['resume_speech>\n'],self.index.allowed(self.openers,'<function='))
    def test_body_is_unconstrained_and_illegal_prefix_is_rejected(self):
        self.assertIsNone(self.index.allowed(self.openers,'<function=answer>\n<parameter=content>Actual text'))
        with self.assertRaises(ValueError):self.index.allowed(self.openers,'<function=cancel_reasoning')
    def test_every_token_of_a_valid_merged_opening_stays_eligible(self):
        for tokens in [['<fun','ction=','answer>\n'],[' ','<','function','=','resume_speech','>\n']]:
            prefix=''
            for token in tokens:
                self.assertIn(self.ids[token],self.index.allowed(self.openers,prefix));prefix+=token
            self.assertIsNone(self.index.allowed(self.openers,prefix))


if __name__=='__main__':unittest.main()
