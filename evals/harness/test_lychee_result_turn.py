import unittest

from evals.lychee_result_turn import ResultTurn


class ResultTurnTest(unittest.TestCase):
    def test_no_activation_before_receipt_or_during_user_or_previous_reply(self):
        turn=ResultTurn()
        for observed,refreshed,state,quiet in [(False,False,'l',3),(True,True,'l',3),
                (True,False,'s',3),(True,False,'l',1)]:
            self.assertFalse(turn.eligible(observed,refreshed,state,quiet))
        self.assertTrue(turn.eligible(True,False,'l',2))

    def test_interrupt_ends_one_attempt_and_restores_only_at_listening_boundary(self):
        turn=ResultTurn();turn.activate()
        turn.observe({'type':'speaking_done'})
        self.assertFalse(turn.should_restore('l'))
        turn.observe({'type':'state_change','from':'L','to':'S'})
        turn.observe({'type':'state_change','from':'S','to':'L','reason':'interrupt'})
        self.assertFalse(turn.should_restore('s'))
        self.assertTrue(turn.should_restore('l'))
        turn.restored()
        self.assertFalse(turn.eligible(True,False,'l',20))
        with self.assertRaisesRegex(RuntimeError,'twice'):turn.activate()

    def test_eos_is_finished_but_not_a_success_claim(self):
        turn=ResultTurn();turn.activate()
        turn.observe({'type':'state_change','from':'L','to':'S'})
        turn.observe({'type':'speaking_done','reason':'eos'})
        self.assertTrue(turn.finished)
        self.assertFalse(turn.should_restore('s'))
        self.assertTrue(turn.should_restore('l'))


if __name__=='__main__':unittest.main()
