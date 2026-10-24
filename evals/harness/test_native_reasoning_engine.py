import unittest
from evals.native_reasoning_engine import span_plan


class ActualContextClocks(unittest.TestCase):
    def test_delivered_text_is_inserted_after_complete_wire_write(self):
        arrivals=[dict(ready_at_s=.401,text='Actual receipt',server_ack_received_at_s=.7)]
        self.assertEqual(span_plan(arrivals,10),[dict(inject_frame=6,reference='Actual receipt')])
        self.assertEqual(arrivals[0]['server_ack_received_at_s'],.7)

    def test_no_future_missing_or_ambiguous_delivery_is_accepted(self):
        for rows in ([dict(ready_at_s=1.,text='late')],
                [dict(ready_at_s=float('nan'),text='invalid')],
                [dict(ready_at_s=.401,text='first'),dict(ready_at_s=.402,text='second')]):
            with self.assertRaises(ValueError):span_plan(rows,10)


if __name__=='__main__':unittest.main()
