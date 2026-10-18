import unittest
from evals.duplex_endpoint import TurnEndpoint


class EndpointTest(unittest.TestCase):
    def decoder(self,ids):
        words={4:'Project',5:'Elm',6:'North.',7:'Cancel',8:'that',9:'lookup.'}
        return ' '.join(words[i] for i in ids)

    def test_interword_head_pulses_do_not_split_project_or_cancellation(self):
        for sequence,expected in [([4,5,6],'Project Elm North.'),([7,8,9],'Cancel that lookup.')]:
            p=TurnEndpoint(self.decoder);events=[];frame=0
            for token in sequence:
                for value in [token,3,3,3]:
                    frame+=1;r=p.update(value,.99,0.,frame*.08)
                    if r['utterance']:events.append(r['utterance']['text'])
            self.assertEqual(events,[expected])
            for _ in range(8):
                frame+=1;self.assertIsNone(p.update(3,.99,0.,frame*.08)['utterance'])

    def test_terminal_text_does_not_dispatch_while_user_still_speaks(self):
        p=TurnEndpoint(self.decoder);events=[]
        for frame,token in enumerate([4,5,6,3,3,3],1):
            r=p.update(token,.99,.1,frame*.08);self.assertIsNone(r['utterance'])
            if r['speech_started']:events.append('start')
        self.assertEqual(events,['start']);self.assertTrue(p.speech_active)
        for frame in (7,8):self.assertIsNone(p.update(3,.99,0.,frame*.08)['utterance'])
        r=p.update(3,.99,0.,9*.08)
        self.assertTrue(r['speech_ended']);self.assertEqual(r['utterance']['text'],'Project Elm North.')

    def test_invalid_or_repeated_time_is_rejected(self):
        p=TurnEndpoint(self.decoder);p.update(4,.8,0.,.08)
        for score,rms,time in [(float('nan'),0.,.16),(.9,-1.,.16),(.9,0.,.08)]:
            with self.assertRaises(ValueError):p.update(3,score,rms,time)


if __name__=='__main__':unittest.main()
