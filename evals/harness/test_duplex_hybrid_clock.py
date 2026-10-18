import unittest
from types import SimpleNamespace
from evals.duplex_hybrid_probe import PlaybackClock
from evals.duplex_speaker import StreamingSpeaker


class HybridClockTests(unittest.TestCase):
    def test_online_clock_matches_actual_causal_playout_and_stop(self):
        import numpy as np
        from evals.utterance_playout import render_utterances
        clock=PlaybackClock();packets=[];ends=[]
        for at in (.1,.15,.9):
            _,end=clock.append(at,1920);ends.append(end)
            packets.append(dict(event_id='speech',available_s=at,pcm=np.ones(1920,dtype=np.float32)))
        _,report=render_utterances(packets,buffer_s=.16)
        self.assertEqual([round(e*24000) for e in ends],
            [r['playback_start_sample']+r['samples'] for r in report['events']['speech']['packets']])
        clock.stop(.94)
        _,stopped=render_utterances(packets,buffer_s=.16,stops={'speech':.94})
        last=stopped['events']['speech']['packets'][-1]['playback_start_sample']
        self.assertEqual(stopped['played_samples'],3840+max(0,clock.end-last))

    def test_single_steps_keep_exact_upstream_completion_and_flush(self):
        # Upstream process_last first drains entries/end_step, then exactly
        # delay_steps + max(delays) + 8 further steps. No guessed EOS duration.
        speaker=StreamingSpeaker.__new__(StreamingSpeaker)
        state=SimpleNamespace(entries=['word'],end_step=None)
        gen=SimpleNamespace(offset=0,state=state)
        def step():
            gen.offset+=1
            if gen.offset==2:state.entries=[];state.end_step=8
            if gen.offset==8:state.end_step=None
        gen._step=step;speaker.gen=gen;speaker.tts=SimpleNamespace(delay_steps=3,lm=SimpleNamespace(delays=[0,1]))
        speaker.active=True;speaker.frames=[];speaker.flush_remaining=None
        while speaker.active:speaker.step()
        self.assertEqual(gen.offset,8+3+1+8)


if __name__=='__main__':unittest.main()
