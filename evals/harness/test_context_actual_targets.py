"""Causal/ownership guards, not model or training qualification."""
import unittest
import numpy as np
from evals.context_actual_targets import matching_user_stimulus, actual_private_receipt, sentence_target, captured_packets


class FutureTargetGuards(unittest.TestCase):
    def test_user_audio_allows_only_zero_tail_and_identical_words(self):
        a=np.array([.1,.2,0.],dtype=np.float32)
        self.assertTrue(matching_user_stimulus(a,np.pad(a,(0,2)),['same'],['same']))
        b=np.pad(a,(0,2));b[-1]=.1
        self.assertFalse(matching_user_stimulus(a,b,['same'],['same']))
        self.assertFalse(matching_user_stimulus(a,a,['same'],['different']))

    def test_actual_failed_cancelled_and_wrong_world_are_not_facts(self):
        event={'event':'backend_result_ready','task_id':'ret-0','applied_at_s':2.,
            'result':{'route':'delegate','content':'The badge is purple star.',
            'receipts':[{'tool':'lookup_record','executed':True,'read_only':True,
                'project':'Birch North','field':'badge','value':'purple star'}]}}
        self.assertIs(actual_private_receipt([event],'Birch North','badge','purple star',3.),event)
        for events,value,cut in [([event],'orange square',3.),([event],'purple star',1.),
            ([{**event,'event':'backend_request_failed'}],'purple star',3.),
            ([{**event,'result':{**event['result'],'route':'error'}}],'purple star',3.),
            ([event,{'event':'cancelled_backend_rejected','task_id':'ret-0'}],'purple star',3.)]:
            with self.assertRaises(RuntimeError):actual_private_receipt(events,'Birch North','badge',value,cut)

    def test_quiet_audio_cannot_hide_unaccounted_speech(self):
        rows=[{'output_samples':2,'available_s':.1}]
        ledger=[{'packet':0,'samples':2,'available_s':.1,
            'playback_start_sample':2,'inserted_gap_samples':2}]
        audio=np.array([0.,0.,.2,.3],dtype=np.float32)
        np.testing.assert_array_equal(captured_packets(audio,ledger,rows)[0],audio[2:])
        audio[0]=.1
        with self.assertRaises(RuntimeError):captured_packets(audio,ledger,rows)
        with self.assertRaises(RuntimeError):captured_packets(np.zeros(5),ledger,rows)

    def test_fact_gate_uses_delivered_payload_and_still_requires_receipt(self):
        event={'event':'backend_result_ready','task_id':'ret-0','applied_at_s':2.,
            'injected_reference':'Verified badge: purple star.',
            'result':{'route':'delegate','content':'Physics explanation only.',
                'receipts':[{'tool':'lookup_record','executed':True,'read_only':True,
                    'project':'Birch North','field':'badge','value':'purple star'}]}}
        self.assertIs(actual_private_receipt([event],'Birch North','badge','purple star',3.),event)
        for bad in [{**event,'injected_reference':'No result supplied.'},
                {**event,'result':{**event['result'],'receipts':[]}}]:
            with self.assertRaises(RuntimeError):actual_private_receipt([bad],'Birch North','badge','purple star',3.)

    def test_pending_payload_is_state_only_and_never_a_fact_receipt(self):
        from evals.context_engine_replay import recorded_span_text
        event={'event':'backend_pending_status_applied','reference':'Request pending.',
            'word_or_acoustic_tokens_forced':False,'applied_at_s':1.}
        self.assertEqual(recorded_span_text(event),'Request pending.')
        with self.assertRaises(RuntimeError):
            recorded_span_text({**event,'result':{'content':'purple star'}})
        with self.assertRaises(RuntimeError):actual_private_receipt([event],'Birch North','badge','purple star',3.)

    def test_explicit_empty_span_cannot_fall_back_to_uninjected_worker_prose(self):
        from evals.context_engine_replay import recorded_span_text
        with self.assertRaises(RuntimeError):
            recorded_span_text({'injected_reference':'','result':{'content':'purple star'}})

    def test_sentence_keeps_native_word_spacing_and_requires_receipt(self):
        encode=lambda w:{'The':[5],'badge':[6],'is':[7],'purple':[8],'star.':[9]}[w]
        words=[['The',[1.,1.15],'SPEAKER_MAIN'],['badge',[1.2,1.35],'SPEAKER_MAIN'],
            ['is',[1.4,1.55],'SPEAKER_MAIN'],['purple',[1.6,1.75],'SPEAKER_MAIN'],
            ['star.',[1.8,2.],'SPEAKER_MAIN']]
        codes=[3]*30
        for i,t in zip([13,15,18,20,23],[5,6,7,8,9]):codes[i]=t
        result=sentence_target(codes,words,'purple star',encode,.9)
        self.assertEqual((result['source_start_frame'],result['source_end_frame']),(12,27))
        with self.assertRaises(RuntimeError):sentence_target(codes,words,'purple star',encode,1.1)
        codes[12]=4
        with self.assertRaises(RuntimeError):sentence_target(codes,words,'purple star',encode,.9)


if __name__=='__main__':unittest.main()
