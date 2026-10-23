from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from evals import native_reasoning_convert as conversion


def clock_fixture():
    tasks={'obsolete':dict(native_request_observed_at_s=.21,cancelled=True),
           'cancel':dict(native_request_observed_at_s=.41,provider_finished_at_s=.6,wire_write_completed_at_s=.66,
                         receipt=dict(content='The search is off.',operation='cancel_reasoning',revision=2,actual_status_committed_at_s=.61))}
    events=[dict(type='session.delegation.created',received_at_s=.2,delegation=dict(id='obsolete')),
            dict(type='session.delegation.created',received_at_s=.4,delegation=dict(id='cancel')),
            dict(type='local.commentary.write',received_at_s=.65,delegation_id='cancel',event_id='write-cancel',content='The search is off.'),
            dict(type='session.commentary.appended',received_at_s=.7,client_event_id='write-cancel')]
    return dict(tasks=tasks),events


class Conversion(unittest.TestCase):
    def test_retired_unwritten_result_has_no_fictitious_span_but_both_requests_survive(self):
        ledger,events=clock_fixture();markers,arrivals=conversion.task_clocks(ledger,events,1.5)
        self.assertEqual([m['task_id']for m in markers],['obsolete','cancel'])
        self.assertEqual(len(arrivals),1);self.assertEqual(arrivals[0]['task_id'],'cancel')
        self.assertEqual(arrivals[0]['ready_at_s'],.66)
        self.assertEqual(arrivals[0]['server_ack_received_at_s'],.7)
        self.assertTrue(ledger['tasks']['obsolete']['cancelled'])
    def test_missing_ack_or_result_before_native_request_rejects(self):
        ledger,events=clock_fixture()
        with self.assertRaises(ValueError):conversion.task_clocks(ledger,events[:-1],1.5)
        ledger['tasks']['cancel']['provider_finished_at_s']=.1
        with self.assertRaises(ValueError):conversion.task_clocks(ledger,events,1.5)
    def test_all_words_and_measured_user_offsets_survive_native_markers(self):
        ledger,events=clock_fixture()
        scene=dict(clips=[dict(object_id='question',text='Seven tokens.')])
        report=dict(source_starts=[1920],source_lengths=[12000],observed_until_s=1.5)
        checks={'question':dict(passed=True,measured_words=[dict(text='Seven',start=.1,end=.2),dict(text='tokens.',start=.25,end=.4)])}
        words=[dict(text='All',start=.8,end=.9),dict(text='right.',start=1.0,end=1.2)]
        timeline,user,markers,arrivals,offset=conversion.measured_timeline('scene','cancel-reasoning',scene,report,
            checks,{'question':dict(wav_sha256='original')},words,ledger,events)
        self.assertEqual([e.text for e in timeline.events if e.ch=='assistant'and e.kind!='marker'],['All right.'])
        self.assertAlmostEqual(user[0]['start_s'],.18);self.assertAlmostEqual(user[-1]['end_s'],.48)
        self.assertEqual(len(markers),2);self.assertEqual(len(arrivals),1);self.assertLessEqual(offset,200)
    def test_missing_semantic_approval_cannot_emit_a_directory_or_subset(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);out=root/'converted'
            with self.assertRaisesRegex(ValueError,'Complete32 independent semantic'):
                conversion.convert(SimpleNamespace(),root,root/'speech',root/'prepared',root/'missing-semantic',out)
            self.assertFalse(out.exists())

    def test_new_output_cannot_be_created_inside_an_original_input_root(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);source=root/'source';source.mkdir()
            c=SimpleNamespace(source=source,source_quality=root/'quality/source.json',old_failed=root/'failed/speech.json')
            semantic=root/'semantic/semantic.json';semantic.parent.mkdir();semantic.write_text('{}')
            out=source/'derived'
            with self.assertRaisesRegex(ValueError,'outside every immutable input root'):
                conversion.convert(c,root/'batch',root/'speech/report.json',root/'prepared.json',semantic,out)
            self.assertFalse(out.exists());self.assertEqual(list(source.iterdir()),[])
    def test_last_partial_frame_or_duplicate_request_frame_does_not_silently_drop_control(self):
        ledger,events=clock_fixture()
        ledger['tasks']['obsolete']['native_request_observed_at_s']=.415
        with self.assertRaises(ValueError):conversion.task_clocks(ledger,events,1.5)
        ledger,events=clock_fixture();ledger['tasks']['cancel']['native_request_observed_at_s']=1.49
        ledger['tasks']['cancel'].pop('wire_write_completed_at_s')
        with self.assertRaises(ValueError):conversion.task_clocks(ledger,events,1.5)


class PCMReadback(unittest.TestCase):
    def fixture(self,root):
        import json
        import numpy as np
        import soundfile as sf
        from data.synthetic.timeline import Timeline
        values=[];rows=[]
        for i,identity in enumerate(conversion.mixed.IDS):
            folder=root/identity;folder.mkdir()
            family=sorted(conversion.FAMILIES)[i//8]
            timeline=Timeline(identity,'native_reasoning4_teacher',[],dict(family=family))
            user=np.array([.125]*1920+[0.]*1920,dtype='float32')
            assistant=np.array([0.]*1920+[-.25]*1920,dtype='float32')
            side=dict(timeline=timeline.to_dict(),original_lifecycle=dict(tasks={}))
            for n,pcm in [('stem_user.wav',user),('stem_assistant.wav',assistant),('conversation.wav',np.stack([assistant,user],axis=1))]:
                sf.write(folder/n,pcm,24000,subtype='FLOAT')
            timeline.save(folder/'timeline.actual.json');(folder/'conversation.json').write_text(json.dumps(side))
            row=dict(scenario_id=identity,dir=identity,split='train',family=family,scenario_class='native_reasoning4_teacher',
                path=(Path(identity)/'conversation.wav').as_posix(),samples=3840,native_frames=2,duration=.16,
                max_grid_offset_ms=0.,artifacts={n:conversion.sha(folder/n)for n in conversion.FILES})
            rows.append(row);values.append((timeline,side,user,assistant,0.))
        for fold in('train','validation','test'):(root/(fold+'.jsonl')).write_text(''.join(json.dumps(r)+'\n'for r in rows if r['split']==fold))
        value=dict(schema=conversion.SCHEMA,accepted=rows,complete32=True,code_pins=conversion.code_pins(),
            channel_order=['assistant','user'],optimizer_updates=0,training_approved=False,
            native_codec_numerical_gradient_and_behavior_approval_required=True,original_words_audio_receipts_folds_unchanged=True)
        (root/'ingestion.json').write_text(json.dumps(value))
        return value,values
    def test_complete32_pcm_readback_rejects_rescale_even_if_artifact_hash_is_recomputed(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);value,derived=self.fixture(root)
            conversion.verify_derived(root,value,derived)
            target=root/conversion.mixed.IDS[0]/'stem_assistant.wav';pcm,rate=sf.read(target,dtype='float32')
            sf.write(target,pcm*.5,rate,subtype='FLOAT')
            value['accepted'][0]['artifacts']['stem_assistant.wav']=conversion.sha(target)
            with self.assertRaisesRegex(ValueError,'Original PCM'):
                conversion.verify_derived(root,value,derived)
    def test_partial32_and_nonempty_heldout_folds_are_rejected(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);value,derived=self.fixture(root)
            with self.assertRaises(ValueError):conversion.verify_derived(root,{**value,'accepted':value['accepted'][:-1]},derived)
            (root/'test.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'held-out folds'):
                conversion.verify_derived(root,value,derived)

    def test_extra_or_linked_artifacts_do_not_enter_transfer_inventory(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);value,derived=self.fixture(root)
            extra=root/conversion.mixed.IDS[0]/'unapproved.txt';extra.write_text('unapproved')
            with self.assertRaisesRegex(ValueError,'Unexpected or linked'):
                conversion.verify_derived(root,value,derived)
            extra.unlink();original=extra.parent/'stem_assistant.wav';copy=root/conversion.mixed.IDS[1]/'stem_assistant.wav'
            original.unlink();original.symlink_to(copy)
            with self.assertRaises(ValueError):conversion.verify_derived(root,value,derived)


if __name__=='__main__':unittest.main()
