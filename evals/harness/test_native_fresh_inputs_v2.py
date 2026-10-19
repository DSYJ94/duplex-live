"""Mutation checks for full20, exact raw speech and paired private stimuli."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from evals.native_fresh_inputs_v2 import specifications,generate,load_inputs,waveform,inference_input,PLAN_SHA256,ASR_WEIGHT,agreement
from evals.moshi_model_download import sha256


class NativeFreshInputs(unittest.TestCase):
    def synthesize(self,text,voice,path):
        import numpy as np
        import soundfile as sf
        from data.synthetic.recovery_curriculum import MODEL
        pcm=np.full(2400,.1,dtype=np.float32);sf.write(path,pcm,24000,subtype='FLOAT')
        receipt=dict(source_model=MODEL,voice=voice,input_text=text,sha256=sha256(path),sample_rate=24000)
        path.with_suffix('.receipt.json').write_text(json.dumps(receipt))
        return receipt

    def fixture(self,root):
        data=Path(root)/'inputs'
        with patch('data.synthetic.recovery_curriculum.synthesize',side_effect=self.synthesize) as mock:
            generate(data)
        self.assertEqual(mock.call_count,36)
        return data

    def test_fixed20_and_counterfactual_source_is_byte_identical(self):
        self.assertEqual(len(specifications()),20)
        with tempfile.TemporaryDirectory() as root:
            m=load_inputs(self.fixture(root))
            a,b=m['rows'][8:10]
            self.assertEqual(a['wav_sha256'],b['wav_sha256'])
            self.assertEqual(a['reference_sha256'],b['reference_sha256'])
            self.assertFalse(m['training_approved'])
            self.assertTrue(m['do_not_train'])

    def test_missing_or_duplicate_case_is_rejected_even_with_all_selected_passes(self):
        with tempfile.TemporaryDirectory() as root:
            data=self.fixture(root);path=data/'inputs.json';original=path.read_text()
            for mode in ('missing','duplicate'):
                m=json.loads(original)
                if mode=='missing':m['rows'].pop()
                else:m['rows'][-1]=m['rows'][0]
                path.write_text(json.dumps(m))
                with self.subTest(mode=mode),self.assertRaises(RuntimeError):load_inputs(data)

    def test_gain_or_extra_samples_rejected_even_if_manifest_hash_is_refreshed(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as root:
            data=self.fixture(root);p=data/'inputs.json';m=json.loads(p.read_text());row=m['rows'][0]
            wav=Path(row['path']);pcm,sr=sf.read(wav,dtype='float32');pcm*=.5;sf.write(wav,pcm,sr,subtype='FLOAT')
            row['wav_sha256']=sha256(wav);p.write_text(json.dumps(m))
            with self.assertRaisesRegex(RuntimeError,'samples were changed'):load_inputs(data)

    def test_shortened_observation_tail_rejected_with_refreshed_manifest(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as root:
            data=self.fixture(root);p=data/'inputs.json';m=json.loads(p.read_text());row=m['rows'][0]
            wav=Path(row['path']);pcm,sr=sf.read(wav,dtype='float32');pcm=pcm[:-sr]
            sf.write(wav,pcm,sr,subtype='FLOAT');row['wav_sha256']=sha256(wav);row['duration_s']=len(pcm)/sr
            p.write_text(json.dumps(m))
            with self.assertRaisesRegex(RuntimeError,'observation tail'):load_inputs(data)

    def test_bad_words_cannot_be_relabelled_through_references(self):
        with tempfile.TemporaryDirectory() as root:
            data=self.fixture(root);p=data/'inputs.json';m=json.loads(p.read_text());row=m['rows'][0]
            ref=Path(row['reference_path']);r=json.loads(ref.read_text());r['utterances'][0]['text']='A different question.'
            ref.write_text(json.dumps(r));row['reference_sha256']=sha256(ref);p.write_text(json.dumps(m))
            with self.assertRaisesRegex(RuntimeError,'exact words'):load_inputs(data)

    def quality(self,data):
        import soundfile as sf
        m=load_inputs(data);folder=data.parent/'quality';folder.mkdir();results=[]
        for row in m['rows']:
            ref=json.loads(Path(row['reference_path']).read_text());clips=[]
            for i,u in enumerate(ref['utterances']):
                name=row['scenario_id']+'-%d-asr.json'%i;asr=folder/name
                t=dict(text=u['text'],segments=[dict(start=0.,end=.1,words=[dict(text=u['text'],start=0.,end=.1)])])
                asr.write_text(json.dumps(t));wav=Path(row['path']).with_name(u['source_file']);pcm,sr=sf.read(wav,dtype='float32')
                clips.append(dict(ordinal=i,asr_file=name,asr_sha256=sha256(asr),raw_wav_sha256=sha256(wav),
                                  passed=True,valid_bounds=True,waveform=waveform(pcm,sr),wer=0.,agreement=agreement(u['text'],u['text']),original_source_unchanged=True))
            results.append(dict(scenario_id=row['scenario_id'],wav_sha256=row['wav_sha256'],passed=True,clips=clips))
        q=dict(schema='dl-native-fresh-quality-v2',passed=True,inputs_manifest_sha256=sha256(data/'inputs.json'),
               plan_sha256=PLAN_SHA256,threshold=.05,peak_bar=.995,results=results,
               judge_sha256=sha256(Path(__file__).parents[1]/'native_fresh_inputs_v2.py'),training_approved=False,
               do_not_train=True,invalid_recordings_frozen_unscored=True,no_passing_subset=True,
               asr='whisper large-v3 CPU',asr_weight_sha256=ASR_WEIGHT,expected_text_prompt_used=False)
        path=folder/'quality.json';path.write_text(json.dumps(q));return path

    def test_inference_receives_no_script_or_expected_answer_and_full20_is_required(self):
        with tempfile.TemporaryDirectory() as root:
            data=self.fixture(root);quality=self.quality(data)
            safe=inference_input(data,quality,'native-fresh2-008')
            self.assertNotIn('required',safe)
            self.assertNotIn('turns',safe)
            self.assertNotIn('reference_path',safe)
            self.assertEqual(safe['worker_only_world'],specifications()[8]['world'])
            q=json.loads(quality.read_text());q['results'][-1]['passed']=False;quality.write_text(json.dumps(q))
            with self.assertRaises(RuntimeError):inference_input(data,quality,'native-fresh2-008')

    def test_forged_pass_or_refreshed_asr_hash_cannot_hide_wrong_words(self):
        with tempfile.TemporaryDirectory() as root:
            data=self.fixture(root);quality=self.quality(data);q=json.loads(quality.read_text())
            clip=q['results'][0]['clips'][0];asr=quality.parent/clip['asr_file']
            t=json.loads(asr.read_text());t['text']='Incorrect unrelated question.';asr.write_text(json.dumps(t))
            clip['asr_sha256']=sha256(asr);quality.write_text(json.dumps(q))
            with self.assertRaisesRegex(RuntimeError,'word/time/peak'):inference_input(data,quality,'native-fresh2-008')

    def test_tail_and_spelling_equivalence_are_fixed(self):
        self.assertEqual(agreement('What colour?', 'What color?')['spelling_only_wer'],0.)
        self.assertGreater(agreement('warm day','worn day')['spelling_only_wer'],.05)
        with tempfile.TemporaryDirectory() as root:
            data=self.fixture(root);quality=self.quality(data);q=json.loads(quality.read_text())
            q['asr_weight_sha256']='wrong';quality.write_text(json.dumps(q))
            with self.assertRaises(RuntimeError):inference_input(data,quality,'native-fresh2-008')

    def test_peak_finite_and_silence_fail_without_attenuation(self):
        import numpy as np
        for value in (0.,1.,float('nan')):
            with self.subTest(value=value):self.assertFalse(waveform(np.full(2400,value),24000)['passed'])
        self.assertTrue(waveform(np.full(2400,.1),24000)['passed'])


if __name__=='__main__':unittest.main()
