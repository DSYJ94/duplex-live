"""Mutation checks for full20, exact raw speech and paired private stimuli."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from evals.native_fresh_inputs import specifications,generate,load_inputs,waveform
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

    def test_bad_words_cannot_be_relabelled_through_references(self):
        with tempfile.TemporaryDirectory() as root:
            data=self.fixture(root);p=data/'inputs.json';m=json.loads(p.read_text());row=m['rows'][0]
            ref=Path(row['reference_path']);r=json.loads(ref.read_text());r['utterances'][0]['text']='A different question.'
            ref.write_text(json.dumps(r));row['reference_sha256']=sha256(ref);p.write_text(json.dumps(m))
            with self.assertRaisesRegex(RuntimeError,'exact words'):load_inputs(data)

    def test_peak_finite_and_silence_fail_without_attenuation(self):
        import numpy as np
        for value in (0.,1.,float('nan')):
            with self.subTest(value=value):self.assertFalse(waveform(np.full(2400,value),24000)['passed'])
        self.assertTrue(waveform(np.full(2400,.1),24000)['passed'])


if __name__=='__main__':unittest.main()
