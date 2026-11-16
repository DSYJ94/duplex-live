import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from training.personaplex.official_quantitative import quantitative_rows
from training.personaplex.official_quantitative_host import official_command

class MissingOfficialTest(unittest.TestCase):
    def test_only_missing_quantitative_inputs_no_gold_or_duplicate_controls(self):
        rows=[dict(id=str(i),kind='basic'if i<12 else'quantitative',question='question',audio='/a.wav',rate=24000,audio_sha256='hash')for i in range(76)]
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'questions.json';p.write_text(json.dumps(rows));self.assertEqual(len(quantitative_rows(p)),64)
            rows[12]['answer']='forbidden';p.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):quantitative_rows(p)
    def test_preserves_actual_selected_gpu_and_existing_audio_runtime(self):
        stub=['apptainer','CUDA_VISIBLE_DEVICES=0,1,2,3','python','-m']
        with patch('training.personaplex.official_quantitative_host.container_command',return_value=stub.copy())as factory,patch.dict(os.environ,CUDA_VISIBLE_DEVICES='2'):
            cmd=official_command(Path('/work'),True,{},Path('/backup'));self.assertIn('CUDA_VISIBLE_DEVICES=2',cmd)
            env=factory.call_args.kwargs['environment'];self.assertIn('personaplex-runtime-v1/pylibs',env['PYTHONPATH'])
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),patch('training.personaplex.official_quantitative_host.container_command',return_value=stub.copy()):
            with self.assertRaises(ValueError):official_command(Path('/work'),True,{},Path('/backup'))
