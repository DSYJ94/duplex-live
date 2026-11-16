import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from training.personaplex.local_reasoner_reference import digest,verify_pin,question_rows,finish_record,REVISION

class ReferenceTest(unittest.TestCase):
    def test_pin_verifies_actual_bytes_and_rejects_changed_or_escaping_file(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d);(p/'config.json').write_text('{}');record=dict(repository='Qwen/Qwen3.6-27B',revision=REVISION,files={'config.json':digest(p/'config.json')})
            pin=p/'pin.json';pin.write_text(json.dumps(record));expected=digest(pin)
            verify_pin(p,pin,expected)
            (p/'config.json').write_text('{ }')
            with self.assertRaises(ValueError):verify_pin(p,pin,expected)
            record['files']={'../outside':'bad'};pin.write_text(json.dumps(record))
            with self.assertRaises(ValueError):verify_pin(p,pin,digest(pin))

    def test_complete_input_only_cohort_refuses_gold_and_duplicates(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'q.json';rows=[dict(id=str(i),kind='basic',question='What is five minus two?',audio='/existing.wav',rate=24000,audio_sha256='sha')for i in range(76)]
            p.write_text(json.dumps(rows));self.assertEqual(len(question_rows(p)),76)
            rows[0]['answer']='3';p.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):question_rows(p)
            del rows[0]['answer'];rows[1]['id']='0';p.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):question_rows(p)

    def test_truncation_is_explicit_not_claimed_success(self):
        self.assertTrue(finish_record([1,2],[2,3],'ok',.2,10)['ended_with_EOS'])
        self.assertTrue(finish_record([1,4],2,'partial',.2,10)['generation_budget_exhausted'])
        self.assertFalse(finish_record([],2,'',.2,10)['ended_with_EOS'])

    def test_reference_owns_one_gpu_and_binds_only_shared_public_assets(self):
        from training.personaplex.local_reference_host import reference_command,PUBLIC,PIN,RUNTIME
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'2'}),patch('training.personaplex.local_reference_host.container_command',return_value=['apptainer','--env','CUDA_VISIBLE_DEVICES=0,1,2,3','--pwd','/source','python','-m'])as make:
            cmd=reference_command(Path('/source'),True,{},Path('/backup'))
        self.assertIn('CUDA_VISIBLE_DEVICES=2',cmd);self.assertNotIn('CUDA_VISIBLE_DEVICES=0,1,2,3',cmd)
        self.assertIn(str(PUBLIC)+':'+str(PUBLIC)+':ro',cmd);self.assertIn(str(PIN)+':'+str(PIN)+':ro',cmd)
        self.assertEqual(make.call_args.kwargs['environment']['PYTHONPATH'],str(RUNTIME/'pylibs')+':/source')
