import copy
import unittest
from collections import Counter
from training.personaplex.language_schedule import build,window
from training.personaplex.text_rehearsal import TextSpec,validate_window_identity


class LanguageScheduleTest(unittest.TestCase):
    def setUp(self):
        self.spec=TextSpec(2,32,0,3,8,frozenset((0,1,2,3,4)))
        self.rows=[dict(id='doc-%d'%i,tokens=[5+i]*(i+1),mask=[True]*(i+1),fold='TRAIN',
                        semantic_qualified=True,leakage_qualified=True,qualification_sha256='a'*64,
                        tokenizer_sha256='b'*64,source_sha256='c'*64) for i in range(5)]
        self.args=dict(tokenizer_sha256='b'*64,qualification_sha256='a'*64,world_size=3,
                       epochs=2,max_global_target_tokens=7,coefficient=.3,seed=902)

    def test_complete_epochs_token_mass_padding_and_resume(self):
        plan=build(self.rows,self.spec,**self.args)
        self.assertEqual(plan,build(list(reversed(self.rows)),self.spec,**self.args))
        for epoch in range(2):
            ids=[identity for row in plan['updates'] if row['epoch']==epoch for identity in row['document_ids']]
            self.assertEqual(Counter(ids),Counter(row['id'] for row in self.rows))
        for step,row in enumerate(plan['updates']):
            count=0;valid_ids=[];microsteps=[]
            for rank in range(3):
                exposure=window(plan,self.rows,self.spec,rank=rank,next_update=step)
                validate_window_identity(exposure,dict(language_training=plan['identity']),
                    dict(language_data_qualified=True,language_native_gradient_restore_passed=True),step)
                microsteps.append(len(exposure['batches']))
                for identity,batch,padding in zip(exposure['document_ids'],exposure['batches'],exposure['padding_only']):
                    count+=int(batch['mask'].sum())
                    if not padding:valid_ids.append(identity)
            self.assertEqual(count,row['global_target_tokens'])
            self.assertLessEqual(count,7)
            self.assertEqual(Counter(valid_ids),Counter(row['document_ids']))
            self.assertEqual(len(set(microsteps)),1)
        next_step=len(plan['updates'])//2
        self.assertEqual(window(plan,self.rows,self.spec,rank=0,next_update=next_step)['step'],next_step)
        with self.assertRaises(ValueError):window(plan,self.rows,self.spec,rank=0,next_update=len(plan['updates']))

    def test_unqualified_holdout_changed_data_or_recipe_cannot_enter(self):
        for field,value in [('fold','DEV'),('semantic_qualified',False),('leakage_qualified',False),('tokenizer_sha256','d'*64)]:
            rows=copy.deepcopy(self.rows);rows[0][field]=value
            with self.assertRaises(ValueError):build(rows,self.spec,**self.args)
        with self.assertRaises(ValueError):build(self.rows,self.spec,**dict(self.args,max_global_target_tokens=2))
        plan=build(self.rows,self.spec,**self.args)
        rows=copy.deepcopy(self.rows);rows[0]['tokens']=[10]
        with self.assertRaises(ValueError):window(plan,rows,self.spec,rank=0,next_update=0)
        for field in ('coefficient','corpus_sha256','schedule_sha256'):
            altered=copy.deepcopy(plan);altered['identity'][field]=.5 if field=='coefficient' else 'f'*64
            with self.assertRaises(ValueError):window(altered,self.rows,self.spec,rank=0,next_update=0)


if __name__=='__main__':unittest.main()
