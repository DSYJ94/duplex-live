import json,tempfile,unittest
from pathlib import Path
from training.personaplex.causal_probe import prepare,supplied_context

class ProbeInputs(unittest.TestCase):
    def test_all76_inputs_preserve_question_but_never_export_answer(self):
        with tempfile.TemporaryDirectory()as d:
            base=Path(d);parent=base/'parent';parent.mkdir()
            (parent/'comparison.json').write_text(json.dumps(dict(complete192=True,optimizer_updates=0)))
            control=[dict(id='b'+str(i),family='ordinary',folder='/original/basic/'+str(i),user_turns=[dict(text='Question'+str(i))],input_sha256='a'*64)for i in range(12)]
            dev=[dict(id='d'+str(i),folder='/original/dev/'+str(i),input_sha256='b'*64)for i in range(64)]
            refs=[dict(id=r['id'],question='Problem'+str(i),response='SECRET_GOLD',numeric_target='999')for i,r in enumerate(dev)]
            for name,value in [('control-cases.json',control),('dev-cases.json',dev),('scoring-only-reference.json',refs)]:
                (parent/name).write_text(json.dumps(value))
            result=prepare(parent,base/'probe','a'*40);rows=json.loads((base/'probe/questions.json').read_text())
            self.assertEqual(result['required_new_outputs'],304);self.assertEqual(len(rows),76)
            self.assertNotIn('SECRET_GOLD',json.dumps(rows));self.assertNotIn('numeric_target',json.dumps(rows))
            self.assertEqual(rows[12]['question'],'Problem0')
            with self.assertRaises(ValueError):prepare(parent,base/'probe','a'*40)
    def test_supplied_question_context_does_not_claim_ASR_or_invent_times(self):
        row=supplied_context('Question')(0,'Untrusted answer999')[0]
        self.assertEqual(row['text'],'Question');self.assertEqual(row['role'],'user')
        self.assertNotIn('start_s',row);self.assertNotIn('999',json.dumps(row))
