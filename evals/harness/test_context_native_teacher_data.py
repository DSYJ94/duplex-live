import json
from pathlib import Path
import tempfile
import unittest
from evals.context_native_teacher_data import load
from evals.moshi_model_download import sha256


class NativeTeacherDataTest(unittest.TestCase):
    def test_partial_or_duplicate_inventory_cannot_enter_broader_objective(self):
        examples=[{'record':dict(id=str(i),kind=kind,source_model='gpt-live-1',source_split='train')}
            for i,kind in enumerate(['private','simple','clarify','cancel','private'])]
        for ids in (['0','1','2','3'],['0','1','2','3','3']):
            with tempfile.TemporaryDirectory() as name:
                p=Path(name)/'approval.json'
                p.write_text(json.dumps(dict(schema='dl-context-native-teacher-approval-set-v1',
                    passed=True,optimizer_steps=0,training_approved=False,
                    code_sha256=sha256(Path(__file__).parents[1]/'context_teacher_native_approval.py'),
                    cases=[{'teacher':{'id':i}} for i in ids])))
                with self.assertRaises(RuntimeError):load(p,Path('unused'),Path('unused'),Path('unused'),examples,None,None)

    def test_failed_approval_is_not_an_optimizer_preflight(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)/'approval.json';p.write_text(json.dumps(dict(
                schema='dl-context-native-teacher-approval-set-v1',passed=False)))
            with self.assertRaises(RuntimeError):load(p,Path('unused'),Path('unused'),Path('unused'),[],None,None)


if __name__=='__main__':unittest.main()
