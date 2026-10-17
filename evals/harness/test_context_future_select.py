import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class FutureSelectionTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.case={'index':0,'job':'1','teacher':'teacher-1','scenario':'train-1','expected_kind':'simple'}
        self.mapping={'schema':'dl-context-actual-capture-map-v1','training_approved':False,
            'collector_commit':'a'*40,'cases':[self.case]}
        self.plan={'teacher':{'id':'teacher-1'},'training_approved':False,
            'actual_history':'/unused/context-spanning-probe-1','target_kind':'ordinary_answer',
            'ordinary_answer_without_backend_context':True,'native_response_boundary':True,
            'native_response_boundary_version':2,'response_boundary_plan_code_sha256':digest(
                Path(__file__).parents[1]/'context_response_boundary.py')}

    def run_selector(self,explicit=True):
        (self.root/'map.json').write_text(json.dumps(self.mapping))
        (self.root/'plan-0.json').write_text(json.dumps(self.plan))
        row={k:self.case[k] for k in ('index','job','teacher','scenario')}
        row.update(kind='simple',eligible_for_strict_early_capture=True,raw_history_validated=True,
            training_approved=False,ordinary_response_boundary=True,
            plan_file='plan-0.json',plan_sha256=digest(self.root/'plan-0.json'))
        report={'schema':'dl-context-capture-eligibility-v1','training_approved':False,
            'labels_created':False,'heldout_used':False,'optimizer_steps':0,
            'ordinary_response_boundary':True,'capture_map_sha256':digest(self.root/'map.json'),'cases':[row]}
        (self.root/'eligibility.json').write_text(json.dumps(report))
        cmd=[sys.executable,'-m','evals.context_future_select','--eligibility',str(self.root/'eligibility.json'),
            '--sha256',digest(self.root/'eligibility.json'),'--map',str(self.root/'map.json'),'--index','0']
        if explicit:cmd.append('--ordinary-response-boundary')
        return subprocess.run(cmd,capture_output=True,text=True)

    def test_current_explicit_plan_is_selected(self):
        r=self.run_selector();self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(r.stdout.splitlines(),['1','a'*40,'teacher-1'])

    def test_rejected_duplicate_cue_version_cannot_be_reinterpreted(self):
        self.plan['native_response_boundary_version']=1
        self.assertNotEqual(self.run_selector().returncode,0)

    def test_changed_builder_code_requires_new_cpu_review(self):
        self.plan['response_boundary_plan_code_sha256']='0'*64
        self.assertNotEqual(self.run_selector().returncode,0)

    def test_ordinary_composition_cannot_be_selected_implicitly(self):
        self.assertNotEqual(self.run_selector(explicit=False).returncode,0)


if __name__=='__main__':unittest.main()
