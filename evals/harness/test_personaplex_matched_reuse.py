import json
from pathlib import Path
import tempfile
import unittest
from training.personaplex import matched_reuse as r
from evals.harness import test_personaplex_matched_native as fixtures


class ReuseProofs(unittest.TestCase):
    def fixture(self,root):
        _,entries,_=fixtures.PairedAdmission().packet(root)
        for row in entries:
            base,cap=row['base'],row['capacity']
            for kind in ('admission','restore'):
                old=Path(row[kind]);v=json.loads(old.read_text());v['source']=r.ORIGIN
                if kind=='admission':v['data_report_sha256']=r.DATA_SHA
                local=root/base/cap/('admit' if kind=='admission' else 'restore')/(kind+'.json')
                local.parent.mkdir(parents=True,exist_ok=True);local.write_text(json.dumps(v))
                term=json.loads(Path(row[kind+'_terminal']).read_text());term['source']=r.ORIGIN
                terminal=root/'execution-8229989'/(base+'-'+cap+'-'+kind+'-terminal.json')
                terminal.parent.mkdir(exist_ok=True);terminal.write_text(json.dumps(term))
                row[kind]=str(r.ORIGINAL_ROOT/local.relative_to(root));row[kind+'_sha256']=r.digest(local)
                row[kind+'_terminal']=str(r.ORIGINAL_ROOT/terminal.relative_to(root))
        path=root/'joint-admission.json'
        path.write_text(json.dumps(dict(source=r.ORIGIN,data_report_sha256=r.DATA_SHA,entries=entries)))
        return entries,path

    def test_reuses_all_actual_proofs_without_claiming_new_training_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,p=self.fixture(root);report=r.preflight(root,r.digest(p))
            self.assertEqual(len(report['lineages']),4)
            self.assertEqual(report['future_GPU_seconds_with6h_reserve'],36000.)
            self.assertFalse(report['training_or_runtime_extension_authorized'])
            self.assertFalse(report['model_executed'])
            self.assertEqual(report['original_source'],r.ORIGIN)

    def test_modified_proof_and_failed_terminal_cannot_acquire_reuse_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);entries,p=self.fixture(root);row=entries[0]
            admission=root/Path(row['admission']).relative_to(r.ORIGINAL_ROOT)
            admission.write_text(admission.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'successful native'):r.preflight(root,r.digest(p))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);entries,p=self.fixture(root)
            terminal=root/Path(entries[0]['restore_terminal']).relative_to(r.ORIGINAL_ROOT)
            v=json.loads(terminal.read_text());v['actual_exit_code']=143;terminal.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'successful native'):r.preflight(root,r.digest(p))

    def test_three_arms_or_source_relabel_or_namespace_escape_rejected(self):
        for mutation in ('subset','source','escape'):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);_,p=self.fixture(root);value=json.loads(p.read_text())
                if mutation=='subset':value['entries'].pop()
                elif mutation=='source':value['source']='new-producer'
                else:value['entries'][0]['restore_terminal']='/etc/passwd'
                p.write_text(json.dumps(value))
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):r.preflight(root,r.digest(p))

    def test_checkpoint_swap_rejected_even_after_file_and_packet_rehash(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);entries,p=self.fixture(root);row=entries[0]
            file=root/Path(row['restore']).relative_to(r.ORIGINAL_ROOT)
            value=json.loads(file.read_text());value['checkpoint_sha256']='different-state';file.write_text(json.dumps(value))
            row['restore_sha256']=r.digest(file);packet=json.loads(p.read_text());packet['entries']=entries;p.write_text(json.dumps(packet))
            with self.assertRaisesRegex(ValueError,'own exact independent'):r.preflight(root,r.digest(p))


class AllocationCarry(unittest.TestCase):
    expected={'8216374':dict(state='COMPLETED',elapsed=327,GPUs=1),'8229989':dict(state='FAILED',elapsed=951,GPUs=2)}
    actual='8216374|brics.u6ada|COMPLETED|327|cpu=72,gres/gpu=1|\n8229989|brics.u6ada|FAILED|951|cpu=72,gres/gpu=2|\n'

    def test_all_actual_allocations_carried_without_counting_step_rows(self):
        self.assertEqual(r.actual_usage(self.actual+'8229989.batch||FAILED|951||\n',self.expected),2229)

    def test_missing_duplicate_changed_or_unlisted_allocation_rejected(self):
        for text in (self.actual.splitlines()[0],self.actual+self.actual,self.actual.replace('|951|','|950|'),
                     self.actual.replace('gres/gpu=2','gres/gpu=1'),self.actual+'8229999|brics.u6ada|FAILED|2|gres/gpu=2|\n'):
            with self.subTest(text=text),self.assertRaises(ValueError):r.actual_usage(text,self.expected)


if __name__=='__main__':unittest.main()
