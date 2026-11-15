import subprocess,tempfile,unittest
from pathlib import Path
from training.personaplex.expanded_provenance import capture_provenance,CAPTURE_FILES

class CaptureReuse(unittest.TestCase):
    def test_preserves_producer_and_rejects_capture_code_change(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder)
            def git(*args):return subprocess.check_output(['git','-C',folder,*args],stderr=subprocess.DEVNULL).decode().strip()
            git('init')
            for name in CAPTURE_FILES:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('original\n')
            git('add','.');git('-c','user.name=Test','-c','user.email=test@example.com','commit','-m','capture');producer=git('rev-parse','HEAD')
            (root/'training-only.txt').write_text('new learner');git('add','.');git('-c','user.name=Test','-c','user.email=test@example.com','commit','-m','learner');consumer=git('rev-parse','HEAD')
            report={'source':producer};proof=capture_provenance(report,consumer,root)
            self.assertEqual(report['source'],producer);self.assertEqual(proof['original_capture_source'],producer)
            self.assertNotEqual(producer,consumer)
            (root/CAPTURE_FILES[0]).write_text('changed\n')
            with self.assertRaisesRegex(ValueError,'requalification'):capture_provenance(report,consumer,root)

if __name__=='__main__':unittest.main()
