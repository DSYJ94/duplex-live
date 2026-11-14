import json,tempfile,unittest
from pathlib import Path
from training.personaplex.expanded_codec import inputs,measured_events

class CodecAdmissionTest(unittest.TestCase):
    def test_measured_zero_width_words_are_not_dropped_or_stretched(self):
        events=[([5],(1.,1.),'SPEAKER_MAIN'),([6],(1.1,1.3),'SPEAKER_MAIN')]
        self.assertEqual(measured_events(events),events)
        with self.assertRaises(ValueError):measured_events([([5],(1.1,1.),'SPEAKER_MAIN')])

    def test_dev_cannot_be_relabelled_or_replaced_by_duplicate_paths(self):
        rows=[dict(id=str(i),path=str(i)+'.wav',fold='TRAIN' if i<1008 else 'DEV')for i in range(1072)]
        for defect in ['fold','path']:
            changed=[dict(r)for r in rows]
            if defect=='fold':changed[-1]['fold']='TRAIN'
            else:changed[-1]['path']=changed[0]['path']
            manifest=dict(rows=changed,complete1072=True,TRAIN=1008,DEV=64,optimizer_updates=0,native_training_approved=False,source_trimming=False)
            with tempfile.TemporaryDirectory()as folder:
                path=Path(folder);(path/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):inputs(path)
