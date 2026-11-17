import copy
import unittest
from collections import Counter

from training.personaplex.matched_schedule import SIZES
from training.personaplex.mixed_schedule import build, next_window
from training.personaplex.text_rehearsal import TextSpec


class MixedScheduleTest(unittest.TestCase):
    def setUp(self):
        self.groups = {g: [dict(id=g+str(i), fold='TRAIN', kind='same')
                          for i in range(n)] for g, n in SIZES.items()}
        self.speech = [dict(id='speech-'+str(i), fold='TRAIN' if i < 1008 else 'DEV')
                       for i in range(1072)]
        self.documents = [dict(id='doc-'+str(i), tokens=[5]*(i % 3+1),
            mask=[True]*(i % 3+1), fold='TRAIN', semantic_qualified=True,
            leakage_qualified=True, qualification_sha256='a'*64,
            tokenizer_sha256='b'*64, source_sha256='c'*64) for i in range(53)]
        self.spec = TextSpec(2, 32, 0, 3, 8, frozenset((0, 1, 2, 3, 4)))
        self.plan = build(self.groups, self.speech, self.documents, self.spec,
            epochs=2, language_budget=10, coefficient=.3,
            tokenizer_sha256='b'*64, qualification_sha256='a'*64, seed=42)
        self.identity = dict(mixed_schedule_sha256=self.plan['sha256'],
                             language_training=self.plan['language']['identity'])

    def window(self, step, rank=0, plan=None, identity=None):
        return next_window(plan or self.plan, self.documents, self.spec, rank=rank,
            restored_step=step, identity=identity or self.identity)

    def test_complete_paired_coverage_and_resume(self):
        for epoch in range(2):
            speech, language = Counter(), Counter()
            for step in range(epoch*49, (epoch+1)*49):
                for rank in range(4):
                    w = self.window(step, rank)
                    speech.update(r['id'] for r in w['speech_slots'] if r['valid'])
                    language.update(i for i, pad in zip(w['language_window']['document_ids'],
                        w['language_window']['padding_only']) if not pad)
                    self.assertEqual(w['epoch_boundary'], step % 49 == 48)
            self.assertEqual(speech, Counter(r['id'] for r in self.speech if r['fold']=='TRAIN'))
            self.assertEqual(language, Counter(r['id'] for r in self.documents))
        self.assertEqual(self.window(50)['language_window']['step'], 50)
        for step in (-1, 98, True):
            with self.assertRaises(ValueError): self.window(step)

    def test_changed_schedule_or_checkpoint_identity_rejected(self):
        changed = copy.deepcopy(self.plan)
        changed['speech']['schedule'][0][0]['new_coefficient'] *= 2
        with self.assertRaises(ValueError): self.window(0, plan=changed)
        wrong = dict(self.identity, mixed_schedule_sha256='d'*64)
        with self.assertRaises(ValueError): self.window(0, identity=wrong)
        changed_docs = copy.deepcopy(self.documents)
        changed_docs[0]['tokens'] = [6]
        with self.assertRaises(ValueError):
            next_window(self.plan, changed_docs, self.spec, rank=0,
                        restored_step=0, identity=self.identity)


if __name__ == '__main__': unittest.main()
