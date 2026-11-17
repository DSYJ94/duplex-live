import unittest
from training.personaplex.reasoning_family_audit import audit_records


class ReasoningFamilyAuditTest(unittest.TestCase):
    def test_template_overlap_is_distinct_from_exact_question_overlap(self):
        rows = [dict(id='train', fold='TRAIN', question='A pump moves 3 litres for 4 minutes.'),
                dict(id='dev', fold='DEV', question='A pump moves 7 litres for 8 minutes.'),
                dict(id='different', fold='DEV', question='Explain electricity.')]
        report = audit_records(rows)
        self.assertEqual(report['methods']['literal']['crossing_groups'], 0)
        template = report['methods']['numeric_template']
        self.assertEqual((template['TRAIN_affected'], template['DEV_affected']), (1, 1))
        self.assertEqual(template['cross_fold_groups'][0]['TRAIN_ids'], ['train'])
        self.assertEqual(template['cross_fold_groups'][0]['DEV_ids'], ['dev'])
        self.assertEqual([r['fold'] for r in rows], ['TRAIN', 'DEV', 'DEV'])
        self.assertFalse(report['folds_changed'])
        self.assertFalse(report['training_admitted'])

    def test_exact_overlap_duplicate_ids_and_absent_words(self):
        rows = [dict(id='a', fold='TRAIN', question='What is voltage?'),
                dict(id='b', fold='DEV', question='WHAT is voltage!')]
        self.assertEqual(audit_records(rows)['methods']['literal']['crossing_groups'], 1)
        rows[1]['id'] = 'a'
        with self.assertRaises(ValueError): audit_records(rows)
        rows[1]['id'] = 'b'; rows[1]['question'] = ''
        with self.assertRaises(ValueError): audit_records(rows)


if __name__ == '__main__': unittest.main()
