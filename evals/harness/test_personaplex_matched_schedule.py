import copy
from collections import Counter
import unittest
from training.personaplex import matched_schedule as s


def groups():
    # Uneven kinds catch uniform-instead-of-balanced weighting errors.
    return {g: [dict(id=f'{g}:{i}', fold='TRAIN', kind='rare' if i == 0 else 'common')
                for i in range(n)] for g, n in s.SIZES.items()}


def check_complete(test):
    data = groups()
    for e in range(4):
        schedule = s.epoch(data, e)
        assert len(schedule) == 49 and all(len(step) == 11 for step in schedule)
        for group, rows in data.items():
            mass = Counter()
            for step in schedule:
                for item in step:
                    if item['group'] == group:
                        mass[item['id']] += item['weight'] / 49
            assert set(mass) == {r['id'] for r in rows}
            test.assertAlmostEqual(sum(mass.values()),64 if group in ('word','clock') else 1)
            if group in s.BALANCED:
                test.assertAlmostEqual(mass[rows[0]['id']],.5)
                test.assertAlmostEqual(sum(mass[r['id']] for r in rows[1:]),.5)
        assert s.audit(data, schedule)


def check_membership(test):
    data = groups()
    data['ordinary48'][0]['fold'] = 'HOLDOUT'
    with test.assertRaisesRegex(ValueError, 'held-out'):
        s.epoch(data, 0)
    data = groups()
    data['actual'].pop()
    with test.assertRaisesRegex(ValueError, 'membership'):
        s.epoch(data, 0)


def check_corruption(test):
    data = groups()
    schedule = s.epoch(data, 0)
    with test.assertRaisesRegex(ValueError, 'subset'):
        s.audit(data, schedule[:-1])
    for mutation in ('group','identity','mass'):
        changed = copy.deepcopy(schedule)
        if mutation == 'group':
            changed[0].pop()
        elif mutation == 'identity':
            changed[0][0]['id'] = 'another-component'
        else:
            changed[0][0]['weight'] *= .5
        with test.assertRaises(ValueError):
            s.audit(data, changed)


def check_pairing(test):
    data = groups()
    assert s.epoch(data, 2) == s.epoch(copy.deepcopy(data), 2)
    assert [x['id'] for x in s.epoch(data, 0)[0]] != [x['id'] for x in s.epoch(data, 1)[0]]
    for bad in (-1,4,True,1.5):
        with test.assertRaisesRegex(ValueError, 'four'):
            s.epoch(data, bad)


class MatchedSchedule(unittest.TestCase):
    test_complete_original_mass_all_four_epochs = check_complete
    test_heldout_and_missing_member_rejected = check_membership
    test_corrupted_exposure_rejected = check_corruption
    test_paired_exposure_and_no_extra_epoch = check_pairing


if __name__ == '__main__':
    unittest.main()
