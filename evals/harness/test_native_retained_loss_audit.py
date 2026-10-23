import copy
import unittest

from evals.native_retained_loss_audit import audit, weighted


def fixture():
    groups = {}
    for group, size in dict(actual=28, reference=4, old25=25, new24=24,
                            word=10, clock=14, audio49=49).items():
        mass = 64. if group in ('word', 'clock') else 1.
        groups[group] = [dict(id=group + str(i), weight=mass/size, loss=1., kind='fixture',
                              kl=1., hidden_mse=0., scenario=group+str(i)) for i in range(size)]
    steps = []
    for i in range(8):
        steps.append(dict(step=i+1, complete_balanced_microbatches=copy.deepcopy(groups),
            gradient_norm_before_clip=2., native_stream_teacher_microbatch=[
                dict(id='stream'+str(n), weight=1., balanced_spoken_clock_ce=1.)
                for n in range(i, 49, 8)]))
    metrics = [dict(scenario=r['scenario'], nll=1.) for r in groups['actual']]
    return dict(updates_this_run=8, adam_parameter_steps_after=[528],
                warm_start=dict(optimizer_steps=520), steps=steps,
                initial_stage_metrics=metrics, final_metrics=copy.deepcopy(metrics))


class RetainedLossAuditTests(unittest.TestCase):
    def test_full_accounting_separates_rotating_loss_and_preupdate_clock(self):
        result = audit(fixture())
        self.assertEqual(result['retained154_weighted_sum_before_first_update'], 133.)
        self.assertEqual(result['curves'][-1]['optimizer_updates_already_applied_when_losses_computed'], 7)
        self.assertFalse(result['rotating_stream_loss_is_same_examples_convergence_curve'])
        self.assertFalse(result['complete_post_update8_objective_measured'])
        self.assertTrue(result['all_recorded_steps_clip_gradients'])
        self.assertEqual(result['actual28_initial_and_post_eighth_update']['weighted_delta'], 0.)

    def test_changed_membership_cannot_be_presented_as_comparable_curve(self):
        report = fixture(); report['steps'][-1]['complete_balanced_microbatches']['actual'][0]['id'] = 'different'
        with self.assertRaises(ValueError): audit(report)

    def test_partial_steps_or_missing_stream_or_changed_formula_reject(self):
        for mutation in ('steps', 'stream', 'formula', 'final_metrics', 'initial_criterion'):
            report = fixture()
            if mutation == 'steps': report['steps'].pop()
            if mutation == 'stream': report['steps'][-1]['native_stream_teacher_microbatch'].pop()
            if mutation == 'formula': report['steps'][-1]['complete_balanced_microbatches']['word'][0]['loss'] = 2.
            if mutation == 'final_metrics': report['final_metrics'].pop()
            if mutation == 'initial_criterion': report['initial_stage_metrics'][0]['optimization_nll'] = 2.
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): audit(report)

    def test_nonfinite_loss_is_not_silently_averaged(self):
        with self.assertRaises(ValueError): weighted([dict(weight=1., loss=float('nan'))])


if __name__ == '__main__': unittest.main()
