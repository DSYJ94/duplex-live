import copy,unittest
from training.personaplex.adaptive_policy import Policy,SCHEMA,decide,program


def request(policy=Policy(),**changes):
    value=dict(schema=SCHEMA,policy_sha256=policy.identity,boundary=49,resume_step=49,
        data_sha256='a'*64,checkpoint_sha256='b'*64,backup_sha256='c'*64,dev_sha256='d'*64,
        previous_dev_sha256='e'*64,created_unix=100.,current_multiplier=1.,backup_complete=True,
        coverage=dict(new_train=1008,retained=323,dev=64),previous_dev=dict(text=2.,audio=3.),dev=dict(text=2.,audio=3.),
        retention=dict(previous=dict(kl=.01,hidden=.02),current=dict(kl=.01,hidden=.02)))
    value.update(changes);return value

class AdaptivePolicy(unittest.TestCase):
    def test_real_regression_adapts_and_preserves_request(self):
        r=request(dev=dict(text=2.2,audio=3.));before=copy.deepcopy(r);d=decide(r,now=100.)
        self.assertEqual((d['action'],d['multiplier']),('adapt',.75));self.assertEqual(r,before)
    def test_improvement_requires_retention_and_pinned_native_guard_for_rate_above_one(self):
        r=request(dev=dict(text=1.8,audio=2.7),behaviour=dict(qualified_DEV=True,passed=True,report_sha256='f'*64))
        self.assertEqual(decide(r,now=100.)['action'],'keep')
        policy=Policy(behaviour_admission_sha256='f'*64);r['policy_sha256']=policy.identity
        self.assertEqual(decide(r,policy=policy,now=100.)['multiplier'],1.1)
        r['retention']['current']['hidden']=.2;self.assertEqual(decide(r,policy=policy,now=100.)['action'],'keep')
    def test_recovery_to_previously_admitted_rate_needs_stable_retention(self):
        d=decide(request(current_multiplier=.75,dev=dict(text=1.8,audio=2.7)),now=100.)
        self.assertEqual(d['multiplier'],1.)
    def test_missing_noisy_or_incomplete_data_never_forces_adapt(self):
        for change in ({'dev':None},{'coverage':dict(new_train=1007,retained=323,dev=64)},
                       {'backup_complete':False},{'dev':dict(text=2.001,audio=3.001)}):
            self.assertIn(decide(request(**change),now=100.)['action'],('keep','defer'))
    def test_conflicting_dev_results_reduce_rate_instead_of_averaging_away_failure(self):
        self.assertEqual(decide(request(dev=dict(text=1.,audio=4.)),now=100.)['multiplier'],.75)
    def test_stale_fold_leak_bad_boundary_and_unknown_program_are_rejected(self):
        for change in ({'created_unix':-21.},{'created_unix':101.},{'boundary':32},{'checkpoint_sha256':'bad'},
                       {'current_multiplier':2.},{'test_answers':'forbidden'},{'resume_step':98}):
            with self.assertRaises(ValueError):decide(request(**change),now=100.)
        with self.assertRaises(ValueError):program(2.)
    def test_final_boundary_does_not_authorize_new_training(self):
        self.assertEqual(decide(request(boundary=196,resume_step=196),now=100.)['action'],'next_phase')
    def test_failed_behaviour_blocks_rate_recovery(self):
        result=decide(request(current_multiplier=.75,dev=dict(text=1.,audio=2.),behaviour=dict(qualified_DEV=True,passed=False)),now=100.)
        self.assertEqual(result['action'],'keep')
    def test_invalid_policy_is_rejected(self):
        for fields in ({'epoch_updates':0},{'total_updates':197},{'absolute_noise':float('nan')},{'behaviour_admission_sha256':'bad'}):
            with self.assertRaises(ValueError):Policy(**fields)
    def test_state_can_decrease_further_without_forcing_zero_learning(self):
        self.assertEqual(decide(request(current_multiplier=.75,dev=dict(text=3.,audio=3.)),now=100.)['multiplier'],.5)
        self.assertEqual(decide(request(current_multiplier=.5,dev=dict(text=3.,audio=3.)),now=100.)['action'],'keep')

if __name__=='__main__':unittest.main()
