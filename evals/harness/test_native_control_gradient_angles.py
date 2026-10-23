import copy
import math
import unittest
from evals.native_control_gradient_angles import cosine_from_norms,analyze


class TeacherGradientGeometry(unittest.TestCase):
    def test_norm_identity_matches_direct_vector_dot_products(self):
        for a,d in [([1.,2.],[-.1,.3]),([1.,0.],[0.,1.]),([1.,0.],[-1.,0.])]:
            norm=lambda x:math.sqrt(sum(v*v for v in x))
            total=[x+y for x,y in zip(a,d)]
            actual=cosine_from_norms(norm(a),norm(total),norm(d))
            self.assertAlmostEqual(actual,sum(x*y for x,y in zip(a,d))/(norm(a)*norm(d)),places=12)
        with self.assertRaises(ValueError):cosine_from_norms(1.,4.,1.)
        with self.assertRaises(ValueError):cosine_from_norms(float('nan'),1.,1.)

    def test_partial_case_geometry_cannot_masquerade_as_complete_approval(self):
        def gradient(norm):return dict(combined_gradient_norm=norm,all26_gradients_present_and_finite=True,frozen_parameter_gradients=0)
        row=dict(id='case',family='no-release',passed=True,exact_native_streamed_text_outputs=True,
                 exact_native_acoustic_outputs=True,text_gradient=gradient(1.),
                 combined_gradient=dict(gradient(math.sqrt(2)),acoustic_text_head_gradient_delta_exactly_zero=True,acoustic_temporal_gradient_delta_norm=1.))
        r=dict(schema='dl-native-control40-gradient-readiness-v1',source_commit='source',cases=[row],optimizer_updates=0,
               checkpoint_written=False,trainable_parameters=1894400,complete40=False,passed=False)
        with self.assertRaises(ValueError):analyze(r)
        d=analyze(r,allow_partial_diagnostic=True)
        self.assertTrue(d['partial_diagnostic_only']);self.assertFalse(d['training_approved'])
        self.assertFalse(d['original154_or_old49_gradient_conflict_measured'])
        bad=copy.deepcopy(r);bad['cases'][0]['combined_gradient']['acoustic_text_head_gradient_delta_exactly_zero']=False
        with self.assertRaises(ValueError):analyze(bad,allow_partial_diagnostic=True)


if __name__=='__main__':unittest.main()
