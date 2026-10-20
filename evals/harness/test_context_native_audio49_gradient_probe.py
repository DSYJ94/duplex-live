import unittest
from evals.context_native_audio49_gradient_probe import schedule
from evals.harness import test_context_native_audio49_training as pilot_tests
from evals.context_native_audio49_training import validate_configuration

class CompleteGradientSchedule(unittest.TestCase):
    def test_current_complete24_and_audio49_weighting(self):
        kinds=['k'+str(i%8) for i in range(28)]
        a,b=schedule(kinds,['private']*8+['simple']*8+['clarify']*5+['cancel']*4,['private']*8+['simple']*8+['clarify']*4+['cancel']*4)
        for group in ['actual','reference','old25','new24','audio49']:self.assertAlmostEqual(sum(b[group]),1.)
        self.assertTrue(all(b['new24']));self.assertTrue(all(b['audio49']))
        self.assertEqual([i for i,w in enumerate(a['audio49']) if w],list(range(0,49,8)))
        self.assertAlmostEqual(sum(b['word']),64.);self.assertAlmostEqual(sum(b['clock']),64.)
    def test_probe_cannot_enter_optimizer_or_reload(self):
        a=pilot_tests.Audio49PilotPolicy().args();a.native_audio49_gradient_probe=True
        with self.assertRaises(ValueError):validate_configuration(a)
        a.native_audio49_preflight_only=True;a.native_audio49_preflight=None
        self.assertTrue(validate_configuration(a))
        a.reload=True
        with self.assertRaises(ValueError):validate_configuration(a)
