import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from training.personaplex.action_pair_host import execute_pairs, pair_command
from training.personaplex import expanded_guard as g


class ActionPairHostTest(unittest.TestCase):
    def test_fit_backup_precedes_fresh_evaluation_and_next_arm(self):
        events = []
        def run(mode,stage):
            events.append(('run',mode,stage))
        def backup(mode,stage,updates):
            events.append(('backup',mode,stage,updates))
            return {'passed':True}
        receipts = execute_pairs(run,backup)
        self.assertEqual(len(receipts),4)
        self.assertEqual(events[:4],[('run','readout','fit'),('backup','readout','fit',128),('run','readout','evaluate'),('backup','readout','evaluate',0)])
        self.assertEqual(events[4],('run','bias','fit'))

    def test_backup_failure_prevents_evaluation_and_next_arm(self):
        events = []
        def backup(*args):
            raise ValueError('hash mismatch')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            execute_pairs(lambda *args:events.append(args),backup)
        self.assertEqual(events,[('readout','fit')])

    def test_failed_restore_is_not_retried_and_next_arm_does_not_start(self):
        events = []
        def run(mode,stage):
            events.append((mode,stage))
            if stage == 'evaluate':
                raise ValueError('native restore failed')
        with self.assertRaisesRegex(ValueError,'restore failed'):
            execute_pairs(run,lambda *args:{'passed':True})
        self.assertEqual(events,[('readout','fit'),('readout','evaluate')])

    def test_original_caches_states_and_inputs_are_read_only(self):
        with patch('training.personaplex.action_pair_host.feature_command',return_value=['apptainer','--pwd','/work']):
            cmd = pair_command(Path('/work'),{},Path('/backup'),Path('/inputs'))
        mounts = [cmd[i+1] for i,v in enumerate(cmd) if v == '--bind']
        self.assertGreaterEqual(len(mounts),5)
        self.assertTrue(all(v.endswith(':ro') for v in mounts))
        for required in ('duplex-native-feature-backups-v1','native-feature-restore-20261117-v1/cache','native-restore-only-20261117-v1','native-expanded1072-phase-v1','/inputs'):
            self.assertTrue(any(required in v for v in mounts))

    def test_paired_container_binds_the_existing_offline_asr_cache(self):
        with patch('training.personaplex.action_pair_host.feature_command',return_value=['apptainer','--pwd','/work']) as command:
            pair_command(Path('/work'),{},Path('/backup'),Path('/inputs'))
        self.assertEqual(command.call_args.args[2].get('HF_HOME'),'/projects/u6ada/duplex-live/hf-cache')

    def test_prepared_scope_cannot_admit_and_zero_update_scope_cannot_mix(self):
        lease = dict(provider='Slurm',resource_key=g.KEY,account=g.ACCOUNT,authorization_sha256=g.sha(g.AUTH),GPU_count=1,job_id='123',node='node',source='source',start_unix=0.,stop_unix=43200.,prior_award_GPU_seconds=0,other_reserved_GPU_seconds=0,paired_authorization_sha256=g.sha(g.PAIR_AUTH))
        scope = json.loads(g.PAIR_AUTH.read_text())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'scope.json'
            scope['status'] = 'PREPARED_NOT_AUTHORIZED'
            path.write_text(json.dumps(scope))
            with patch.object(g,'PAIR_AUTH',path):
                prepared = dict(lease,paired_authorization_sha256=g.sha(path))
                with self.assertRaisesRegex(ValueError,'authorized paired'):
                    g.resource(prepared)
                scope['status'] = 'OWNER_AUTHORIZED_PAIRED_RET'
                path.write_text(json.dumps(scope))
                admitted = dict(lease,paired_authorization_sha256=g.sha(path))
                g.resource(admitted)
                with self.assertRaises(ValueError):
                    g.resource(dict(admitted,stop_unix=43201.))
                with self.assertRaises(ValueError):
                    g.resource(dict(admitted,feature_authorization_sha256=g.sha(g.FEATURE_AUTH)))
                with self.assertRaises(ValueError):
                    g.resource(dict(admitted,GPU_count=4))
