from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from training.personaplex.matched_paths import command,legacy_mounts


class DeviceAndDataBoundary(unittest.TestCase):
    def test_cpu_preparation_cannot_accidentally_enable_gpu_or_custom_devices(self):
        cmd=command(Path('/projects/u6ada/duplex-live/source-v1'))
        self.assertNotIn('--nv',cmd);self.assertNotIn('--containall',cmd)
        self.assertIn('CUDA_VISIBLE_DEVICES=-1',cmd)
        self.assertFalse(any('/dev/' in x for x in cmd))
        with self.assertRaises(ValueError):command(Path('/work'),environment={'CUDA_VISIBLE_DEVICES':'0'})

    def test_gpu_worker_gets_only_its_explicit_scheduler_selection(self):
        cmd=command(Path('/work'),gpu='GPU-fixture',environment={'SLURM_JOB_ID':'123'})
        self.assertIn('--nv',cmd);self.assertIn('--containall',cmd)
        self.assertIn('CUDA_VISIBLE_DEVICES=GPU-fixture',cmd);self.assertIn('SLURM_JOB_ID=123',cmd)

    def test_all_original_virtual_paths_are_read_only(self):
        mounts=legacy_mounts()
        self.assertEqual(len(mounts),9)
        self.assertTrue(all(mode=='ro' for _,_,mode in mounts[2:]))
        self.assertEqual({target for _,target,_ in mounts if target.startswith('/workspace/')},
            {'/workspace/migration','/workspace/control40-vendor-moshi-rag','/workspace/control40-vendor-moshi-finetune'})

    def test_cpu_restore_precedes_container_mounts_and_never_enables_gpu(self):
        from training.personaplex import matched_cpu as c
        with tempfile.TemporaryDirectory() as root,patch.object(c,'BASE',Path(root)):
            stages=c.stages(Path(root)/'work','a'*40,Path(root)/'inputs')
            self.assertEqual([name for name,_ in stages],['restore-source','unit','temporary-inputs','legacy-inputs','prepare'])
            self.assertIn('training.personaplex.matched_snapshot',stages[0][1])
            self.assertNotIn('--nv',stages[0][1])
            self.assertTrue(all('CUDA_VISIBLE_DEVICES=-1' in args for _,args in stages[1:]))

    def test_cpu_slurm_guard_checks_actual_zero_gpu_account_and_remaining_time(self):
        from training.personaplex import matched_cpu as c
        from training.personaplex.matched_guard import stamp
        job='JobId=10 Account=brics.u6ada JobState=RUNNING AllocTRES=cpu=4,mem=16G NumNodes=1 NodeList=nid1 StartTime=2026-11-09T12:00:00 EndTime=2026-11-09T12:30:00'
        with patch.dict(c.os.environ,{'SLURM_JOB_ID':'10'},clear=True),patch.object(c.time,'time',return_value=stamp('2026-11-09T12:00:10')):
            with patch.object(c.subprocess,'check_output',return_value=job):self.assertEqual(c.cpu_allocation()['GPU_count'],0)
            for changed in (job.replace('AllocTRES=cpu=4','AllocTRES=gres/gpu=1,cpu=4'),
                job.replace('AllocTRES=cpu=4','AllocTRES=gres/gpu:gh200=1,cpu=4'),
                job.replace('brics.u6ada','wrong'),job.replace('12:30:00','12:30:01')):
                with patch.object(c.subprocess,'check_output',return_value=changed),self.assertRaises(ValueError):c.cpu_allocation()


if __name__=='__main__':unittest.main()
