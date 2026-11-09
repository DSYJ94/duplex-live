from pathlib import Path
import unittest
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


if __name__=='__main__':unittest.main()
