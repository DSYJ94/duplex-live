import os,unittest
from pathlib import Path
from unittest.mock import patch
from training.personaplex.native_feature_host import feature_command

class NativeFeatureHostTest(unittest.TestCase):
    def test_single_allocated_gpu_preserves_native_runtime_and_never_forces_gpu_zero(self):
        stub=['apptainer','CUDA_VISIBLE_DEVICES=0,1,2,3','python','-m']
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='3'),patch('training.personaplex.native_feature_host.container_command',return_value=stub.copy())as build:
            result=feature_command(Path('/work'),True,{},Path('/backup'));self.assertIn('CUDA_VISIBLE_DEVICES=3',result);self.assertNotIn('PYTHONPATH',build.call_args.kwargs['environment'])
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),patch('training.personaplex.native_feature_host.container_command',return_value=stub.copy()):
            with self.assertRaises(ValueError):feature_command(Path('/work'),True,{},Path('/backup'))
